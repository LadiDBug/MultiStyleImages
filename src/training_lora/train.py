"""Training LoRA Pixel Art per SD 1.5, con selezione automatica del device.

Riferimento: Hugging Face Diffusers, Apache-2.0,
examples/text_to_image/train_text_to_image_lora.py, tag v0.35.2.
Qui: dataset locale, TOML opzionale, validation loss con rumore fissato, checkpoint
con optimizer/scheduler/RNG e nessuna generazione o integrazione ControlNet.
Accelerate seleziona automaticamente il device disponibile (CUDA, MPS o CPU).
Su MPS/CPU si usa FP32; su CUDA si applica la mixed precision richiesta.

Esempi (dalla radice repository):
    python -m src.training_lora.train --check-data
    python -m src.training_lora.train --dataset-dir /kaggle/input/pixel-art/pixel_art_v1
    python -m src.training_lora.train --resume-from-checkpoint latest
    python -m src.training_lora.train --rank 4 --output-dir outputs/rq2/rank04

Dipendenze training: torch, diffusers>=0.35,<0.36, transformers>=4.44,<5,
accelerate>=0.26, peft>=0.17,<0.18, safetensors, huggingface_hub>=0.34,<1, Pillow, numpy.
Usare un ambiente con versioni compatibili e conservarle dal run_config.json.
--help, --print-config e --check-data non scaricano modelli e non addestrano.

TOML opzionale (i flag CLI prevalgono; percorsi relativi alla directory corrente):
    [model]
    id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision = "main"  # risolta a commit prima di caricare i componenti
    [data]
    dataset_dir = "data/style_pixel_art/v1"
    resolution = 512
    interpolation = "nearest"
    [lora]
    rank = 8
    alpha = 8
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    [training]
    max_train_steps = 1000
    learning_rate = 1e-4
    batch_size = 1
    gradient_accumulation_steps = 4
    mixed_precision = "fp16"
    seed = 42
    [output]
    dir = "outputs/rq2/pixel_art_rank08"
    checkpointing_steps = 250
    validation_steps = 250

I checkpoint NON sono test dello stile: la validation loss misura denoising.
Non usare foto di test per selezionare budget o checkpoint.
La riproducibilita dipende anche da hardware e versioni, non soltanto dal seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training_lora.dataset import PixelArtDataset, collate_batch

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: str = "main"
    dataset_dir: str = str(ROOT / "data/style_pixel_art/v1")
    train_dir: str | None = None
    validation_dir: str | None = None
    output_dir: str | None = None
    rank: int = 8
    alpha: int | None = None
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    resolution: int = 512
    interpolation: str = "nearest"
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "constant_with_warmup"
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    num_workers: int = 0
    seed: int = 42
    checkpointing_steps: int = 250
    validation_steps: int = 250

    def validate(self):
        for key in ["rank", "resolution", "batch_size", "gradient_accumulation_steps",
                    "max_train_steps", "checkpointing_steps"]:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} deve essere un intero positivo.")
        for key in ["warmup_steps", "validation_steps", "num_workers", "seed"]:
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} deve essere un intero non negativo.")
        if self.resolution % 8:
            raise ValueError("resolution deve essere multipla di 8.")
        if self.alpha is None:
            self.alpha = self.rank
        if type(self.alpha) is not int or self.alpha != self.rank:
            raise ValueError("Il protocollo RQ2 richiede alpha = rank (scaling alpha/r = 1).")
        for key in ["learning_rate", "max_grad_norm"]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} deve essere positivo e finito.")
        if not isinstance(self.weight_decay, (int, float)) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay deve essere non negativo e finito.")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision deve essere no, fp16 o bf16.")
        if self.interpolation not in {"nearest", "lanczos"}:
            raise ValueError("interpolation deve essere nearest o lanczos.")
        if self.lr_scheduler not in {"constant", "constant_with_warmup", "linear", "cosine"}:
            raise ValueError("lr_scheduler non supportato.")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing deve essere booleano.")
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules or any(
            not isinstance(x, str) or not x for x in self.target_modules
        ) or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules deve contenere nomi non vuoti e unici.")
        for key in ["model_id", "revision"]:
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} deve essere una stringa non vuota.")
        base = Path(self.dataset_dir).expanduser().resolve()
        self.dataset_dir = str(base)
        self.train_dir = str(Path(self.train_dir).expanduser().resolve()) if self.train_dir else str(base/"train")
        self.validation_dir = (str(Path(self.validation_dir).expanduser().resolve())
                               if self.validation_dir else str(base/"validation"))
        self.output_dir = (str(Path(self.output_dir).expanduser().resolve()) if self.output_dir
                           else str(ROOT/f"outputs/rq2/pixel_art_rank{self.rank:02d}"))
        for split in [self.train_dir, self.validation_dir]:
            if Path(self.output_dir).is_relative_to(Path(split)) or Path(split).is_relative_to(Path(self.output_dir)):
                raise ValueError("La directory output deve essere separata dagli split dei dati.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="TOML opzionale; nessuna modifica ai TOML esistenti")
    parser.add_argument("--check-data", action="store_true", help="Audit di entrambi gli split, senza modelli o training")
    parser.add_argument("--print-config", action="store_true", help="Stampa la configurazione effettiva ed esce")
    parser.add_argument("--resume-from-checkpoint", help="latest oppure percorso di un checkpoint completo di questo runner")
    for key in ["model_id", "revision", "dataset_dir", "train_dir", "validation_dir", "output_dir",
                "mixed_precision", "interpolation", "lr_scheduler"]:
        parser.add_argument("--"+key.replace("_", "-"), default=argparse.SUPPRESS)
    for key in ["rank", "alpha", "resolution", "batch_size", "gradient_accumulation_steps", "max_train_steps",
                "warmup_steps", "seed", "num_workers", "checkpointing_steps", "validation_steps"]:
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    for key in ["learning_rate", "weight_decay", "max_grad_norm"]:
        parser.add_argument("--"+key.replace("_", "-"), type=float, default=argparse.SUPPRESS)
    parser.add_argument("--target-modules", nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    values = {}
    sections = {
        "model": {"id": "model_id", "revision": "revision"},
        "data": {k:k for k in ["dataset_dir", "train_dir", "validation_dir", "resolution", "interpolation"]},
        "lora": {k:k for k in ["rank", "alpha", "target_modules"]},
        "training": {k:k for k in ["batch_size", "gradient_accumulation_steps", "max_train_steps", "learning_rate",
                                  "weight_decay", "lr_scheduler", "warmup_steps", "max_grad_norm", "mixed_precision",
                                  "gradient_checkpointing", "num_workers", "seed"]},
        "output": {"dir": "output_dir", "checkpointing_steps": "checkpointing_steps", "validation_steps": "validation_steps"},
    }
    if args.config:
        with args.config.open("rb") as stream:
            content = tomllib.load(stream)
        for section, entries in content.items():
            if section not in sections or not isinstance(entries, dict):
                parser.error(f"Sezione TOML non supportata: {section}")
            for key, value in entries.items():
                if key not in sections[section]:
                    parser.error(f"Chiave TOML non supportata: {section}.{key}")
                values[sections[section][key]] = value
    values.update({k:v for k,v in vars(args).items() if k in TrainingConfig.__dataclass_fields__})
    config = TrainingConfig(**values)
    try:
        config.validate()
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return args, config


def load_data(config, tokenizer=None):
    train = PixelArtDataset(config.train_dir, tokenizer, config.resolution, config.interpolation)
    validation = PixelArtDataset(config.validation_dir, tokenizer, config.resolution, config.interpolation)
    audits = {"train": train.audit(), "validation": validation.audit()}
    if set(audits["train"]["rgb_hashes"]) & set(audits["validation"]["rgb_hashes"]):
        raise ValueError("Overlap di immagini identiche fra train e validation.")
    for audit in audits.values():
        audit.pop("rgb_hashes")
    return train, validation, audits


class EpochSampler(Sampler):
    """Ordine riproducibile per epoca, indipendente dal RNG del rumore di training."""
    def __init__(self, dataset, seed):
        self.length, self.seed, self.epoch = len(dataset), seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        return iter(torch.randperm(self.length, generator=generator).tolist())

    def __len__(self):
        return self.length


def diffusion_target(scheduler, latents, noise, timesteps):
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return noise
    if kind == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"prediction_type non supportato: {kind}")


def batch_loss(batch, unet, vae, text_encoder, noise_scheduler, dtype, device, generator=None):
    with torch.no_grad():
        posterior = vae.encode(batch["pixel_values"].to(device=device, dtype=dtype)).latent_dist
        latents = posterior.sample(generator=generator) * vae.config.scaling_factor
        embeddings = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]
        # MPS usa un generatore CPU per la validation: rumore e timestep vengono
        # creati sul device del generatore, poi trasferiti insieme ai latenti.
        random_device = generator.device if generator is not None else device
        noise = torch.randn(latents.shape, device=random_device, dtype=latents.dtype,
                            generator=generator).to(device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=random_device,
                                  generator=generator).to(device=device, dtype=torch.long)
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        target = diffusion_target(noise_scheduler, latents, noise, timesteps)
    prediction = unet(noisy, timesteps, embeddings, return_dict=False)[0]
    return F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))


def validation_loss(loader, accelerator, unet, vae, text_encoder, scheduler, dtype, seed):
    was_training = unet.training
    unet.eval()
    generator_device = "cpu" if accelerator.device.type == "mps" else accelerator.device
    generator = torch.Generator(device=generator_device).manual_seed(seed+10000)
    total, count = 0.0, 0
    try:
        with torch.no_grad(), accelerator.autocast():
            for batch in loader:
                losses = batch_loss(batch, unet, vae, text_encoder, scheduler, dtype,
                                    accelerator.device, generator)
                total += losses.sum().item()
                count += len(losses)
    finally:
        unet.train(was_training)
    return total/count


def find_checkpoint(output: Path, requested: str | None) -> Path | None:
    if not requested:
        return None
    if requested == "latest":
        candidates = [p for p in output.glob("checkpoint-*") if (p/"progress.json").is_file()]
        if not candidates:
            raise ValueError(f"Nessun checkpoint completo in {output}")
        return max(candidates, key=lambda p: int(json.loads((p/"progress.json").read_text())["global_step"]))
    path = Path(requested).expanduser().resolve()
    if not (path/"progress.json").is_file():
        raise ValueError(f"Checkpoint incompleto o non supportato: {path}")
    return path


def check_training_dependencies():
    from packaging.requirements import Requirement
    problems = []
    for text in ["diffusers>=0.35,<0.36", "transformers>=4.44,<5", "accelerate>=0.26",
                 "peft>=0.17,<0.18", "huggingface_hub>=0.34,<1", "safetensors>=0.4"]:
        requirement = Requirement(text)
        try:
            installed = importlib.metadata.version(requirement.name)
            if installed not in requirement.specifier:
                problems.append(f"{requirement.name}=={installed}; richiesto {text}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"manca {text}")
    if problems:
        raise RuntimeError("Ambiente training da configurare: " + "; ".join(problems))


def effective_precision(requested: str, device: torch.device) -> str:
    """FP32 come percorso comune su CPU/MPS; mixed precision su CUDA."""
    if device.type != "cuda":
        return "no"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        return "fp16"
    return requested


def run_training(config, resume):
    # Import ritardati: audit/help funzionano anche senza PEFT o CUDA.
    check_training_dependencies()
    try:
        from accelerate import Accelerator
        from accelerate.state import PartialState
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import cast_training_params
        from diffusers.utils import convert_state_dict_to_diffusers
        from huggingface_hub import model_info
        from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file, save_file
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:
        raise RuntimeError("Dipendenze training mancanti/incompatibili. Vedere l'intestazione di train.py; "
                           f"dettaglio: {error}") from error
    # PartialState usa la selezione automatica di Accelerate, senza richiedere
    # CUDA. Rispetta anche l'eventuale richiesta CPU dell'ambiente Accelerate.
    force_cpu = os.environ.get("ACCELERATE_USE_CPU", "false").lower() in {"1", "true", "yes"}
    device = PartialState(cpu=force_cpu).device
    precision = effective_precision(config.mixed_precision, device)
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps,
                              cpu=force_cpu, mixed_precision=precision)
    if accelerator.num_processes != 1:
        raise ValueError("Questo runner richiede un solo processo per mantenere controllato il batch effettivo.")
    print(f"Device auto: {accelerator.device}; mixed precision effettiva: {precision}", flush=True)
    output = Path(config.output_dir)
    checkpoint = find_checkpoint(output, resume)
    if output.exists() and any(output.iterdir()) and not checkpoint:
        raise ValueError(f"Output non vuoto: {output}. Usare una nuova directory oppure --resume-from-checkpoint.")
    set_seed(config.seed)
    # Anche 'main' viene risolta una sola volta: tutti i componenti usano lo stesso commit.
    local_model = Path(config.model_id).is_dir()
    revision = None if local_model else model_info(config.model_id, revision=config.revision).sha
    kwargs = {} if local_model else {"revision": revision}
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer", **kwargs)
    train, validation, audits = load_data(config, tokenizer)
    signature_values = asdict(config)
    for key in ["dataset_dir", "train_dir", "validation_dir", "output_dir", "num_workers"]:
        signature_values.pop(key)
    signature_values.update(resolved_revision=revision, data=audits, effective_mixed_precision=precision)
    signature_values["revision"] = revision
    signature = hashlib.sha256(json.dumps(signature_values, sort_keys=True).encode()).hexdigest()
    progress = {"global_step": 0, "epoch": 0, "next_batch": 0, "signature": signature}
    if checkpoint:
        progress = json.loads((checkpoint/"progress.json").read_text())
        if progress["signature"] != signature:
            raise ValueError("Checkpoint incompatibile: dati, modello o configurazione di training differenti.")
        if progress["global_step"] >= config.max_train_steps:
            raise ValueError("Il checkpoint ha gia raggiunto max_train_steps; nessun nuovo training necessario.")
    if (output/"run_config.json").exists():
        previous = json.loads((output/"run_config.json").read_text())
        if previous.get("signature") != signature:
            raise ValueError("La directory output appartiene a un altro esperimento.")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler", **kwargs)
    if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("Il checkpoint usa un prediction_type non supportato.")
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder", **kwargs)
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae", **kwargs)
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet", **kwargs)
    for model in [vae, text_encoder, unet]:
        model.requires_grad_(False)
        model.to(accelerator.device, dtype=dtype)
    vae.eval()
    text_encoder.eval()
    unet.add_adapter(LoraConfig(r=config.rank, lora_alpha=config.alpha,
                                init_lora_weights="gaussian", target_modules=list(config.target_modules)))
    # Frozen backbone in mixed precision; pesi addestrabili FP32 per stabilita.
    cast_training_params(unet, dtype=torch.float32)
    parameters = [p for p in unet.parameters() if p.requires_grad]
    if not parameters or any(p.requires_grad and "lora_" not in name for name,p in unet.named_parameters()):
        raise RuntimeError("Devono essere addestrabili soltanto i parametri LoRA della U-Net.")
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 betas=(0.9, 0.999), weight_decay=config.weight_decay, eps=1e-8)
    sampler = EpochSampler(train, config.seed)
    loader = DataLoader(train, sampler=sampler, batch_size=config.batch_size, collate_fn=collate_batch,
                        num_workers=config.num_workers, pin_memory=accelerator.device.type == "cuda",
                        generator=torch.Generator().manual_seed(config.seed))
    val_loader = DataLoader(validation, batch_size=config.batch_size, collate_fn=collate_batch,
                            num_workers=config.num_workers, pin_memory=accelerator.device.type == "cuda",
                            generator=torch.Generator().manual_seed(config.seed+1))
    lr_scheduler = get_scheduler(config.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=config.warmup_steps, num_training_steps=config.max_train_steps)
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)

    def export_adapter(directory):
        state = get_peft_model_state_dict(accelerator.unwrap_model(unet))
        state = {name:value.detach().float().cpu().contiguous() for name,value in state.items()}
        StableDiffusionPipeline.save_lora_weights(
            str(directory), unet_lora_layers=convert_state_dict_to_diffusers(state), safe_serialization=True)
        return state

    def save_hook(models, weights, directory):
        state = export_adapter(directory)
        save_file(state, str(Path(directory)/"adapter_training.safetensors"))
        # Accelerate salva optimizer, scheduler, scaler e RNG; non duplica la U-Net congelata.
        weights.clear()

    def load_hook(models, directory):
        state = load_file(str(Path(directory)/"adapter_training.safetensors"))
        model = accelerator.unwrap_model(unet)
        expected = get_peft_model_state_dict(model)
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError("Checkpoint LoRA con chiavi o shape incompatibili.")
        set_peft_model_state_dict(model, state)
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    if checkpoint:
        accelerator.load_state(str(checkpoint))
    output.mkdir(parents=True, exist_ok=True)
    versions = {name:importlib.metadata.version(name) for name in
                ["torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "Pillow", "numpy"]}
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=False).stdout.strip() or None
    except OSError:
        git_commit = None
    manifest = dict(config=asdict(config), signature=signature, resolved_revision=revision, data=audits,
                    versions=versions, python=platform.python_version(), git_commit=git_commit,
                    device_selection="auto", device=str(accelerator.device),
                    effective_mixed_precision=precision,
                    gpu=(torch.cuda.get_device_name(accelerator.device)
                         if accelerator.device.type == "cuda" else None),
                    trainable_parameters=sum(p.numel() for p in parameters),
                    effective_batch_size=config.batch_size*config.gradient_accumulation_steps,
                    validation="Held-out denoising MSE; fixed noise/timesteps seed and fixed order; no image generation",
                    scope="U-Net LoRA only; VAE/text encoder frozen; no ControlNet or SD integration")
    if not (output/"run_config.json").exists():
        (output/"run_config.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2), flush=True)
    (output/"tokenizer").mkdir(exist_ok=True)
    tokenizer.save_pretrained(output/"tokenizer")
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    start = time.monotonic()

    def save_checkpoint():
        destination = output/f"checkpoint-{progress['global_step']:06d}"
        temporary = output/f".checkpoint-{progress['global_step']:06d}.incomplete"
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint gia presente: {destination} oppure {temporary}")
        accelerator.save_state(str(temporary))
        (temporary/"progress.json").write_text(json.dumps(progress, indent=2)+"\n")
        temporary.rename(destination)

    # Un log distinto per ogni avvio evita righe ambigue dopo una ripresa.
    with (output/f"metrics_from_{progress['global_step']:06d}_{time.time_ns()}.csv").open("w", newline="") as stream:
        log = csv.DictWriter(stream, fieldnames=["step", "epoch", "train_loss", "validation_loss", "learning_rate", "elapsed_seconds"])
        log.writeheader()
        optimizer.zero_grad(set_to_none=True)
        first_epoch, skip = progress["epoch"], progress["next_batch"]
        unet.train()
        for epoch in range(first_epoch, first_epoch+config.max_train_steps+1):
            sampler.set_epoch(epoch)
            loader.set_epoch(epoch)
            active = accelerator.skip_first_batches(loader, skip) if epoch == first_epoch and skip else loader
            offset = skip if epoch == first_epoch else 0
            accumulated, microbatches = 0.0, 0
            for batch_index, batch in enumerate(active, start=offset):
                with accelerator.accumulate(unet):
                    with accelerator.autocast():
                        loss = batch_loss(batch, unet, vae, text_encoder, noise_scheduler,
                                          dtype, accelerator.device).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Loss non finita allo step {progress['global_step']}")
                    accumulated += loss.detach().item()
                    microbatches += 1
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                if accelerator.optimizer_step_was_skipped:
                    accumulated, microbatches = 0.0, 0
                    continue
                progress["global_step"] += 1
                at_end = batch_index+1 == len(loader)
                progress.update(epoch=epoch+1 if at_end else epoch, next_batch=0 if at_end else batch_index+1)
                step = progress["global_step"]
                val = ""
                if config.validation_steps and (step % config.validation_steps == 0 or step == config.max_train_steps):
                    val = validation_loss(val_loader, accelerator, unet, vae, text_encoder, noise_scheduler, dtype, config.seed)
                    if not math.isfinite(val):
                        raise FloatingPointError("Validation loss non finita.")
                record = dict(step=step, epoch=epoch, train_loss=accumulated/microbatches,
                              validation_loss=val, learning_rate=lr_scheduler.get_last_lr()[0],
                              elapsed_seconds=time.monotonic()-start)
                log.writerow(record)
                stream.flush()
                if step == 1 or step % 10 == 0 or val != "":
                    print(json.dumps(record), flush=True)
                accumulated, microbatches = 0.0, 0
                if step % config.checkpointing_steps == 0 or step == config.max_train_steps:
                    save_checkpoint()
                if step >= config.max_train_steps:
                    break
            if progress["global_step"] >= config.max_train_steps:
                break
        else:
            raise RuntimeError("Training non completato: troppi aggiornamenti saltati.")
    export_adapter(output)
    summary = dict(global_step=progress["global_step"], elapsed_seconds_this_session=time.monotonic()-start,
                   peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(accelerator.device)
                                              if accelerator.device.type == "cuda" else None),
                   resumed_from=str(checkpoint) if checkpoint else None,
                   adapter=str(output/"pytorch_lora_weights.safetensors"), status="completed")
    (output/"training_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    accelerator.end_training()


def main(argv=None):
    args, config = parse_args(argv)
    if args.print_config:
        print(json.dumps(asdict(config), indent=2))
        return
    if args.check_data:
        train, validation, audits = load_data(config)
        for dataset in [train, validation]:
            for i in range(len(dataset)):
                pixels = dataset[i]["pixel_values"]
                if pixels.shape != (3, config.resolution, config.resolution) or not torch.isfinite(pixels).all():
                    raise ValueError(f"Tensore immagine non valido: {dataset.records[i].path}")
        print(json.dumps(audits, indent=2))
        return
    run_training(config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
