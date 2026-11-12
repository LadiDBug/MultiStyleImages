"""Runtime persistente: un backbone condiviso, tre ControlNet e LoRA opzionali."""

import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image

from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("none", "canny", "depth", "segmentation")
STYLES = ("gta5", "pixel_art", "watercolor")


def find_loras(root=ROOT, overrides=None):
    """Il rank identifica un checkpoint addestrato, non un parametro di inferenza."""
    paths = {}
    for rank in (4, 8, 16, 32):
        config = load_config(Path(root) / "configs" / f"lora_rank{rank}.toml")
        paths[rank] = Path(root) / config["data"]["output_dir"]
    paths.update(overrides or {})
    available = {}
    for rank, path in paths.items():
        if type(rank) is not int or rank <= 0:
            raise ValueError("I rank LoRA devono essere interi positivi.")
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        weight = path / "pytorch_lora_weights.safetensors" if path.is_dir() else path
        if weight.is_file() and weight.suffix == ".safetensors":
            available[rank] = weight.resolve()
        elif overrides and rank in overrides:
            raise ValueError(f"Pesi LoRA rank {rank} non trovati: {weight}")
    return dict(sorted(available.items()))


class DemoRuntime:
    def __init__(self, root=ROOT, *, device="cuda", lora_paths=None):
        self.root = Path(root).resolve()
        self.config = load_config(self.root / "configs/base.toml")
        self.styles = load_config(self.root / "configs/styles.toml")["styles"]
        self.controlnets = load_config(self.root / "configs/models.toml")["controlnets"]
        self.lora_paths = find_loras(self.root, lora_paths)
        self.device = device
        self.pipes = {}
        self.preprocessors = {}
        self.lock = Lock()
        self.ready = False

    def load(self):
        """Eseguire nel notebook PRIMA di avviare il sito; nessun download nei callback."""
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        from src.generation.pipeline import attach_lora, load_pipeline
        from src.preprocessing.depth import DepthEstimator
        from src.preprocessing.semantic import SemanticSegmentor

        with self.lock:
            if self.ready:
                return self
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("GPU CUDA assente: attiva un runtime GPU nel notebook.")
            dtype = "float32" if self.device == "cpu" else "float16"
            model = self.config["model"]
            revisions = set()
            for rank, weight in self.lora_paths.items():
                metadata_path = weight.parent / "run_config.json"
                if not metadata_path.is_file() and weight.parent.name.startswith("checkpoint-"):
                    metadata_path = weight.parent.parent / "run_config.json"
                if metadata_path.is_file():
                    training = json.loads(metadata_path.read_text())
                    metadata = training.get("config", {})
                    if metadata.get("rank", rank) != rank:
                        raise ValueError(f"Rank incompatibile con il checkpoint {weight}")
                    if metadata.get("model_id", model["id"]) != model["id"]:
                        raise ValueError(f"Backbone incompatibile con la LoRA {weight}")
                    if training.get("resolved_revision"):
                        revisions.add(training["resolved_revision"])
            if model.get("revision"):
                revisions.add(model["revision"])
            if len(revisions) > 1:
                raise ValueError("Le LoRA disponibili richiedono revisioni diverse del backbone.")
            print("Caricamento backbone Stable Diffusion…")
            base = load_pipeline(model["id"], device="cpu", dtype="float32",
                                 revision=next(iter(revisions), None))
            base.to(dtype=getattr(torch, dtype))
            self.pipes = {"none": base}
            for rank, weight in self.lora_paths.items():
                attach_lora(base, weight, adapter_name=f"rank_{rank}")
            if self.lora_paths:
                base.disable_lora()
            for condition in CONDITIONS[1:]:
                print(f"Caricamento ControlNet {condition}…")
                controlnet = ControlNetModel.from_pretrained(
                    self.controlnets[condition], torch_dtype=getattr(torch, dtype))
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pipe(
                    base, controlnet=controlnet)
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
                pipe.enable_vae_tiling()
                self.pipes[condition] = pipe
            base.enable_attention_slicing()
            base.enable_vae_slicing()
            base.enable_vae_tiling()
            # CPU per i preprocessori: evita di tenere anche DPT/UperNet in VRAM.
            print("Caricamento preprocessori depth e segmentation su CPU…")
            prep = self.config["preprocessing"]
            self.preprocessors = {
                "depth": DepthEstimator(prep["depth"]["model_id"], device="cpu"),
                "segmentation": SemanticSegmentor(prep["segmentation"]["model_id"], device="cpu"),
            }
            self.ready = True
            print(f"Modelli pronti. Rank LoRA disponibili: {list(self.lora_paths) or 'nessuno'}")
        return self

    def run(self, image, condition, style, use_lora=False, rank=None, seed=42):
        from src.generation.generator import generate

        if not self.ready:
            raise ValueError("Esegui prima la cella di caricamento dei modelli.")
        if not isinstance(image, Image.Image):
            raise ValueError("Carica un'immagine prima di generare.")
        if condition not in CONDITIONS or style not in STYLES:
            raise ValueError("Mappa o stile non supportati.")
        if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not math.isfinite(seed) or not 0 <= seed <= 2**32 - 1 or int(seed) != seed:
            raise ValueError("Il seed deve essere un intero tra 0 e 4294967295.")
        # Valori di controlli nascosti non possono applicare LoRA ad altri stili.
        use_lora = bool(use_lora and style == "pixel_art")
        if use_lora and rank not in self.lora_paths:
            raise ValueError("Scegli un rank LoRA con pesi disponibili.")
        with self.lock:
            start = perf_counter()
            generation = self.config["generation"]
            # Come --no-preprocess del runner: solo RGB, senza crop o resize.
            prepared = image.convert("RGB")
            condition_image = None
            if condition == "canny":
                from src.preprocessing.canny import extract_canny
                condition_image = extract_canny(prepared, **self.config["preprocessing"]["canny"])
            elif condition != "none":
                condition_image = self.preprocessors[condition].process(prepared)
            base = self.pipes["none"]
            pipe = self.pipes[condition]
            use_offload = self.device.split(":")[0] == "cuda"
            try:
                if use_lora:
                    base.enable_lora()
                    base.set_adapters([f"rank_{int(rank)}"], adapter_weights=[1.0])
                elif self.lora_paths:
                    base.disable_lora()
                if use_offload:
                    # Installa gli hook solo sulla pipeline attiva: UNet/VAE/text
                    # encoder sono condivisi anche con le altre condizioni.
                    pipe.enable_model_cpu_offload(device=self.device)
                else:
                    pipe.to(self.device)
                output = generate(
                    pipe, prepared, self.styles[style]["prompt"],
                    self.styles[style]["negative_prompt"], int(seed), **generation,
                    use_original_image=True,
                    condition_image=condition_image,
                    controlnet_conditioning_scale=(self.config["controlnet"]["conditioning_scale"]
                                                   if condition != "none" else None))
            finally:
                try:
                    if self.lora_paths:
                        base.disable_lora()
                finally:
                    if use_offload:
                        pipe.remove_all_hooks()
                    pipe.to("cpu", silence_dtype_warnings=True)
            info = {"stile": style, "mappa": condition, "seed": int(seed),
                    "lora_rank": int(rank) if use_lora else None,
                    "secondi": round(perf_counter() - start, 2)}
            return output, condition_image, info
