"""Modelli caricati per richiesta e rilasciati al termine della generazione."""

import gc
import json
import logging
import traceback
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
        self._pipe = None
        self.lock = Lock()
        self.ready = False

    def load(self):
        """Verifica il runtime senza caricare modelli; compatibile con il notebook."""
        import torch

        if self.device.split(":")[0] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA assente: attiva un runtime GPU nel notebook.")
        self.ready = True
        print(f"Demo pronta. Modelli caricati solo alla generazione. Rank LoRA: {list(self.lora_paths) or 'nessuno'}")
        return self

    def _lora_revision(self, rank):
        model = self.config["model"]
        revision = model.get("revision")
        if rank is None:
            return revision
        weight = self.lora_paths[rank]
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
            trained_revision = training.get("resolved_revision")
            if revision and trained_revision and revision != trained_revision:
                raise ValueError("La LoRA selezionata richiede una revisione diversa del backbone.")
            revision = trained_revision or revision
        return revision

    def _condition_image(self, image, condition):
        """Carica solo l'estrattore richiesto e lo elimina prima di caricare SD."""
        if condition == "none":
            return None
        if condition == "canny":
            from src.preprocessing.canny import extract_canny
            return extract_canny(image, **self.config["preprocessing"]["canny"])
        processor = None
        try:
            config = self.config["preprocessing"][condition]
            if condition == "depth":
                from src.preprocessing.depth import DepthEstimator
                processor = DepthEstimator(config["model_id"], device="cpu")
            else:
                from src.preprocessing.semantic import SemanticSegmentor
                processor = SemanticSegmentor(config["model_id"], device="cpu")
            return processor.process(image)
        finally:
            processor = None
            gc.collect()

    def _load_request_pipeline(self, condition, rank):
        import torch
        from diffusers import (
            ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionImg2ImgPipeline,
        )
        from src.generation.pipeline import attach_lora

        revision = self._lora_revision(rank)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        kwargs = {"torch_dtype": dtype}
        if revision is not None:
            kwargs["revision"] = revision
        if condition == "none":
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config["model"]["id"], **kwargs)
        else:
            controlnet = ControlNetModel.from_pretrained(
                self.controlnets[condition], torch_dtype=dtype)
            self._pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config["model"]["id"], controlnet=controlnet, **kwargs)
        if rank is not None:
            attach_lora(self._pipe, self.lora_paths[rank], adapter_name=f"rank_{rank}")
        self._pipe.enable_attention_slicing()
        self._pipe.enable_vae_slicing()
        self._pipe.enable_vae_tiling()
        if self.device.split(":")[0] == "cuda":
            self._pipe.enable_model_cpu_offload(device=self.device)
        else:
            self._pipe.to(self.device)

    def _release_models(self):
        """Rimuove riferimenti e hook, poi libera la cache CUDA inutilizzata."""
        import torch

        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            try:
                pipe.remove_all_hooks()
                pipe.to("cpu", silence_dtype_warnings=True)
            except Exception:
                # La rimozione dei riferimenti deve avvenire anche se il cleanup fallisce.
                logging.warning("Rilascio demo: trasferimento CPU incompleto; eliminazione dei riferimenti.")
            finally:
                pipe = None
        gc.collect()
        if self.device.split(":")[0] == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def run(self, image, condition, style, use_lora=False, rank=None, seed=42):
        from src.generation.generator import generate

        if not self.ready:
            raise ValueError("Inizializza la demo con runtime.load().")
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
            selected_rank = int(rank) if use_lora else None
            try:
                condition_image = self._condition_image(prepared, condition)
                self._load_request_pipeline(condition, selected_rank)
                output = generate(
                    self._pipe, prepared, self.styles[style]["prompt"],
                    self.styles[style]["negative_prompt"], int(seed), **generation,
                    use_original_image=True,
                    condition_image=condition_image,
                    controlnet_conditioning_scale=(self.config["controlnet"]["conditioning_scale"]
                                                   if condition != "none" else None))
            except Exception as exc:
                # I frame di un errore possono trattenere tensori e modelli anche
                # dopo aver eliminato self._pipe. Conserva il traceback, svuota i locali.
                traceback.clear_frames(exc.__traceback__)
                raise
            finally:
                self._release_models()
            info = {"stile": style, "mappa": condition, "seed": int(seed),
                    "lora_rank": selected_rank,
                    "secondi": round(perf_counter() - start, 2)}
            return output, condition_image, info
