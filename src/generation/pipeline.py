"""Caricamento Stable Diffusion img2img, con ControlNet e LoRA opzionali."""

import math
from pathlib import Path
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
)


def attach_lora(pipe, lora_path: str | Path, scale: float = 1.0, *, adapter_name="pixel_art") -> dict:
    """Carica un adattatore locale; non modifica ControlNet o i prompt.

    Chiamare soltanto sulla pipeline destinata allo stile dell'adattatore.
    Senza questa chiamata, il percorso RQ1 rimane senza LoRA.
    """
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError("La scala LoRA deve essere finita e non negativa.")
    path = Path(lora_path).expanduser().resolve()
    weight = path / "pytorch_lora_weights.safetensors" if path.is_dir() else path
    if not weight.is_file() or weight.suffix != ".safetensors":
        raise ValueError(f"Pesi LoRA locali .safetensors non trovati: {weight}")
    pipe.load_lora_weights(str(weight.parent), weight_name=weight.name,
                           adapter_name=adapter_name, local_files_only=True)
    pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])
    return {"path": str(weight), "adapter_name": adapter_name, "scale": float(scale)}


def load_pipeline(
    model_id: str,
    device: str = "cuda",
    dtype: str = "float16",
    *,
    condition: str = "none",
    controlnet_id: str | None = None,
    revision: str | None = None,
    lora_path: str | Path | None = None,
    lora_scale: float = 1.0,
    cache_dir: str | Path | None = None,
) -> StableDiffusionImg2ImgPipeline | StableDiffusionControlNetImg2ImgPipeline:
    """Carica il modello e lo scheduler, senza eseguire generazioni.

    model_id, device e dtype provengono dalla sezione model di base.toml.
    Per canny, depth e segmentation, controlnet_id proviene da models.toml.
    Il primo caricamento puo' scaricare il checkpoint se non e' in cache.
    device='auto' sceglie CUDA/MPS/CPU; dtype='auto' usa FP16 su GPU e
    FP32 su CPU. La LoRA viene caricata soltanto se lora_path e' specificato.
    """
    if condition not in {"none", "canny", "depth", "segmentation"}:
        raise ValueError(
            "Condizioni supportate: 'none', 'canny', 'depth', 'segmentation'."
        )
    if condition == "none" and controlnet_id is not None:
        raise ValueError("La baseline non deve ricevere un checkpoint ControlNet.")
    if condition != "none" and (
        not isinstance(controlnet_id, str) or not controlnet_id.strip()
    ):
        raise ValueError(f"La condizione {condition} richiede controlnet_id.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    target_device = torch.device(device)
    if dtype == "auto":
        dtype = "float16" if target_device.type in {"cuda", "mps"} else "float32"
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA richiesta ma non disponibile. Per usare la CPU, impostare "
            'device = "cpu" e dtype = "float32" in base.toml.'
        )

    precisions = {"float16": torch.float16, "float32": torch.float32}
    if dtype not in precisions:
        raise ValueError("dtype deve essere 'float16' oppure 'float32'.")
    if target_device.type == "cpu" and dtype == "float16":
        raise ValueError("Per la CPU usare dtype = 'float32'.")

    model_kwargs = {"torch_dtype": precisions[dtype]}
    if revision is not None:
        model_kwargs["revision"] = revision
    if cache_dir is not None:
        model_kwargs["cache_dir"] = str(cache_dir)
    if condition == "none":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            **model_kwargs,
        )
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=precisions[dtype],
            **({"cache_dir": str(cache_dir)} if cache_dir is not None else {}),
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            **model_kwargs,
        )

    if lora_path is not None:
        attach_lora(pipe, lora_path, lora_scale)

    # Stesso scheduler e precisione per il confronto tra le condizioni.
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(type(pipe.scheduler).__name__)
    return pipe.to(target_device)
