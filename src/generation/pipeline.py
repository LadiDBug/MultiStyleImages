"""Caricamento della pipeline Stable Diffusion img2img per la baseline."""

import torch
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler


# TODO RQ1: estendere load_pipeline con condition e checkpoint ControlNet.
# - none: StableDiffusionImg2ImgPipeline
# - canny/depth/segmentation: StableDiffusionControlNetImg2ImgPipeline
# Mantenere backbone, dtype e scheduler coerenti tra le condizioni.


def load_pipeline(
    model_id: str,
    device: str = "cuda",
    dtype: str = "float16",
) -> StableDiffusionImg2ImgPipeline:
    """Carica il modello e lo scheduler, senza eseguire generazioni.

    Gli argomenti corrispondono alla sezione model di base.toml.
    Il primo caricamento puo' scaricare il checkpoint se non e' in cache.
    """
    target_device = torch.device(device)
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

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=precisions[dtype],
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(target_device)
