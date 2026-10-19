"""Caricamento Stable Diffusion img2img, con ControlNet opzionale."""

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
)


# TODO RQ1: abilitare depth dopo aver validato il relativo preprocessore.


def load_pipeline(
    model_id: str,
    device: str = "cuda",
    dtype: str = "float16",
    *,
    condition: str = "none",
    controlnet_id: str | None = None,
) -> StableDiffusionImg2ImgPipeline | StableDiffusionControlNetImg2ImgPipeline:
    """Carica il modello e lo scheduler, senza eseguire generazioni.

    model_id, device e dtype provengono dalla sezione model di base.toml.
    Per canny e segmentation, controlnet_id deve provenire da models.toml.
    Il primo caricamento puo' scaricare il checkpoint se non e' in cache.
    """
    if condition not in {"none", "canny", "segmentation"}:
        raise ValueError("Condizioni supportate: 'none', 'canny', 'segmentation'.")
    if condition == "none" and controlnet_id is not None:
        raise ValueError("La baseline non deve ricevere un checkpoint ControlNet.")
    if condition != "none" and (
        not isinstance(controlnet_id, str) or not controlnet_id.strip()
    ):
        raise ValueError(f"La condizione {condition} richiede controlnet_id.")

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

    if condition == "none":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=precisions[dtype],
        )
    else:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=precisions[dtype],
        )
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=precisions[dtype],
        )

    # Stesso scheduler e precisione per il confronto tra le condizioni.
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print(type(pipe.scheduler).__name__)
    return pipe.to(target_device)
