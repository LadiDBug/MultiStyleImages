"""Generazione img2img di una singola immagine, con mappa opzionale."""

import math
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image


def generate(
    pipe: StableDiffusionImg2ImgPipeline | StableDiffusionControlNetImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    strength: float,
    condition_image: Image.Image | None = None,
    controlnet_conditioning_scale: float | None = None,
) -> Image.Image:
    """Genera dalla foto gia' preparata, senza caricare o salvare file.

    I parametri keyword corrispondono alla sezione generation di base.toml.
    Il preprocessing deve fornire una foto RGB delle dimensioni configurate.
    Con ControlNet servono anche una mappa RGB allineata e la sua forza.
    La funzione non deve sapere come e' stata prodotta la mappa.
    Il seed e' ripetibile nei limiti delle versioni software e dell'hardware.
    """
    if image.mode != "RGB":
        raise ValueError("La foto deve essere convertita in RGB dal preprocessing.")
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError("width e height devono essere positivi e multipli di 8.")
    if image.size != (width, height):
        raise ValueError(
            f"Foto di dimensioni {image.size}; attese {(width, height)}. "
            "Applicare il preprocessing prima della generazione."
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed deve essere un intero non negativo.")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps deve essere un intero positivo.")
    if not 0 < strength <= 1 or int(steps * strength) < 1:
        raise ValueError("strength deve essere in (0, 1] e steps * strength >= 1.")

    control_kwargs = {}
    if isinstance(pipe, StableDiffusionControlNetImg2ImgPipeline):
        if not isinstance(condition_image, Image.Image):
            raise ValueError("La pipeline ControlNet richiede una mappa PIL.")
        if condition_image.mode != "RGB" or condition_image.size != image.size:
            raise ValueError("La mappa deve essere RGB e avere le dimensioni della foto.")
        if (
            type(controlnet_conditioning_scale) not in (int, float)
            or not math.isfinite(controlnet_conditioning_scale)
            or controlnet_conditioning_scale < 0
        ):
            raise ValueError("ControlNet richiede una conditioning scale finita e non negativa.")
        control_kwargs = {
            "control_image": condition_image,
            "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        }
    elif condition_image is not None or controlnet_conditioning_scale is not None:
        raise ValueError("La baseline non accetta mappe o parametri ControlNet.")

    # Un generatore nuovo per ogni chiamata evita dipendenze dall'ordine dei run.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=1,
            output_type="pil",
            **control_kwargs,
        )
    return result.images[0]
