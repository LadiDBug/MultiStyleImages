"""Generazione di una singola immagine con la baseline img2img."""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


# TODO RQ1: accettare una condition_image opzionale e passarla come
# control_image solo quando la pipeline usa ControlNet.

def generate(
    pipe: StableDiffusionImg2ImgPipeline,
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
) -> Image.Image:
    """Genera dalla foto gia' preparata, senza caricare o salvare file.

    I parametri keyword corrispondono alla sezione generation di base.toml.
    Il preprocessing deve fornire una foto RGB delle dimensioni configurate.
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
        )
    return result.images[0]
