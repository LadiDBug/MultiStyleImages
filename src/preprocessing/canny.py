"""Estrazione dei bordi Canny dalla fotografia gia' preprocessata."""

import cv2
import numpy as np
from PIL import Image


def extract_canny(
    image: Image.Image,
    *,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Restituisce una mappa RGB con bordi bianchi su sfondo nero.

    Riceve la stessa foto RGB preparata per img2img e ne mantiene le dimensioni.
    Le soglie iniziali sono da validare nel pilot e registrare nei metadati.
    Non carica modelli, non ridimensiona e non salva file.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image deve essere un'immagine PIL gia' preprocessata.")
    if image.mode != "RGB":
        raise ValueError("La foto deve essere RGB, come l'input di img2img.")
    if (
        type(low_threshold) is not int
        or type(high_threshold) is not int
        or not 0 <= low_threshold < high_threshold <= 255
    ):
        raise ValueError("Le soglie devono essere interi: 0 <= low < high <= 255.")

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=3,
        L2gradient=False,
    )
    # ControlNet ricevera' tre canali identici, non una mappa a canale singolo.
    return Image.fromarray(edges).convert("RGB")
