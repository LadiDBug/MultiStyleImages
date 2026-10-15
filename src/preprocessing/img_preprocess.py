"""Preparazione condivisa delle fotografie per gli esperimenti img2img."""

from pathlib import Path

from PIL import Image, ImageOps


def preprocess_image(
    image: str | Path | Image.Image,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Restituisce una foto RGB con crop centrale e resize Lanczos.

    width e height provengono dalla sezione generation di base.toml.
    Il rapporto d'aspetto viene preservato, tagliando le parti periferiche
    necessarie a riempire il formato richiesto. Non modifica colori o contrasto.
    Non salva file e non modifica l'immagine PIL ricevuta.
    """
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width % 8
        or height % 8
    ):
        raise ValueError("width e height devono essere interi positivi multipli di 8.")

    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            prepared = ImageOps.exif_transpose(source).convert("RGB")
    elif isinstance(image, Image.Image):
        prepared = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError("image deve essere un percorso o un'immagine PIL.")

    return ImageOps.fit(
        prepared,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


# TODO RQ1: estrarre le mappe dalla stessa foto preparata passata a img2img,
# mantenendo allineati crop e dimensioni in tutte le condizioni.
