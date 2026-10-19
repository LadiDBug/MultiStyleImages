"""Stima MiDaS della profondita' relativa, senza preprocessing fotografico."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation


class DepthEstimator:
    """Carica DPT-Hybrid MiDaS una volta e lo riutilizza sulle fotografie.

    Il runner sceglie se fornire la foto originale o preprocessata.
    Il primo caricamento puo' scaricare processore e pesi da Hugging Face.
    """

    def __init__(
        self,
        model_name: str = "Intel/dpt-hybrid-midas",
        device: str | None = None,
        *,
        revision: str | None = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name deve essere un checkpoint non vuoto.")
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA richiesta ma non disponibile per lo stimatore depth.")
        self.processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = DPTForDepthEstimation.from_pretrained(
            model_name, revision=revision
        ).to(self.device).eval()

    def process_depth(self, image: Image.Image) -> np.ndarray:
        """Restituisce la predizione float32 (H, W), alle dimensioni ricevute.

        MiDaS stima profondita' inversa relativa: valori maggiori indicano
        generalmente superfici piu' vicine. Non sono distanze in metri.
        Il processore applica le trasformazioni interne richieste dal modello;
        non vengono applicati crop o miglioramenti di colore aggiuntivi.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image deve essere un'immagine PIL.")
        if image.mode != "RGB":
            raise ValueError("Fornire la stessa foto RGB usata dalla generazione.")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        depth = prediction.cpu().numpy().astype(np.float32)
        if not np.isfinite(depth).all():
            raise ValueError("La predizione depth contiene valori non finiti.")
        return depth

    @staticmethod
    def depth_to_image(depth: np.ndarray) -> Image.Image:
        """Normalizza min-max per foto a 0..255 e replica su tre canali RGB.

        Vicino = chiaro, lontano = scuro. Una predizione costante produce nero.
        Per conservare i valori float, salvare separatamente process_depth().
        """
        if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth deve essere un array numpy 2D non vuoto.")
        if not np.issubdtype(depth.dtype, np.number) or np.iscomplexobj(depth):
            raise ValueError("depth deve contenere numeri reali.")
        values = depth.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("depth deve contenere solo valori finiti.")
        minimum = float(values.min())
        span = float(values.max()) - minimum
        if span > 0:
            normalized = (values - minimum) / span
            pixels = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        else:
            pixels = np.zeros(values.shape, dtype=np.uint8)
        return Image.fromarray(pixels).convert("RGB")

    def process(self, image: Image.Image) -> Image.Image:
        """Restituisce la mappa RGB per ControlNet, senza salvare file."""
        return self.depth_to_image(self.process_depth(image))
