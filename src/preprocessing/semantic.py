"""Segmentazione ADE20K della foto scelta dal runner, originale o preprocessata."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


# Palette ADE20K ripresa dal vecchio segmentatore.
# ID 0..149, senza aggiungere una classe background o uno spostamento degli ID.
# Fonte: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)

class SemanticSegmentor:
    """Carica UperNet una volta e lo riutilizza per tutte le foto.

    Non chiama img_preprocess.py: il runner decide quale foto fornire.
    Il primo caricamento puo' scaricare processore e pesi da Hugging Face.
    """

    def __init__(
        self,
        model_name: str = "openmmlab/upernet-convnext-base",
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
            raise ValueError("CUDA richiesta ma non disponibile per il segmentatore.")

        self.processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            model_name, revision=revision
        ).to(self.device).eval()
        if self.model.config.num_labels != len(ADE20K_PALETTE):
            raise ValueError("Il modello deve usare le 150 classi ADE20K della palette.")

    def process_labels(self, image: Image.Image) -> np.ndarray:
        """Restituisce gli ID ADE20K in un array uint8 (altezza, larghezza).

        Il processore esegue resize e normalizzazione richiesti dal modello.
        I logits sono riportati alla dimensione ricevuta PRIMA dell'argmax,
        evitando interpolazioni dei colori o degli ID di classe.
        Non modifica la foto e non applica enhancement o crop aggiuntivi.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image deve essere un'immagine PIL.")
        if image.mode != "RGB":
            raise ValueError("Fornire la stessa foto RGB usata dalla generazione.")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            labels = logits.argmax(dim=1)[0]
        return labels.cpu().numpy().astype(np.uint8)

    def process(self, image: Image.Image) -> Image.Image:
        """Restituisce la mappa RGB ADE20K delle stesse dimensioni della foto.

        Non carica o salva immagini su disco. Per salvare anche gli ID senza
        ripetere l'inferenza, chiamare process_labels e poi colorize_labels.
        """
        return self.colorize_labels(self.process_labels(image))

    @staticmethod
    def colorize_labels(labels: np.ndarray) -> Image.Image:
        """Converte gli ID 0..149 nei colori ADE20K, senza offset."""
        if not isinstance(labels, np.ndarray) or labels.ndim != 2 or labels.size == 0:
            raise ValueError("labels deve essere un array numpy 2D non vuoto.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Gli ID di classe devono essere interi.")
        if labels.min() < 0 or labels.max() >= len(ADE20K_PALETTE):
            raise ValueError("Gli ID ADE20K devono essere compresi tra 0 e 149.")
        return Image.fromarray(ADE20K_PALETTE[labels])
