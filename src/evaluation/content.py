"""Consistenza semantica fra predizioni, senza ground truth annotata."""

# la domanda a cui vogliamo rispondere è: gli elementi della scena occupano ancora le stesse zone dopo la stilizzazione?à
#Fotografia originale → segmentatore → mappa delle classi ─┐
#                                                        ├→ mIoU
#Immagine generata     → segmentatore → mappa delle classi ─┘

import numpy as np


def semantic_miou(reference: np.ndarray, prediction: np.ndarray, num_classes: int) -> float:
    """Macro IoU sulle classi presenti nell'unione delle due mappe.

    Le classi assenti in entrambe sono escluse; una classe presente soltanto
    in una delle mappe contribuisce zero. Nessuna classe background implicita.
    Gli ID devono essere gia' allineati nello spazio e nello stesso vocabolario.
    """
    if reference.shape != prediction.shape or reference.ndim != 2 or not reference.size:
        raise ValueError("Le mappe devono essere 2D, non vuote e della stessa forma.")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes deve essere un intero positivo.")
    for labels in (reference, prediction):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Le mappe devono contenere ID interi.")
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError("ID di classe fuori dal vocabolario del valutatore.")
    encoded = reference.astype(np.int64).ravel() * num_classes + prediction.ravel()
    matrix = np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    intersection = matrix.diagonal()
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    present = union > 0
    return float((intersection[present] / union[present]).mean())


def evaluation_size(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """Griglia (width, height) comune, senza crop; non supera long_edge."""
    if long_edge < 1 or min(size) < 1:
        raise ValueError("Dimensioni non valide.")
    factor = min(1.0, long_edge / max(size))
    return tuple(max(1, round(value * factor)) for value in size)


class SemanticEvaluator:
    def __init__(self, config: dict, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.torch = torch
        self.device = device
        self.num_classes = config["num_classes"]
        kwargs = {"revision": config["revision"]}
        self.processor = AutoImageProcessor.from_pretrained(config["model_id"], **kwargs)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            config["model_id"], **kwargs
        ).to(device).eval()
        if self.model.config.num_labels != self.num_classes:
            raise ValueError("Il numero di classi del checkpoint non coincide col protocollo.")

    def predict(self, image, size: tuple[int, int]) -> np.ndarray:
        """Interpola i logits sulla griglia comune PRIMA dell'argmax.

        Non riusa mappe di conditioning e non applica offset ADE20K alle
        predizioni, che hanno gia' ID 0..149. Nessun crop di valutazione.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits, size=(size[1], size[0]), mode="bilinear", align_corners=False
            )
            labels = logits.argmax(dim=1)[0]
        return labels.cpu().numpy().astype(np.int32)
