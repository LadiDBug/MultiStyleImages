"""Embedding CLIP condivisi per CLIP-I e CLIP-T (coseno non scalato)."""
# traforma immagini e testi in vettori clip e li confronta.
# CLIP-I: confronta foto originale e output → quanto il contenuto rimane simile.
# CLIP-T: confronta output e testo dello stile → quanto l’immagine è allineata allo stile descritto.

import numpy as np


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    if first.shape != second.shape or not first.size:
        raise ValueError("Embedding incompatibili.")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if not np.isfinite(first).all() or not np.isfinite(second).all() or denominator <= 0:
        raise ValueError("Embedding non finiti o nulli.")
    return float(np.clip(np.dot(first, second) / denominator, -1, 1))


class CLIPEvaluator:
    def __init__(self, config: dict, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        kwargs = {"revision": config["revision"]}
        self.processor = CLIPProcessor.from_pretrained(config["model_id"], **kwargs)
        self.model = CLIPModel.from_pretrained(config["model_id"], **kwargs).to(device).eval()

    def _array(self, features) -> np.ndarray:
        # Transformers 4 restituisce un Tensor; versioni nuove un output con pooling.
        if not isinstance(features, self.torch.Tensor):
            features = features.pooler_output
        return features[0].detach().float().cpu().numpy()

    def image_embedding(self, image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self._array(self.model.get_image_features(**inputs))

    def text_embedding(self, text: str) -> np.ndarray:
        # Fallire sui testi troppo lunghi anziche' troncare silenziosamente.
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Testo CLIP troppo lungo: ridurre style_texts nel TOML.")
        with self.torch.inference_mode():
            return self._array(self.model.get_text_features(**inputs.to(self.device)))
