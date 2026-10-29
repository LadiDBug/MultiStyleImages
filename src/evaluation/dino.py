"""Embedding globali DINOv2 per la similarita' visiva originale/output."""

import numpy as np


class DINOEvaluator:
    def __init__(self, config: dict, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        kwargs = {"revision": config["revision"]}
        self.processor = AutoImageProcessor.from_pretrained(config["model_id"], **kwargs)
        self.model = AutoModel.from_pretrained(
            config["model_id"], **kwargs
        ).to(device).eval().requires_grad_(False)

    def image_embedding(self, image) -> np.ndarray:
        """Restituisce il token CLS finale, non la media dei patch token.

        Usa resize/crop/normalizzazione del processore del checkpoint su entrambe
        le immagini. Il coseno viene calcolato dall'orchestratore, come CLIP-I.
        Nessuna segmentazione, confronto locale o misura diretta dello stile.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.model(**inputs).last_hidden_state[:, 0, :]
        return features[0].detach().float().cpu().numpy()
