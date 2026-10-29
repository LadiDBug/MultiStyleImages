"""Compatibilita' API con piccoli modelli casuali locali, senza download."""

import unittest

import numpy as np
from PIL import Image
import torch
from transformers import (
    CLIPConfig, CLIPImageProcessor, CLIPModel,
    SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor,
    BitImageProcessor, Dinov2Config, Dinov2Model,
)

from src.evaluation.content import SemanticEvaluator
from src.evaluation.dino import DINOEvaluator
from src.evaluation.style import CLIPEvaluator, cosine_similarity


class LocalModelTests(unittest.TestCase):
    def test_dino_extracts_cls_embedding(self):
        config = Dinov2Config(
            image_size=28, patch_size=14, hidden_size=32,
            num_hidden_layers=1, num_attention_heads=4, mlp_ratio=2,
        )
        evaluator = DINOEvaluator.__new__(DINOEvaluator)
        evaluator.torch = torch
        evaluator.device = "cpu"
        evaluator.model = Dinov2Model(config).eval()
        evaluator.processor = BitImageProcessor(
            size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28},
        )
        image = Image.new("RGB", (64, 48), "blue")
        embedding = evaluator.image_embedding(image)
        with torch.inference_mode():
            outputs = evaluator.model(**evaluator.processor(images=image, return_tensors="pt"))
        np.testing.assert_allclose(embedding, outputs.last_hidden_state[0, 0].numpy())
        self.assertEqual(embedding.shape, (32,))
        self.assertAlmostEqual(cosine_similarity(embedding, embedding), 1)

    def test_clip_image_and_text_projection_outputs(self):
        config = CLIPConfig(
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=32, patch_size=16),
            projection_dim=8,
        )
        evaluator = CLIPEvaluator.__new__(CLIPEvaluator)
        evaluator.torch = torch
        evaluator.device = "cpu"
        evaluator.model = CLIPModel(config).eval()
        evaluator.processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
        embedding = evaluator.image_embedding(Image.new("RGB", (64, 48), "blue"))
        with torch.inference_mode():
            features = evaluator.model.get_text_features(input_ids=torch.tensor([[1, 5, 2]]))
        text_embedding = evaluator._array(features)
        self.assertEqual(embedding.shape, (8,))
        self.assertEqual(text_embedding.shape, (8,))
        self.assertTrue(np.isfinite(cosine_similarity(embedding, text_embedding)))
        self.assertAlmostEqual(cosine_similarity(embedding, embedding), 1)

    def test_segformer_predictions_on_common_nonsquare_grid(self):
        config = SegformerConfig(
            num_labels=150, hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1],
            num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=16,
        )
        evaluator = SemanticEvaluator.__new__(SemanticEvaluator)
        evaluator.torch = torch
        evaluator.device = "cpu"
        evaluator.model = SegformerForSemanticSegmentation(config).eval()
        evaluator.processor = SegformerImageProcessor(size={"height": 64, "width": 64})
        labels = evaluator.predict(Image.new("RGB", (80, 60), "blue"), (40, 30))
        self.assertEqual(labels.shape, (30, 40))
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLess(int(labels.max()), 150)


if __name__ == "__main__":
    unittest.main()
