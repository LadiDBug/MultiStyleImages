"""Verifiche dei criteri di campionamento, senza aprire immagini o usare modelli."""

import unittest

from src.evaluation.visual_review_rq1 import select_cases


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.selection = {"image_ids": ["d.png", "c.png", "b.png", "a.png"],
                          "styles": ["gta5", "pixel_art", "watercolor"]}
        self.contrasts = []
        base = {"a.png": (.9, .1, .1), "b.png": (.5, .1, -.9),
                "c.png": (.8, -.2, -.5), "d.png": (.1, .1, .0)}
        for style in self.selection["styles"]:
            for image_id, (dino, miou, clip_t) in base.items():
                self.contrasts.append({"image_id": image_id, "style": style, "contrast": "canny_minus_none",
                                       "dino_i": dino, "semantic_miou": miou, "clip_t": clip_t})
                self.contrasts.append({"image_id": image_id, "style": style, "contrast": "depth_minus_segmentation",
                                       "dino_i": dino, "semantic_miou": -.4 if image_id == "a.png" else .1,
                                       "clip_t": -.2})
                self.contrasts.append({"image_id": image_id, "style": style, "contrast": "canny_minus_depth",
                                       "dino_i": .7, "semantic_miou": .5,
                                       "clip_t": .0 if image_id == "d.png" else -.2})

    def test_random_photos_are_shared_by_all_styles_and_ignore_input_order(self):
        ids, cases = select_cases(self.selection, self.contrasts, random_count=2)
        for style in self.selection["styles"]:
            self.assertEqual([c["image_id"] for c in cases if c["style"] == style and c["set"] == "random"], ids)
        reversed_selection = {**self.selection, "image_ids": list(reversed(self.selection["image_ids"]))}
        self.assertEqual(select_cases(reversed_selection, self.contrasts, random_count=2), (ids, cases))

    def test_diagnostic_criteria_select_known_extremes_and_disagreement(self):
        _, cases = select_cases(self.selection, self.contrasts, random_count=2)
        actual = {(c["style"], c["reason"]): c["image_id"] for c in cases if c["set"] == "diagnostic"}
        for style in self.selection["styles"]:
            self.assertEqual(actual[style, "content_gain"], "a.png")
            self.assertEqual(actual[style, "style_loss"], "b.png")
            self.assertEqual(actual[style, "content_disagreement"], "c.png")
        self.assertEqual(actual["gta5", "gta_layout"], "a.png")
        self.assertEqual(actual["pixel_art", "pixel_tradeoff"], "a.png")
        self.assertEqual(actual["watercolor", "watercolor_tradeoff"], "d.png")

    def test_missing_diagnostic_criterion_does_not_silently_select_another_case(self):
        rows = [{**r, "semantic_miou": .1} for r in self.contrasts]
        with self.assertRaisesRegex(ValueError, "Nessun caso"):
            select_cases(self.selection, rows, random_count=2)


if __name__ == "__main__":
    unittest.main()
