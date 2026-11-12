"""Verifiche statistiche offline: nessun modello, download o GPU."""

import copy
import unittest

import numpy as np

from src.evaluation.aggregate_rq1 import aggregate, CONDITIONS, METRICS


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.selection = {"image_ids": ["a.png", "b.png"], "styles": ["pixel_art"], "seeds": [42, 123, 456]}
        self.rows = []
        for i, image_id in enumerate(self.selection["image_ids"]):
            for j, seed in enumerate(self.selection["seeds"]):
                baseline = ((.2, .4, .6), (.8, .8, .8))[i][j]
                for condition in CONDITIONS:
                    delta = {"none": 0, "canny": .1, "depth": .05, "segmentation": -.05}[condition]
                    self.rows.append({"image_id": image_id, "style": "pixel_art", "seed": seed,
                                      "condition": condition, **{m: baseline + delta for m in METRICS}})

    def test_mean_and_variance_are_computed_on_photos_not_outputs(self):
        result = aggregate(self.selection, self.rows)
        row = next(r for r in result["summary_metrics.csv"] if r["condition"] == "none" and r["metric"] == "clip_i")
        self.assertAlmostEqual(row["mean"], .6)
        self.assertAlmostEqual(row["std_across_inputs"], np.std([.4, .8], ddof=1))
        self.assertEqual(row["n_inputs"], 2)
        self.assertEqual(row["n_outputs"], 6)

    def test_paired_ci_keeps_conditions_together(self):
        result = aggregate(self.selection, self.rows)
        row = next(r for r in result["paired_contrasts.csv"] if r["contrast"] == "canny_minus_none" and r["metric"] == "clip_i")
        for field in ("mean", "ci95_low", "ci95_high"):
            self.assertAlmostEqual(row[field], .1)
        self.assertEqual(row["fraction_inputs_positive"], 1)
        self.assertEqual(len(result["paired_contrasts.csv"]), 6 * 4)

    def test_seed_sign_changes_are_not_hidden_by_positive_pooled_mean(self):
        rows = copy.deepcopy(self.rows)
        lookup = {(r["image_id"], r["condition"], r["seed"]): r for r in rows}
        for i, image_id in enumerate(self.selection["image_ids"]):
            for j, seed in enumerate(self.selection["seeds"]):
                delta = ((-.3, .1, .4), (-.1, .2, .3))[i][j]
                lookup[(image_id, "canny", seed)]["clip_i"] = lookup[(image_id, "none", seed)]["clip_i"] + delta
        result = aggregate(self.selection, rows)
        row = next(r for r in result["seed_stability.csv"] if r["contrast"] == "canny_minus_none" and r["metric"] == "clip_i")
        self.assertFalse(row["same_sign_across_seed_means"])
        self.assertAlmostEqual(row["mean_delta_seed_42"], -.2)
        self.assertEqual(row["fraction_inputs_mixed_or_zero"], 1)

    def test_incomplete_or_duplicate_matrix_is_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaisesRegex(ValueError, "matrice completa"):
                aggregate(self.selection, rows)


if __name__ == "__main__":
    unittest.main()
