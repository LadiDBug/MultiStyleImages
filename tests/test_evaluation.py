"""Test numerici e di integrazione offline; nessun peso preaddestrato richiesto."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.evaluation.content import evaluation_size, semantic_miou
from src.evaluation.evaluate import evaluate_rows, main, summarize
from src.evaluation.records import audit_run
from src.evaluation.style import cosine_similarity


def make_run(root: Path) -> Path:
    run = root / "downloaded_from_kaggle"
    run.mkdir()
    image_id = "beach/example.jpg"
    manifest = {
        "mode": "generation", "input_images": [image_id],
        "preprocessing_enabled": False,
        "model": {"id": "example"}, "generation": {"steps": 50},
        "experiment": {"styles": ["watercolor"], "conditions": ["none", "canny"], "seeds": [42]},
        "styles": {"watercolor": {"prompt": "watercolor", "negative_prompt": ""}},
        "controlnets": {"canny": "example-control"}, "controlnet": {"conditioning_scale": 0.8},
    }
    (run / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    sample = run / "samples" / image_id
    sample.mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(sample / "input_original.png")
    records = []
    for condition in ("none", "canny"):
        output = sample / "images" / "watercolor" / condition / "example_seed42.png"
        output.parent.mkdir(parents=True)
        Image.new("RGB", (64, 48)).save(output)
        records.append({
            "image_id": image_id, "style": "watercolor", "condition": condition, "seed": 42,
            "status": "success", "model": manifest["model"], "generation": manifest["generation"],
            "preprocessing_enabled": False, "input_size": [64, 48], "scheduler": "ExampleScheduler",
            "prompt": "watercolor", "negative_prompt": "",
            "controlnet_checkpoint": "example-control" if condition == "canny" else None,
            "controlnet_conditioning_scale": 0.8 if condition == "canny" else None,
            "input_path": "/kaggle/input/never/open/this.jpg",
            "output_path": "/kaggle/working/never/open/this.png",
        })
    (sample / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return run


class MetricTests(unittest.TestCase):
    def test_iou_excludes_absent_both_but_penalizes_new_class(self):
        first = np.array([[0, 0], [1, 1]])
        second = np.array([[0, 2], [1, 1]])
        # IoU: classe 0 = 1/2; classe 1 = 1; classe 2 = 0; classe 3 assente.
        self.assertAlmostEqual(semantic_miou(first, second, 4), 0.5)
        self.assertEqual(semantic_miou(first, first, 4), 1.0)

    def test_iou_rejects_invalid_labels(self):
        with self.assertRaises(ValueError):
            semantic_miou(np.array([[255]]), np.array([[0]]), 150)

    def test_cosine_is_not_scaled_logit_or_probability(self):
        self.assertAlmostEqual(cosine_similarity([2, 0], [20, 0]), 1)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0)

    def test_grid_preserves_aspect_and_does_not_upscale(self):
        self.assertEqual(evaluation_size((1600, 800), 512), (512, 256))
        self.assertEqual(evaluation_size((64, 48), 512), (64, 48))

    def test_paired_summary_averages_seeds_before_images(self):
        rows = []
        # A ha due seed; B uno: ciascuna foto deve pesare ugualmente.
        for image_id, seeds, base in (("a.jpg", (42, 123), 0.2), ("b.jpg", (42,), 0.8)):
            for seed in seeds:
                for condition, delta in (("none", 0), ("canny", 0.1)):
                    rows.append(dict(image_id=image_id, style="watercolor", seed=seed,
                                     condition=condition, category="beach", status="success", clip_i=base + delta))
        # Campione spaiato: non deve spostare il riepilogo della baseline.
        rows.append(dict(image_id="incomplete.jpg", style="watercolor", seed=42,
                         condition="none", category="beach", status="success", clip_i=1))
        summary, comparisons, count = summarize(rows, ["none", "canny"], ["clip_i"])
        self.assertEqual(count, 3)
        baseline = next(r for r in summary if r["category"] == "__all__" and r["condition"] == "none")
        self.assertAlmostEqual(baseline["mean"], 0.5)
        self.assertEqual(baseline["n_inputs"], 2)
        self.assertEqual(baseline["n_outputs"], 3)
        self.assertAlmostEqual(comparisons[0]["mean"], 0.1)


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = make_run(Path(self.temp.name))

    def test_relocated_kaggle_run_and_offline_cli(self):
        _, rows, report = audit_run(self.run)
        self.assertEqual(report["ready_outputs"], 2)
        self.assertTrue(all(Path(row["output_path"]).is_file() for row in rows))
        result = subprocess.run([sys.executable, "-m", "src.evaluation.evaluate",
                                 "--run-dir", str(self.run), "--audit-only"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.run.glob("evaluation/*/audit.csv"))), 1)

    def test_missing_output_is_reported(self):
        next(self.run.glob("samples/*/*/images/*/canny/*.png")).unlink()
        _, rows, report = audit_run(self.run)
        self.assertEqual(report["invalid_outputs"], 1)
        self.assertEqual(rows[1]["status"], "invalid")

    def test_manifest_detects_entirely_missing_photo(self):
        path = self.run / "run_config.json"
        manifest = json.loads(path.read_text())
        manifest["input_images"].append("beach/never_generated.jpg")
        path.write_text(json.dumps(manifest))
        _, _, report = audit_run(self.run)
        self.assertEqual(report["expected_outputs"], 4)
        self.assertEqual(report["invalid_outputs"], 2)

    def test_duplicate_metadata_is_rejected(self):
        path = next(self.run.glob("samples/*/*/metadata.jsonl"))
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "duplicato"):
            audit_run(self.run)

    def test_changed_prompt_is_not_silently_mixed(self):
        path = next(self.run.glob("samples/*/*/metadata.jsonl"))
        path.write_text(path.read_text().replace('"prompt": "watercolor"', '"prompt": "changed"'))
        _, _, report = audit_run(self.run)
        self.assertEqual(report["invalid_outputs"], 2)

    def test_different_output_geometry_is_rejected(self):
        output = next(self.run.glob("samples/*/*/images/*/canny/*.png"))
        Image.new("RGB", (32, 24)).save(output)
        _, _, report = audit_run(self.run)
        self.assertEqual(report["invalid_outputs"], 2)

    def test_full_evaluation_writes_scores_and_paired_summary_offline(self):
        # Valutatori deterministici: verifica l'intero flusso I/O senza download.
        class FakeCLIP:
            def __init__(self, config, device):
                self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="offline", to_dict=lambda: {}))
                self.processor = SimpleNamespace(to_dict=lambda: {})

            def image_embedding(self, image):
                return np.array([1.0, 0.0])

            def text_embedding(self, text):
                return np.array([0.6, 0.8])

        class FakeSemantic(FakeCLIP):
            num_classes = 150

            def predict(self, image, size):
                return np.zeros((size[1], size[0]), dtype=np.int32)

        destination = Path(self.temp.name) / "metrics"
        arguments = ["evaluate", "--run-dir", str(self.run), "--output-dir", str(destination), "--device", "cpu"]
        with patch.object(sys, "argv", arguments), \
             patch("src.evaluation.style.CLIPEvaluator", FakeCLIP), \
             patch("src.evaluation.dino.DINOEvaluator", FakeCLIP), \
             patch("src.evaluation.content.SemanticEvaluator", FakeSemantic):
            self.assertEqual(main(), 0)
        import csv
        with (destination / "per_image_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["clip_i"]), 1.0)
        self.assertAlmostEqual(float(rows[0]["clip_t"]), 0.6)
        self.assertEqual(float(rows[0]["semantic_miou"]), 1.0)
        self.assertEqual(float(rows[0]["dino_i"]), 1.0)
        self.assertEqual(len(rows[0]["input_sha256"]), 64)
        report = json.loads((destination / "evaluation_summary.json").read_text())
        self.assertEqual(report["outputs_used_in_summary"], 2)
        with (destination / "paired_vs_none.csv").open(newline="", encoding="utf-8") as stream:
            comparisons = list(csv.DictReader(stream))
        self.assertTrue(all(float(row["mean"]) == 0 for row in comparisons))
        self.assertTrue(any(row["metric"] == "dino_i" for row in comparisons))

    def test_dino_only_does_not_load_clip_or_semantic(self):
        _, rows, _ = audit_run(self.run)
        destination = Path(self.temp.name) / "dino_only"
        destination.mkdir()
        fake = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_commit_hash="offline", to_dict=lambda: {})),
            processor=SimpleNamespace(to_dict=lambda: {}),
            image_embedding=lambda image: np.array([1.0, 0.0]),
        )
        with patch("src.evaluation.dino.DINOEvaluator", return_value=fake) as dino, \
             patch("src.evaluation.style.CLIPEvaluator") as clip, \
             patch("src.evaluation.content.SemanticEvaluator") as semantic:
            evaluate_rows(rows, {"dino": {}}, ["dino_i"], "cpu", destination)
        clip.assert_not_called()
        semantic.assert_not_called()
        dino.assert_called_once()
        self.assertTrue(all(row["status"] == "success" and row["dino_i"] == 1 for row in rows))
        info = json.loads((destination / "evaluators.json").read_text())
        self.assertEqual(set(info), {"dino"})


if __name__ == "__main__":
    unittest.main()
