"""Evaluation RQ1 separata dalla generazione: python -m src.evaluation.evaluate."""
# orchestratore dell'eval.


import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import statistics
import subprocess
import tomllib

from PIL import Image

from src.evaluation.records import audit_run, write_csv


METRICS = ("clip_i", "clip_t", "semantic_miou", "dino_i")
ROOT = Path(__file__).resolve().parents[2]


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_and_interval(values: list[float]) -> dict:
    """Bootstrap per input, dopo la media dei seed. Stima descrittiva al 95%."""
    import numpy as np

    result = {"n_inputs": len(values), "mean": statistics.mean(values),
              "std_across_inputs": statistics.stdev(values) if len(values) > 1 else "",
              "ci95_low": "", "ci95_high": ""}
    if len(values) > 1:
        generator = np.random.default_rng(2026)
        data = np.asarray(values)
        means = [float(generator.choice(data, size=len(data), replace=True).mean()) for _ in range(2000)]
        result.update(zip(("ci95_low", "ci95_high"), np.quantile(means, [0.025, 0.975]).tolist()))
    return result


def summarize(rows: list[dict], conditions: list[str], metrics: list[str]):
    """Solo blocchi foto/stile/seed completi per confronti appaiati equi."""
    blocks = defaultdict(dict)
    for row in rows:
        if row["status"] == "success":
            blocks[(row["image_id"], row["style"], row["seed"])][row["condition"]] = row
    complete = [block for block in blocks.values() if set(block) == set(conditions)]
    groups = defaultdict(lambda: defaultdict(list))
    deltas = defaultdict(lambda: defaultdict(list))
    for block in complete:
        for condition, row in block.items():
            for category in ("__all__", row["category"]):
                for metric in metrics:
                    groups[(category, row["style"], condition, metric)][row["image_id"]].append(row[metric])
                    if condition != "none" and "none" in block:
                        delta = row[metric] - block["none"][metric]
                        deltas[(category, row["style"], condition, metric)][row["image_id"]].append(delta)
    summaries = []
    comparisons = []
    for mapping, target in ((groups, summaries), (deltas, comparisons)):
        for (category, style, condition, metric), inputs in sorted(mapping.items()):
            target.append({"category": category, "style": style, "condition": condition,
                           "metric": metric, "n_outputs": sum(map(len, inputs.values())),
                           **mean_and_interval([statistics.mean(values) for values in inputs.values()])})
    return summaries, comparisons, len(complete)


def evaluate_rows(rows: list[dict], protocol: dict, metrics: list[str], device: str, destination: Path):
    import torch
    from src.evaluation.content import SemanticEvaluator, evaluation_size, semantic_miou
    from src.evaluation.dino import DINOEvaluator
    from src.evaluation.style import CLIPEvaluator, cosine_similarity

    clip = CLIPEvaluator(protocol["clip"], device) if {"clip_i", "clip_t"} & set(metrics) else None
    semantic = SemanticEvaluator(protocol["semantic"], device) if "semantic_miou" in metrics else None
    dino = DINOEvaluator(protocol["dino"], device) if "dino_i" in metrics else None
    text_embeddings = {}
    if "clip_t" in metrics:
        text_embeddings = {style: clip.text_embedding(protocol["style_texts"][style])
                           for style in {row["style"] for row in rows}}
    # Registrare revisioni risolte e preprocessing effettivo, non solo 'main'.
    models = {}
    for name, evaluator in (("clip", clip), ("semantic", semantic), ("dino", dino)):
        if evaluator is not None:
            models[name] = {
                "resolved_revision": getattr(evaluator.model.config, "_commit_hash", None),
                "model_config": evaluator.model.config.to_dict(),
                "processor_config": evaluator.processor.to_dict()
                if hasattr(evaluator.processor, "to_dict") else evaluator.processor.image_processor.to_dict(),
            }
    dump_json(destination / "evaluators.json", models)

    current_source = None
    reference_embedding = reference_labels = size = source_hash = None
    reference_dino_embedding = None
    for index, row in enumerate(rows, 1):
        row.update(status="error", error="")
        try:
            input_path = Path(row["input_path"])
            if current_source != input_path:
                with Image.open(input_path) as source:
                    image = source.convert("RGB")
                size = evaluation_size(image.size, protocol["semantic"]["long_edge"]) if semantic is not None else None
                source_hash = file_hash(input_path)
                reference_embedding = clip.image_embedding(image) if "clip_i" in metrics else None
                reference_labels = semantic.predict(image, size) if semantic is not None else None
                reference_dino_embedding = dino.image_embedding(image) if dino is not None else None
                current_source = input_path
            row["input_sha256"] = source_hash
            row["output_sha256"] = file_hash(Path(row["output_path"]))
            with Image.open(row["output_path"]) as output:
                image = output.convert("RGB")
            if clip is not None:
                embedding = clip.image_embedding(image)
                if "clip_i" in metrics:
                    row["clip_i"] = cosine_similarity(reference_embedding, embedding)
                if "clip_t" in metrics:
                    row["clip_t"] = cosine_similarity(embedding, text_embeddings[row["style"]])
            if semantic is not None:
                prediction = semantic.predict(image, size)
                row["semantic_miou"] = semantic_miou(reference_labels, prediction, semantic.num_classes)
                row.update(semantic_grid_width=size[0], semantic_grid_height=size[1])
            if dino is not None:
                row["dino_i"] = cosine_similarity(reference_dino_embedding, dino.image_embedding(image))
            row["status"] = "success"
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}")
        print(f"[{index}/{len(rows)}] {row['image_id']} {row['style']}/{row['condition']}: {row['status']}", flush=True)
        # Conserva i risultati gia' completati anche in caso di interruzione.
        write_csv(destination / "per_image_metrics.csv", rows[:index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Una cartella run con run_config.json e samples/.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.toml")
    parser.add_argument("--output-dir", type=Path, help="Directory nuova; default: <run>/evaluation/<timestamp>.")
    parser.add_argument("--audit-only", action="store_true", help="Inventario senza modelli, GPU o download.")
    parser.add_argument("--allow-partial", action="store_true", help="Valuta output validi; riepiloghi solo su blocchi completi.")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, help="Override delle metriche configurate.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        protocol = tomllib.load(stream)
    metrics = args.metrics or protocol["evaluation"]["metrics"]
    if not metrics or len(metrics) != len(set(metrics)) or set(metrics) - set(METRICS):
        parser.error("Selezione metriche vuota, duplicata o sconosciuta.")
    run_dir = args.run_dir.resolve()
    manifest, rows, report = audit_run(run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = args.output_dir or run_dir / "evaluation" / timestamp
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "audit.csv", rows)
    dump_json(destination / "audit_summary.json", report)
    dump_json(destination / "generation_manifest.json", manifest)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Risultati: {destination}", flush=True)
    if args.audit_only:
        return 0 if report["invalid_outputs"] == 0 else 2
    if report["invalid_outputs"] and not args.allow_partial:
        print("Audit non completo: consultare audit.csv; --allow-partial abilita l'analisi parziale.")
        return 2
    valid_rows = [row.copy() for row in rows if row["status"] == "ready"]
    if not valid_rows:
        print("Nessun output valido da valutare.")
        return 2
    if "clip_t" in metrics:
        for style in manifest["experiment"]["styles"]:
            if not protocol["style_texts"].get(style, "").strip():
                parser.error(f"Manca style_texts.{style} nella configurazione di evaluation.")

    import torch

    device = args.device or protocol["evaluation"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        parser.error(f"Device non disponibile: {device}")
    versions = {}
    for name in ("torch", "transformers", "Pillow", "numpy", "huggingface-hub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    dump_json(destination / "evaluation_config.json", {
        "protocol": protocol, "metrics": metrics, "device": device, "versions": versions,
        "git_commit": commit, "run_dir": str(run_dir), "timestamp_utc": timestamp,
        "partial_requested": args.allow_partial,
        "aggregation": "complete image/style/seed blocks; mean seeds per input then mean inputs",
        "bootstrap": {"unit": "input", "samples": 2000, "seed": 2026, "confidence": 0.95},
    })
    evaluate_rows(valid_rows, protocol, metrics, device, destination)
    summaries, comparisons, n_blocks = summarize(valid_rows, manifest["experiment"]["conditions"], metrics)
    fields = ["category", "style", "condition", "metric", "n_outputs", "n_inputs", "mean", "std_across_inputs", "ci95_low", "ci95_high"]
    write_csv(destination / "summary_metrics.csv", summaries, fields)
    write_csv(destination / "paired_vs_none.csv", comparisons, fields)
    dump_json(destination / "evaluation_summary.json", {
        "evaluated_outputs": len(valid_rows),
        "successful_outputs": sum(row["status"] == "success" for row in valid_rows),
        "complete_image_style_seed_blocks": n_blocks,
        "outputs_used_in_summary": n_blocks * len(manifest["experiment"]["conditions"]),
        "note": "paired_vs_none: differenza metrica condizione - none, prima mediata sui seed per input.",
    })
    return 0 if all(row["status"] == "success" for row in valid_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
