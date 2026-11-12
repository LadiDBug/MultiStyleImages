"""Griglie e indice HTML offline per ispezionare gli output RQ1 esistenti.

Non genera nuove immagini con modelli e non assegna giudizi percettivi.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import tempfile
from urllib.parse import quote

from PIL import Image

from src.evaluation.aggregate_rq1 import CONDITIONS, METRICS, digest, read_json, write_csv, write_json
from src.evaluation.prepare_subset import plan_subset


REASONS = {
    "random": "Campione casuale comune ai tre stili",
    "content_gain": "Massimo guadagno DINO-I: Canny - none",
    "style_loss": "Massimo calo CLIP-T: Canny - none",
    "content_disagreement": "DINO-I aumenta ma mIoU diminuisce: Canny - none",
    "gta_layout": "Massimo guadagno mIoU: segmentation - depth",
    "pixel_tradeoff": "Depth guadagna DINO-I ma perde CLIP-T rispetto a segmentation",
    "watercolor_tradeoff": "Canny guadagna DINO-I/mIoU vs depth, delta CLIP-T vicino a zero",
}


def select_cases(selection, contrasts, random_count=6, random_seed=2026):
    """Selezione deterministica, fissata prima di aprire qualsiasi output."""
    ids = sorted(selection["image_ids"])
    if not 1 <= random_count <= len(ids):
        raise ValueError("Dimensione del campione casuale non valida.")
    random_ids = random.Random(random_seed).sample(ids, random_count)
    lookup = {(r["image_id"], r["style"], r["contrast"]): r for r in contrasts}
    cases = []
    for style in selection["styles"]:
        for image_id in random_ids:
            cases.append({"image_id": image_id, "style": style, "set": "random", "reason": "random"})
        base = [lookup[(image_id, style, "canny_minus_none")] for image_id in ids]
        picks = [
            ("content_gain", sorted(base, key=lambda r: (-r["dino_i"], r["image_id"]))),
            ("style_loss", sorted(base, key=lambda r: (r["clip_t"], r["image_id"]))),
            ("content_disagreement", sorted((r for r in base if r["dino_i"] > 0 and r["semantic_miou"] < 0),
                                            key=lambda r: (-r["dino_i"], r["image_id"]))),
        ]
        if style == "gta5":
            values = [lookup[(i, style, "depth_minus_segmentation")] for i in ids]
            picks.append(("gta_layout", sorted(values, key=lambda r: (r["semantic_miou"], r["image_id"]))))
        elif style == "pixel_art":
            values = [lookup[(i, style, "depth_minus_segmentation")] for i in ids]
            picks.append(("pixel_tradeoff", sorted((r for r in values if r["dino_i"] > 0 and r["clip_t"] < 0),
                                                   key=lambda r: (-r["dino_i"], r["image_id"]))))
        elif style == "watercolor":
            values = [lookup[(i, style, "canny_minus_depth")] for i in ids]
            picks.append(("watercolor_tradeoff", sorted((r for r in values if r["dino_i"] > 0 and r["semantic_miou"] > 0),
                                                        key=lambda r: (abs(r["clip_t"]), r["image_id"]))))
        for reason, candidates in picks:
            if not candidates:
                raise ValueError(f"Nessun caso soddisfa il criterio {style}/{reason}; non sostituire silenziosamente.")
            # Criteri indipendenti: se convergono sulla stessa foto, conservarne entrambi i motivi.
            cases.append({"image_id": candidates[0]["image_id"], "style": style,
                          "set": "diagnostic", "reason": reason})
    return random_ids, cases


def read_analysis(analysis):
    manifest = read_json(analysis / "analysis_manifest.json")
    rows = {}
    for filename in ("per_output_metrics.csv", "per_input_contrasts.csv"):
        if digest(analysis / filename) != manifest["artifacts_sha256"][filename]:
            raise ValueError(f"CSV modificato rispetto al manifest: {filename}")
        with (analysis / filename).open(encoding="utf-8", newline="") as stream:
            values = list(csv.DictReader(stream))
        for row in values:
            for metric in METRICS:
                row[metric] = float(row[metric])
            if "seed" in row:
                row["seed"] = int(row["seed"])
        rows[filename] = values
    return rows


def save_sheet(path, image_id, style, seed, reasons, paths, metrics):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 5.8), layout="constrained")
    for ax, condition in zip(axes, ("original", *CONDITIONS)):
        with Image.open(paths[condition]) as source:
            ax.imshow(source.convert("RGB"), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(condition, fontsize=12, fontweight="bold")
        if condition == "original":
            ax.set_xlabel("Riferimento originale\nNessun punteggio", fontsize=10)
        else:
            row = metrics[(image_id, style, condition, seed)]
            ax.set_xlabel(f"CLIP-I {row['clip_i']:.4f} | DINO-I {row['dino_i']:.4f}\n"
                          f"mIoU {row['semantic_miou']:.4f} | CLIP-T {row['clip_t']:.4f}", fontsize=10)
    label = " / ".join(REASONS[r] for r in reasons)
    fig.suptitle(f"{image_id} | {style} | seed {seed}\n{label}\n"
                 "Punteggi mostrati: seed singolo. Casi diagnostici scelti dalle medie sui 3 seed.", fontsize=12)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def relative_url(path, destination):
    return quote(Path(os.path.relpath(path, destination)).as_posix(), safe="/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("result_evaluation_rq1"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("result_evaluation_rq1/analysis_common104_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("result_rq1"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory nuova.")
    parser.add_argument("--seed", type=int, default=42, help="Seed visualizzato, fissato prima dell'ispezione.")
    parser.add_argument("--random-count", type=int, default=6)
    args = parser.parse_args()
    results, analysis, sources, destination = (p.resolve() for p in (
        args.results_dir, args.analysis_dir, args.source_root, args.output_dir))
    selection = read_json(results / "selection.json")
    analysis_manifest = read_json(analysis / "analysis_manifest.json")
    if digest(results / "selection.json") != analysis_manifest["source_sha256"]["selection.json"]:
        raise ValueError("Selezione diversa dall'analisi quantitativa.")
    if args.seed not in selection["seeds"]:
        raise ValueError("Seed da visualizzare assente dalla selezione.")
    tables = read_analysis(analysis)
    random_ids, cases = select_cases(selection, tables["per_input_contrasts.csv"], args.random_count)
    metrics = {(r["image_id"], r["style"], r["condition"], r["seed"]): r for r in tables["per_output_metrics.csv"]}
    _, source_rows = plan_subset(sources, selection)
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["image_id"], case["style"])].append(case)
    file_hashes = {}
    sheet_inputs = {}
    for (image_id, style), items in grouped.items():
        paths = {}
        for condition in CONDITIONS:
            key = (image_id, style, condition, args.seed)
            _, input_path, output_path, _ = source_rows[key]
            actual_hash = digest(output_path)
            if actual_hash != metrics[key]["output_sha256"]:
                raise ValueError(f"PNG diverso da quello valutato: {output_path}")
            file_hashes[str(output_path.relative_to(sources))] = actual_hash
            paths[condition] = output_path
            paths["original"] = input_path
        file_hashes[str(paths["original"].relative_to(sources))] = digest(paths["original"])
        sheet_inputs[(image_id, style)] = paths
    destination.mkdir(parents=True, exist_ok=False)
    sheets = destination / "grids"
    sheets.mkdir()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "selection_id": selection["selection_id"],
        "display_seed": args.seed, "random_seed": 2026, "random_image_ids": random_ids,
        "random_algorithm": "random.Random(2026).sample(sorted(image_ids), random_count)",
        "cases": cases, "n_unique_grids": len(grouped), "source_root": str(sources),
        "analysis_manifest_sha256": digest(analysis / "analysis_manifest.json"),
        "diagnostic_selection": "criteria use per-photo means over seeds 42/123/456; display uses fixed seed",
        "source_file_sha256": file_hashes, "script_sha256": digest(Path(__file__)),
        "judgments": "not_reviewed; no perceptual scores assigned; method names visible, not a blind study",
    }
    # La selezione viene salvata PRIMA di aprire immagini per la composizione delle griglie.
    write_json(destination / "visual_selection.json", manifest)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multistyleimages_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    annotation_rows = []
    metric_rows = []
    for index, ((image_id, style), items) in enumerate(grouped.items(), 1):
        reasons = [c["reason"] for c in items]
        name = f"{style}_{Path(image_id).stem}_seed{args.seed}.png"
        paths = sheet_inputs[(image_id, style)]
        save_sheet(sheets / name, image_id, style, args.seed, reasons, paths, metrics)
        for condition in CONDITIONS:
            annotation_rows.append({"image_id": image_id, "style": style, "seed": args.seed,
                                    "condition": condition, "sampling_reasons": ";".join(reasons),
                                    "review_status": "not_reviewed", "content_1_5": "", "style_1_5": "",
                                    "quality_no_artifacts_1_5": "", "reviewer": "", "notes": ""})
            metric_rows.append({"image_id": image_id, "style": style, "condition": condition, "seed": args.seed,
                                **{m: metrics[(image_id, style, condition, args.seed)][m] for m in METRICS}})
        print(f"[{index}/{len(grouped)}] {image_id} {style}: griglia pronta", flush=True)
    write_csv(destination / "annotations.csv", annotation_rows)
    write_csv(destination / "displayed_metrics.csv", metric_rows)
    manifest["matplotlib_version"] = matplotlib.__version__
    manifest["artifacts_sha256"] = {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*"))
                                   if p.is_file() and p.name != "visual_selection.json"}
    write_json(destination / "visual_selection.json", manifest)
    from src.evaluation.review_gallery import upgrade
    upgrade(destination)
    print(f"Griglie completate: {len(grouped)}. Aprire {destination / 'index.html'}. Nessun giudizio assegnato.")


if __name__ == "__main__":
    main()
