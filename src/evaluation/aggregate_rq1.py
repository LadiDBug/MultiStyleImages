"""Analisi offline RQ1 su CSV completi; nessun import Torch o inferenza.

python -m src.evaluation.aggregate_rq1 --results-dir result_evaluation_rq1 --output-dir analysis/rq1_common104
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import tomllib
import csv

import numpy as np


METRICS = ("clip_i", "dino_i", "semantic_miou", "clip_t")
CONDITIONS = ("none", "canny", "depth", "segmentation")
PAIRS = (("canny", "none"), ("depth", "none"), ("segmentation", "none"),
         ("canny", "depth"), ("canny", "segmentation"), ("depth", "segmentation"))
KEYS = ("image_id", "style", "condition", "seed")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_export(root):
    """Rifiuta matrici incomplete e protocolli/hash incoerenti prima di aggregare."""
    selection = read_json(root / "selection.json")
    protocol = tomllib.loads((root / "evaluation_frozen.toml").read_text(encoding="utf-8"))
    environment = read_json(root / "environment.json")
    if read_json(root / "export_status.json")["status"] != "COMPLETE":
        raise ValueError("Export non dichiarato completo.")
    for key, filename in (("selection_sha256", "selection.json"),
                          ("protocol_sha256", "evaluation_frozen.toml"),
                          ("bundle_sha256", "rq1_evaluation_code.zip")):
        if environment[key] != digest(root / filename):
            raise ValueError(f"Hash diverso per {filename}.")
    ids, styles, seeds = (selection[k] for k in ("image_ids", "styles", "seeds"))
    for values in (ids, styles, seeds):
        if not values or len(values) != len(set(values)):
            raise ValueError("Selezione vuota o con duplicati.")
    if set(selection["conditions"]) != set(CONDITIONS):
        raise ValueError("Servono tutte e quattro le condizioni RQ1.")
    expected = set(itertools.product(ids, styles, CONDITIONS, seeds))
    if len(expected) != selection["expected_outputs"]:
        raise ValueError("Conteggio selezione incoerente.")
    provenance_rows = read_json(root / "provenance.json")
    provenance = {tuple(r[k] for k in KEYS): r for r in provenance_rows}
    if len(provenance) != len(provenance_rows) or set(provenance) != expected:
        raise ValueError("Provenienza incompleta o duplicata.")
    completed = read_json(root / "completed_evaluations.json")
    if set(completed) != {str(k) for k in seeds}:
        raise ValueError("Elenco dei seed completati incoerente.")
    rows = []
    seen = set()
    inputs = defaultdict(set)
    evaluator_reference = None
    signature_reference = None
    source_files = {name: digest(root / name) for name in (
        "selection.json", "evaluation_frozen.toml", "environment.json",
        "rq1_evaluation_code.zip", "provenance.json", "completed_evaluations.json", "export_status.json")}
    for seed, relative in completed.items():
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Directory di evaluation esterna all'export.")
        config = read_json(directory / "evaluation_config.json")
        evaluators = read_json(directory / "evaluators.json")
        manifest = read_json(directory / "generation_manifest.json")
        signature = {k: v for k, v in manifest.items()
                     if k not in {"input_images", "experiment", "data", "subset_provenance"}}
        if signature_reference is None:
            signature_reference = signature
        if signature != signature_reference:
            raise ValueError("Configurazioni di generazione non omogenee.")
        if set(manifest["input_images"]) != set(ids) or manifest["experiment"]["seeds"] != [int(seed)]:
            raise ValueError("Manifest di generazione non corrispondente alla selezione.")
        if config["protocol"] != protocol or set(config["metrics"]) != set(METRICS):
            raise ValueError("Protocolli di evaluation diversi.")
        if evaluator_reference is None:
            evaluator_reference = evaluators
        if evaluators != evaluator_reference:
            raise ValueError("Valutatori diversi tra seed.")
        for name, evaluator in evaluators.items():
            if evaluator["resolved_revision"] != environment["resolved_revisions"][name]:
                raise ValueError("Revisione valutatore incoerente con l'ambiente.")
        for filename in ("per_image_metrics.csv", "evaluation_config.json", "evaluators.json",
                         "evaluation_summary.json", "generation_manifest.json", "audit_summary.json"):
            source_files[f"{relative}/{filename}"] = digest(directory / filename)
        audit = read_json(directory / "audit_summary.json")
        if audit["invalid_outputs"] or not audit["full_rq1_conditions"]:
            raise ValueError("Audit incompleto.")
        with (directory / "per_image_metrics.csv").open(encoding="utf-8", newline="") as stream:
            local_rows = list(csv.DictReader(stream))
        expected_count = len(ids) * len(styles) * len(CONDITIONS)
        report = read_json(directory / "evaluation_summary.json")
        if len(local_rows) != expected_count or any(report[k] != expected_count for k in (
                "evaluated_outputs", "successful_outputs", "outputs_used_in_summary")):
            raise ValueError("Conteggio dei risultati incoerente.")
        for row in local_rows:
            row["seed"] = int(row["seed"])
            key = tuple(row[k] for k in KEYS)
            if key not in expected or key in seen or row["seed"] != int(seed) or row["status"] != "success":
                raise ValueError(f"Chiave non valida, fallita o duplicata: {key}")
            seen.add(key)
            if row["input_sha256"] != selection["input_sha256"][row["image_id"]]:
                raise ValueError("Input diverso dalla selezione congelata.")
            if row["output_sha256"] != provenance[key]["copied_output_sha256"]:
                raise ValueError("Hash output diverso dalla provenienza.")
            for metric in METRICS:
                value = row[metric] = float(row[metric])
                if not math.isfinite(value) or not (-1 <= value <= 1) or (metric == "semantic_miou" and value < 0):
                    raise ValueError(f"Punteggio non valido: {key}, {metric}")
            inputs[row["image_id"]].add((row["category"], row["input_width"], row["input_height"],
                                         row["output_width"], row["output_height"], row["scheduler"]))
            row["evaluation_source"] = relative
            rows.append(row)
    if seen != expected or any(len(values) != 1 for values in inputs.values()):
        raise ValueError("Matrice incompleta o geometrie/categorie/scheduler diversi per la stessa foto.")
    if len({r["scheduler"] for r in rows}) != 1:
        raise ValueError("Scheduler non omogenei.")
    return selection, rows, source_files


def stats(values, draws):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Campione statistico non valido.")
    interval = np.quantile(values[draws].mean(axis=1), [0.025, 0.975]) if len(values) > 1 else (None, None)
    return {"n_inputs": len(values), "mean": float(values.mean()),
            "std_across_inputs": float(values.std(ddof=1)) if len(values) > 1 else "",
            "ci95_low": float(interval[0]) if interval[0] is not None else "",
            "ci95_high": float(interval[1]) if interval[1] is not None else "",
            "median": float(np.median(values)), "q05": float(np.quantile(values, .05)),
            "q95": float(np.quantile(values, .95))}


def aggregate(selection, rows, bootstrap_samples=2000, bootstrap_seed=2026):
    """Bootstrap delle foto; conserva insieme i seed e le condizioni di ogni foto."""
    ids, styles, seeds = (selection[k] for k in ("image_ids", "styles", "seeds"))
    lookup = {tuple(r[k] for k in KEYS): r for r in rows}
    expected = set(itertools.product(ids, styles, CONDITIONS, seeds))
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Aggregazione consentita solo su una matrice completa senza duplicati.")
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(ids), size=(bootstrap_samples, len(ids)))
    seed_means = []
    summaries = []
    paired = []
    paired_seed = []
    per_input = []
    per_input_paired = []
    stability = []
    for style in styles:
        arrays = {}
        for condition in CONDITIONS:
            for metric in METRICS:
                # Righe = foto indipendenti, colonne = seed ripetuti sulla stessa foto.
                data = np.array([[lookup[(i, style, condition, seed)][metric] for seed in seeds] for i in ids])
                arrays[(condition, metric)] = data
                summaries.append({"style": style, "condition": condition, "metric": metric,
                                  "n_seeds": len(seeds), "n_outputs": data.size,
                                  **stats(data.mean(axis=1), draws)})
                for j, seed in enumerate(seeds):
                    seed_means.append({"style": style, "condition": condition, "metric": metric,
                                       "seed": seed, **stats(data[:, j], draws)})
            for i, image_id in enumerate(ids):
                per_input.append({"image_id": image_id, "style": style, "condition": condition,
                                  "n_seeds": len(seeds), **{m: float(arrays[(condition, m)][i].mean()) for m in METRICS}})
        for a, b in PAIRS:
            pair = f"{a}_minus_{b}"
            for metric in METRICS:
                delta = arrays[(a, metric)] - arrays[(b, metric)]
                means = delta.mean(axis=1)
                paired.append({"style": style, "contrast": pair, "condition_a": a, "condition_b": b,
                               "metric": metric, "n_seeds": len(seeds), "n_paired_outputs": delta.size,
                               **stats(means, draws),
                               "fraction_inputs_positive": float((means > 0).mean()),
                               "fraction_inputs_negative": float((means < 0).mean()),
                               "fraction_inputs_zero": float((means == 0).mean())})
                for j, seed in enumerate(seeds):
                    paired_seed.append({"style": style, "contrast": pair, "metric": metric, "seed": seed,
                                        **stats(delta[:, j], draws),
                                        "fraction_inputs_positive": float((delta[:, j] > 0).mean())})
                seed_delta = delta.mean(axis=0)
                stability.append({"style": style, "contrast": pair, "metric": metric,
                                  **{f"mean_delta_seed_{seed}": float(seed_delta[j]) for j, seed in enumerate(seeds)},
                                  "same_sign_across_seed_means": bool(np.all(seed_delta > 0) or np.all(seed_delta < 0)),
                                  "seed_mean_min": float(seed_delta.min()), "seed_mean_max": float(seed_delta.max()),
                                  "fraction_inputs_all_seeds_positive": float(np.all(delta > 0, axis=1).mean()),
                                  "fraction_inputs_all_seeds_negative": float(np.all(delta < 0, axis=1).mean()),
                                  "fraction_inputs_mixed_or_zero": float((~(np.all(delta > 0, axis=1) | np.all(delta < 0, axis=1))).mean())})
            for i, image_id in enumerate(ids):
                per_input_paired.append({"image_id": image_id, "style": style, "contrast": pair,
                                         "n_seeds": len(seeds), **{m: float((arrays[(a, m)] - arrays[(b, m)])[i].mean()) for m in METRICS}})
    return {"summary_metrics.csv": summaries, "paired_contrasts.csv": paired,
            "per_input_metrics.csv": per_input, "per_input_contrasts.csv": per_input_paired,
            "summary_by_seed.csv": seed_means, "paired_contrasts_by_seed.csv": paired_seed,
            "seed_stability.csv": stability}


def make_figures(output, tables, styles):
    """Figure statiche esportabili; non richiede modelli o GPU."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "multistyleimages_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors = {"none": "#777777", "canny": "#0072B2", "depth": "#D55E00", "segmentation": "#009E73"}
    lookup = {(r["style"], r["condition"], r["metric"]): r for r in tables["summary_metrics.csv"]}
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for j, metric in enumerate(METRICS[:3]):
        for i, style in enumerate(styles):
            ax = axes[i, j]
            for condition in CONDITIONS:
                points = [r for r in tables["per_input_metrics.csv"] if r["style"] == style and r["condition"] == condition]
                ax.scatter([r["clip_t"] for r in points], [r[metric] for r in points], s=8, alpha=.12, color=colors[condition])
                x, y = lookup[(style, condition, "clip_t")], lookup[(style, condition, metric)]
                ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"]-x["ci95_low"]], [x["ci95_high"]-x["mean"]]],
                            yerr=[[y["mean"]-y["ci95_low"]], [y["ci95_high"]-y["mean"]]],
                            fmt="o", capsize=3, color=colors[condition], label=condition)
            ax.set(title=f"{style}: {metric} / CLIP-T", xlabel="CLIP-T", ylabel=metric)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("RQ1: contenuto e stile — 104 foto, media dei 3 seed per foto\nPunti trasparenti: singole foto. Barre: IC marginali descrittivi al 95%.")
    fig.savefig(output / "content_style_tradeoff.png", dpi=180)
    fig.savefig(output / "content_style_tradeoff.pdf")
    plt.close(fig)
    paired = {(r["style"], r["condition_a"], r["metric"]): r for r in tables["paired_contrasts.csv"] if r["condition_b"] == "none"}
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for i, style in enumerate(styles):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for k, condition in enumerate(CONDITIONS[1:]):
                r = paired[(style, condition, metric)]
                ax.errorbar(r["mean"], k, xerr=[[r["mean"]-r["ci95_low"]], [r["ci95_high"]-r["mean"]]], fmt="o", capsize=4, color=colors[condition])
            ax.axvline(0, color="black", linestyle="--", linewidth=.8)
            ax.set_yticks(range(3), CONDITIONS[1:])
            ax.set(title=f"{style}: {metric}", xlabel="Differenza rispetto a none")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("RQ1: delta appaiati vs none — media e IC bootstrap descrittivo al 95%\nRicampionamento di 104 foto, mantenendo insieme i 3 seed; nessuna correzione per confronti multipli.")
    fig.savefig(output / "paired_vs_none.png", dpi=180)
    fig.savefig(output / "paired_vs_none.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory nuova.")
    parser.add_argument("--figures", action="store_true", help="Richiede matplotlib; genera PNG/PDF statici.")
    args = parser.parse_args()
    selection, rows, source_files = read_export(args.results_dir.resolve())
    tables = aggregate(selection, rows)
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "per_output_metrics.csv", rows)
    for filename, values in tables.items():
        write_csv(destination / filename, values)
    if args.figures:
        make_figures(destination, tables, selection["styles"])
    from importlib import metadata
    write_json(destination / "analysis_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(args.results_dir.resolve()),
        "selection_id": selection["selection_id"], "image_ids": selection["image_ids"],
        "excluded_image_ids": selection["excluded_image_ids"], "n_outputs": len(rows),
        "styles": selection["styles"], "conditions": CONDITIONS, "seeds": selection["seeds"], "metrics": METRICS,
        "aggregation": "paired conditions within photo/seed; mean seeds per photo; equal-weight photos",
        "bootstrap": {"unit": "photo", "samples": 2000, "seed": 2026, "confidence": .95,
                      "shared_resampling_indices_across_metrics_and_contrasts": True},
        "inference": "descriptive intervals, no hypothesis tests or multiplicity correction",
        "positive_fraction": "fraction of photo-level seed-averaged deltas strictly above zero",
        "source_sha256": source_files, "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": metadata.version("matplotlib") if args.figures else None,
        "artifacts_sha256": {p.name: digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
    })
    print(f"Analisi completata: {len(selection['image_ids'])} foto, {len(rows)} output; {destination}")
    print("Nessun modello o GPU utilizzato. Metriche per stile e condizione: 48; contrasti: 72.")


if __name__ == "__main__":
    main()
