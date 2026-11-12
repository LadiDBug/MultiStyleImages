"""Prepara copie RQ1 per seed senza modelli, inferenza o modifica degli originali.

Senza --output-dir verifica soltanto selezione, metadati, file e hash input.
"""

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil

from src.evaluation.records import KEY_FIELDS, _safe_id, read_json


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_subset(source_root, selection):
    """Valida l'intera matrice prima di creare directory o copiare file."""
    ids = selection["image_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Lista immagini vuota o con duplicati.")
    for image_id in ids:
        _safe_id(image_id)
    selected = set(ids)
    expected = {(i, s, c, k) for i in ids for s in selection["styles"]
                for c in selection["conditions"] for k in selection["seeds"]}
    if len(expected) != selection["expected_outputs"]:
        raise ValueError("Conteggio della matrice diverso dalla selezione congelata.")
    found = {}
    templates = {}
    reference = None
    checked_inputs = set()
    schedulers = set()
    for relative_run in selection["source_runs"]:
        run = source_root / _safe_id(relative_run)
        manifest = read_json(run / "run_config.json")
        if manifest.get("mode") != "generation" or manifest["preprocessing_enabled"]:
            raise ValueError(f"Run non compatibile con gli input congelati: {run}")
        # Questi campi cambiano tra shard; tutti gli altri devono coincidere.
        signature = {k: v for k, v in manifest.items()
                     if k not in {"experiment", "input_images", "data"}}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Configurazioni diverse tra run: {run}")
        experiment = manifest["experiment"]
        for field in ("styles", "conditions"):
            if set(experiment[field]) != set(selection[field]):
                raise ValueError(f"Matrice {field} diversa: {run}")
        for seed in experiment["seeds"]:
            if seed not in selection["seeds"]:
                raise ValueError(f"Seed inatteso: {seed}")
            templates.setdefault(seed, manifest)
        for image_id in ids:
            if image_id not in manifest["input_images"]:
                continue
            sample = run / "samples" / _safe_id(image_id)
            metadata = sample / "metadata.jsonl"
            # Uno shard interrotto può dichiarare foto presenti nella continuazione.
            if not metadata.exists():
                continue
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = tuple(record[f] for f in KEY_FIELDS)
                if record["image_id"] not in selected or key not in expected:
                    raise ValueError(f"Record inatteso nel sample selezionato: {key}")
                if key in found:
                    raise ValueError(f"Record duplicato: {key}")
                if record["status"] != "success":
                    raise ValueError(f"Record non riuscito: {key}")
                for field in ("model", "generation", "preprocessing_enabled"):
                    if record[field] != manifest[field]:
                        raise ValueError(f"{field} incoerente: {key}")
                for field in ("prompt", "negative_prompt"):
                    if record[field] != manifest["styles"][record["style"]][field]:
                        raise ValueError(f"{field} incoerente: {key}")
                if record["condition"] != "none":
                    if (record["controlnet_checkpoint"] != manifest["controlnets"][record["condition"]]
                            or record["controlnet_conditioning_scale"] != manifest["controlnet"]["conditioning_scale"]):
                        raise ValueError(f"ControlNet incoerente: {key}")
                output = (sample / "images" / record["style"] / record["condition"]
                          / f"{Path(image_id).stem}_seed{record['seed']}.png")
                if not output.is_file():
                    raise ValueError(f"Output assente: {output}")
                input_path = sample / "input_original.png"
                if input_path not in checked_inputs:
                    if sha256(input_path) != selection["input_sha256"][image_id]:
                        raise ValueError(f"Hash input diverso da quello congelato: {input_path}")
                    checked_inputs.add(input_path)
                schedulers.add(record["scheduler"])
                found[key] = (record, input_path, output, relative_run)
    if set(found) != expected:
        raise ValueError(f"Matrice incompleta: {len(expected - set(found))} output mancanti.")
    if len(schedulers) != 1:
        raise ValueError(f"Scheduler non omogenei: {schedulers}")
    return templates, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "result_rq1")
    parser.add_argument("--selection", type=Path, default=ROOT / "configs/rq1_common104.json")
    parser.add_argument("--output-dir", type=Path, help="Directory nuova; se omessa esegue solo verifiche.")
    args = parser.parse_args()
    selection = read_json(args.selection)
    templates, rows = plan_subset(args.source_root.resolve(), selection)
    print(f"Selezione verificata: {len(selection['image_ids'])} foto, {len(rows)} output; nessuna inferenza.")
    if args.output_dir is None:
        return
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "selection.json").write_bytes(args.selection.read_bytes())
    provenance = []
    for seed in selection["seeds"]:
        run = destination / f"seed_{seed}"
        run.mkdir()
        manifest = copy.deepcopy(templates[seed])
        manifest["input_images"] = selection["image_ids"]
        manifest["experiment"]["seeds"] = [seed]
        manifest["subset_provenance"] = {"selection_id": selection["selection_id"],
                                         "selection_sha256": sha256(args.selection),
                                         "source_runs": selection["source_runs"]}
        (run / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        samples = defaultdict(list)
        for key, (record, input_path, output, source_run) in rows.items():
            if record["seed"] != seed:
                continue
            sample = run / "samples" / _safe_id(record["image_id"])
            sample.mkdir(parents=True, exist_ok=True)
            snapshot = sample / "input_original.png"
            if not snapshot.exists():
                shutil.copy2(input_path, snapshot)
            target = sample / "images" / record["style"] / record["condition"] / output.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, target)
            samples[sample].append(record)
            provenance.append({**dict(zip(KEY_FIELDS, key)), "source_run": source_run,
                               "source_output": str(output), "copied_output_sha256": sha256(target)})
        for sample, records in samples.items():
            (sample / "metadata.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Copie preparate in {destination}. Eseguire --audit-only su ciascuna seed_*/ prima delle metriche.")


if __name__ == "__main__":
    main()
