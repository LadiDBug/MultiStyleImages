"""Inventario verificabile degli output di UN run, anche trasferito da Kaggle."""
# prepara l'elenco degli output da valutare e controlla che siano utilizzabili.  
import csv
import json
from pathlib import Path, PurePosixPath

from PIL import Image


KEY_FIELDS = ("image_id", "style", "condition", "seed")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(value: str) -> Path:
    # Il run corrente registra image_id con slash POSIX anche su Windows.
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
        raise ValueError(f"image_id non relativo o non sicuro: {value!r}")
    return Path(*relative.parts)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.load()
        return image.size


def audit_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    """Usa il manifest originale per rilevare anche foto interamente mancanti.

    Gli assoluti registrati da Kaggle non vengono aperti. I percorsi sono
    ricostruiti dalla struttura samples/image_id del runner corrente.
    """
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "run_config.json")
    if manifest.get("mode") != "generation" or not manifest.get("input_images"):
        raise ValueError("Serve un run di generazione batch con input_images nel manifest.")
    experiment = manifest["experiment"]
    for key in ("styles", "conditions", "seeds"):
        values = experiment[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"experiment.{key} vuoto o con duplicati.")
    if len(manifest["input_images"]) != len(set(manifest["input_images"])):
        raise ValueError("input_images contiene duplicati.")

    records = {}
    for metadata in sorted((run_dir / "samples").rglob("metadata.jsonl")):
        for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = tuple(record[field] for field in KEY_FIELDS)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Record non valido: {metadata}:{line_number}: {error}") from error
            if key in records:
                raise ValueError(f"Record duplicato, confronto ambiguo: {key}")
            expected_parent = run_dir / "samples" / _safe_id(record["image_id"])
            if metadata.parent != expected_parent:
                raise ValueError(f"image_id non corrisponde alla cartella: {metadata}")
            records[key] = record

    rows = []
    for image_id in manifest["input_images"]:
        relative = _safe_id(image_id)
        sample_dir = run_dir / "samples" / relative
        prepared = manifest["preprocessing_enabled"]
        input_path = sample_dir / ("input_preprocessed.png" if prepared else "input_original.png")
        for style in experiment["styles"]:
            for condition in experiment["conditions"]:
                if style not in {"gta5", "pixel_art", "watercolor", "cyberpunk", "custom"}:
                    raise ValueError(f"Stile non supportato nell'inventario: {style}")
                if condition not in {"none", "canny", "depth", "segmentation"}:
                    raise ValueError(f"Condizione sconosciuta: {condition}")
                for seed in experiment["seeds"]:
                    output = sample_dir / "images" / style / condition / f"{relative.stem}_seed{seed}.png"
                    key = (image_id, style, condition, seed)
                    record = records.pop(key, None)
                    row = dict(zip(KEY_FIELDS, key))
                    row.update(
                        category=relative.parent.as_posix() if len(relative.parts) > 1 else "unassigned",
                        input_path=str(input_path), output_path=str(output),
                        preprocessing_enabled=prepared, status="ready", error="",
                    )
                    try:
                        if record is None:
                            raise ValueError("Record di generazione assente.")
                        if record.get("status") != "success":
                            raise ValueError(f"Generazione non riuscita: {record.get('error', 'stato sconosciuto')}")
                        for field in ("generation", "model", "preprocessing_enabled"):
                            if record.get(field) != manifest[field]:
                                raise ValueError(f"{field} differisce dal manifest del run.")
                        for field in ("prompt", "negative_prompt"):
                            if record.get(field) != manifest["styles"][style][field]:
                                raise ValueError(f"{field} differisce dal manifest del run.")
                        if condition != "none":
                            if record.get("controlnet_checkpoint") != manifest["controlnets"][condition]:
                                raise ValueError("Checkpoint ControlNet diverso dal manifest.")
                            if record.get("controlnet_conditioning_scale") != manifest["controlnet"]["conditioning_scale"]:
                                raise ValueError("Control scale diversa dal manifest.")
                        source_size = _image_size(input_path)
                        output_size = _image_size(output)
                        if tuple(record.get("input_size", ())) != source_size:
                            raise ValueError("Dimensioni input diverse dai metadati.")
                        row.update(input_width=source_size[0], input_height=source_size[1],
                                   output_width=output_size[0], output_height=output_size[1],
                                   scheduler=record["scheduler"])
                        # Piccoli arrotondamenti ai multipli del VAE sono ammessi.
                        ratio_change = (output_size[0] / output_size[1]) / (source_size[0] / source_size[1])
                        if abs(ratio_change - 1) > 0.02:
                            raise ValueError("Rapporto d'aspetto cambiato oltre il 2%: verificare allineamento.")
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row.update(status="invalid", error=str(error))
                    rows.append(row)
    if records:
        raise ValueError(f"Metadati estranei alla matrice dichiarata: {list(records)[:3]}")
    # Nessun confronto automatico fra condizioni con geometrie diverse.
    sizes = {}
    for row in rows:
        if row["status"] == "ready":
            sizes.setdefault(row["image_id"], set()).add((row["output_width"], row["output_height"]))
    for row in rows:
        if row["status"] == "ready" and len(sizes[row["image_id"]]) > 1:
            row.update(status="invalid", error="Dimensioni output diverse tra condizioni/stili/seed dello stesso input.")
    schedulers = {row["scheduler"] for row in rows if row["status"] == "ready"}
    if len(schedulers) > 1:
        raise ValueError(f"Scheduler diversi nello stesso confronto: {schedulers}")
    report = {
        "expected_outputs": len(rows),
        "ready_outputs": sum(row["status"] == "ready" for row in rows),
        "invalid_outputs": sum(row["status"] != "ready" for row in rows),
        "n_inputs": len(manifest["input_images"]),
        "styles": experiment["styles"], "conditions": experiment["conditions"],
        "seeds": experiment["seeds"],
        "full_rq1_conditions": set(experiment["conditions"]) == {"none", "canny", "depth", "segmentation"},
    }
    return manifest, rows, report
