"""Aggiunge moduli offline alla galleria RQ1 esistente, senza rigenerare le griglie."""

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from src.evaluation.aggregate_rq1 import digest, read_json, write_json
from src.evaluation.visual_review_rq1 import REASONS, relative_url


ASSETS = Path(__file__).parent / "review_ui"
STYLE_LABELS = {"gta5": "GTA V", "pixel_art": "Pixel Art", "watercolor": "Watercolor"}
STYLE_HINTS = {
    "gta5": "Quali output ricordano una scena di videogioco 3D? Descrivi materiali, illuminazione e resa delle superfici; distinguili da una fotografia o un'illustrazione generica.",
    "pixel_art": "In quali output riconosci pixel visibili, blocchi coerenti, bordi e palette limitata? Controlla i dettagli a piena risoluzione: una semplice riduzione dei colori non basta da sola.",
    "watercolor": "In quali output riconosci lavature, trasparenze, pigmento, carta e bordi pittorici? Descrivi ciò che vedi, senza presumere che questi segni debbano essere presenti.",
}


def source_files(manifest, image_id, style):
    files = {}
    for condition in ("original", "none", "canny", "depth", "segmentation"):
        suffix = (f"/samples/{image_id}/input_original.png" if condition == "original" else
                  f"/samples/{image_id}/images/{style}/{condition}/{Path(image_id).stem}_seed{manifest['display_seed']}.png")
        matches = [relative for relative in manifest["source_file_sha256"] if relative.replace("\\", "/").endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Sorgente assente o ambiguo per {image_id}/{style}/{condition}.")
        files[condition] = matches[0]
    return files


def gallery_config(directory, manifest, portable=False):
    grouped = OrderedDict()
    for case in manifest["cases"]:
        grouped.setdefault((case["image_id"], case["style"]), []).append(case)
    items = []
    for number, ((image_id, style), cases) in enumerate(grouped.items(), 1):
        grid = f"grids/{style}_{Path(image_id).stem}_seed{manifest['display_seed']}.png"
        if not (directory / grid).is_file():
            raise ValueError(f"Griglia assente: {grid}")
        paths = source_files(manifest, image_id, style)
        urls = {c: ("sources/" + p.replace("\\", "/") if portable else
                    relative_url(Path(manifest["source_root"]) / p, directory)) for c, p in paths.items()}
        # Queste stringhe vanno in attributi URL tramite DOM, non in markup non escapato.
        if portable:
            from urllib.parse import quote
            urls = {c: quote(p, safe="/") for c, p in urls.items()}
        items.append({"id": json.dumps([image_id, style, manifest["display_seed"]], separators=(",", ":")),
                      "number": number, "image_id": image_id, "style": style, "style_label": STYLE_LABELS[style],
                      "seed": manifest["display_seed"], "grid": grid, "part": "A" if number % 2 else "B",
                      "sets": sorted({c["set"] for c in cases}), "reasons": [REASONS[c["reason"]] for c in cases],
                      "style_hint": STYLE_HINTS[style], "sources": urls})
    identity = {"selection_id": manifest["selection_id"], "analysis_sha256": manifest["analysis_manifest_sha256"],
                "display_seed": manifest["display_seed"],
                "grids": {item["id"]: digest(directory / item["grid"]) for item in items}}
    dataset_id = "rq1-grids-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return {"dataset_id": dataset_id, "questionnaire_version": 1, "items": items}


def render_page(config):
    page = (ASSETS / "template.html").read_text(encoding="utf-8")
    replacements = {
        "__STYLE__": (ASSETS / "style.css").read_text(encoding="utf-8"),
        "__CORE__": (ASSETS / "core.js").read_text(encoding="utf-8"),
        "__APP__": (ASSETS / "app.js").read_text(encoding="utf-8"),
        "__CONFIG__": json.dumps(config, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
    }
    for token, value in replacements.items():
        if token not in page:
            raise ValueError(f"Segnaposto mancante: {token}")
        page = page.replace(token, value)
    return page


def upgrade(directory, share_zip=False):
    manifest = read_json(directory / "visual_selection.json")
    for relative, expected_hash in manifest["artifacts_sha256"].items():
        # I CSV di annotazione legacy possono essere stati compilati manualmente: non toccarli.
        if relative.startswith("grids/") or relative.startswith("grids\\"):
            if digest(directory / relative) != expected_hash:
                raise ValueError(f"Griglia modificata: {relative}")
    config = gallery_config(directory, manifest)
    page = render_page(config)
    index = directory / "index.html"
    backup = directory / "index_before_forms.html"
    if index.exists() and not backup.exists():
        shutil.copy2(index, backup)
    index.write_text(page, encoding="utf-8")
    manifest["questionnaire"] = {"version": 1, "dataset_id": config["dataset_id"],
        "unit": "one grid/photo/style/seed per reviewer, four qualitative text answers",
        "storage": "browser localStorage plus user-downloaded JSON; no network or automatic CSV writes",
        "division": "odd grid numbers A, even B; 15 grids and five per style in each part",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "ui_source_sha256": {p.name: digest(p) for p in sorted(ASSETS.iterdir()) if p.is_file()},
        "renderer_sha256": digest(Path(__file__))}
    if share_zip:
        portable = render_page(gallery_config(directory, manifest, portable=True))
        archive = directory / "galleria_rq1_da_condividere.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.writestr("index.html", portable)
            z.writestr("LEGGIMI.txt", "Estrai tutto lo ZIP in una cartella e apri index.html nel browser.\n"
                       "Inserisci il tuo nome, scegli Parte A o Parte B, rispondi alle quattro domande.\n"
                       "Il salvataggio automatico resta nel tuo browser. A fine sessione esporta il JSON.\n"
                       "Per riunire i lavori importa i JSON: revisori diversi restano separati.\n"
                       "Lo ZIP non contiene risposte: per condividerle serve il JSON esportato.\n")
            for item in config["items"]:
                z.write(directory / item["grid"], item["grid"])
            for relative, expected_hash in manifest["source_file_sha256"].items():
                source = Path(manifest["source_root"]) / relative
                if digest(source) != expected_hash:
                    raise ValueError(f"Sorgente modificato prima della condivisione: {relative}")
                z.write(source, "sources/" + relative.replace("\\", "/"))
        manifest["questionnaire"]["share_zip"] = archive.name
    manifest["artifacts_sha256"] = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*"))
                                   if p.is_file() and p.name != "visual_selection.json"}
    write_json(directory / "visual_selection.json", manifest)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--share-zip", action="store_true", help="Include griglie e 137 PNG sorgenti in un pacchetto portabile.")
    args = parser.parse_args()
    config = upgrade(args.gallery_dir.resolve(), args.share_zip)
    print(f"Galleria aggiornata: {len(config['items'])} moduli, quattro domande per revisore; {config['dataset_id']}.")
    print("Nessuna risposta assegnata, nessun CSV utente modificato, nessun modello o GPU utilizzato.")


if __name__ == "__main__":
    main()
