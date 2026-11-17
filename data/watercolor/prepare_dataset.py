"""Restore the frozen selection or audit it, without training a model.

From the repository root:
    python data/watercolor/prepare_dataset.py --download
    python data/watercolor/prepare_dataset.py --audit
Downloads use immutable source revisions and verify every SHA256 before writing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def read_manifest():
    rows = [json.loads(line) for line in (ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    assert len(rows) == 400
    assert len({r["file_name"] for r in rows}) == 400
    for row in rows:
        assert row["split"] in {"train", "validation"}
        assert Path(row["file_name"]).name == row["file_name"]
        assert row["text"].strip() and row["source"] in sources
    return rows, sources


def restore(rows, sources):
    import requests

    def fetch(row):
        path = ROOT / row["split"] / row["file_name"]
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Existing file differs from the manifest: {path}")
            return
        source = sources[row["source"]]
        url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
               f"{source['revision']}/{quote(row['source_path'], safe='/')}")
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                content = response.content
                if hashlib.sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"Downloaded bytes differ from the manifest: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(content)
                temporary.replace(path)
                return
            except requests.RequestException:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, rows))
    for split in ("train", "validation"):
        path = ROOT / split / "metadata.jsonl"
        records = [{k: row[k] for k in ("file_name", "text")} for row in rows if row["split"] == split]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def training_crop(image):
    scale = 512 / min(image.size)
    image = image.resize(tuple(max(512, math.ceil(n * scale)) for n in image.size), Image.Resampling.LANCZOS)
    left, top = (image.width - 512) // 2, (image.height - 512) // 2
    return image.crop((left, top, left + 512, top + 512))


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int("".join("1" if b else "0" for b in (pixels[:, 1:] > pixels[:, :-1]).flat), 2)


def audit(rows):
    fingerprints, rgb_seen, dimensions = [], {}, Counter()
    total_bytes = 0
    for row in rows:
        path = ROOT / row["split"] / row["file_name"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["sha256"], path
        total_bytes += len(content)
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            assert image.size == (row["width"], row["height"]), path
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            assert digest not in rgb_seen, (path, rgb_seen.get(digest))
            rgb_seen[digest] = row["file_name"]
            dimensions[f"{image.width}x{image.height}"] += 1
            fingerprints.append((dhash(image), dhash(training_crop(image))))
    signatures = {}
    for split, count in (("train", 360), ("validation", 40)):
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == count
        expected = [{k: row[k] for k in ("file_name", "text")} for row in selected]
        actual = [json.loads(line) for line in (ROOT / split / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        assert actual == expected, f"Metadata differ from manifest: {split}"
        files = {p.name for p in (ROOT / split).iterdir() if p.is_file()}
        assert files == {r["file_name"] for r in selected} | {"metadata.jsonl"}
        signature = hashlib.sha256()
        for row in selected:
            signature.update(json.dumps([row["file_name"], row["text"]], ensure_ascii=False).encode())
            signature.update(bytes.fromhex(row["sha256"]))
        signatures[split] = signature.hexdigest()
    near_pairs = []
    minimum = 64
    for i, first in enumerate(rows):
        for j in range(i):
            if first["split"] == rows[j]["split"]:
                continue
            distance = min((fingerprints[i][k] ^ fingerprints[j][k]).bit_count() for k in (0, 1))
            minimum = min(minimum, distance)
            if distance <= 4:
                near_pairs.append([first["file_name"], rows[j]["file_name"], distance])
    assert not near_pairs, near_pairs
    people = [r for r in rows if r["cohort"] == "people100"]
    assert Counter(r["composition"] for r in people) == {"crowd": 10, "small_group": 42, "single": 48}
    assert sum(r["distant_or_wide"] for r in people) == 24
    assert Counter(r["composition"] for r in people if r["split"] == "validation") == {"crowd": 1, "small_group": 4, "single": 5}
    assert Counter(r["cohort"] for r in rows if r["split"] == "validation") == {"core300": 30, "people100": 10}
    assert Counter(r["source"] for r in rows if r["split"] == "validation") == {"dark": 26, "neon": 10, "w2k": 2, "marie": 2}
    report = {
        "selection_revision": "watercolor400-v1-2026-11-17", "images": len(rows),
        "train": 360, "validation": 40, "bytes": total_bytes,
        "sources": dict(Counter(r["source"] for r in rows)),
        "people_composition": dict(Counter(r["composition"] for r in people)),
        "people_distant_or_wide": 24, "exact_rgb_duplicates": 0,
        "cross_split_dhash_le4_original_or_crop": near_pairs,
        "minimum_cross_split_dhash_original_or_crop": minimum,
        "loader_signatures": signatures, "dimensions": dict(sorted(dimensions.items())),
    }
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dimensions"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Restore missing images from pinned sources")
    parser.add_argument("--audit", action="store_true", help="Verify files, metadata, quotas and duplicates locally")
    args = parser.parse_args()
    records, source_info = read_manifest()
    if args.download:
        restore(records, source_info)
    if args.audit or args.download:
        audit(records)
    if not (args.download or args.audit):
        parser.print_help()
