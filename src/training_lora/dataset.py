"""Dataset immagini/caption per la LoRA Pixel Art, senza modelli generativi.

Ogni split contiene immagini e metadata.jsonl con file_name e text.
Il resize conserva le proporzioni, poi applica un crop centrale quadrato.
Nearest-neighbor e il default, anche per l'ingrandimento 128 -> 512.
Nessun flip, jitter o crop casuale: le caption restano coerenti con l'input.

Controllo indipendente, senza download o training:
    python -m src.training_lora.dataset --data-dir data/style_pixel_art/v1/train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageCaption:
    file_name: str
    path: Path
    text: str


class PixelArtDataset(Dataset):
    """Carica uno split e tokenizza senza troncare silenziosamente le caption.

    tokenizer=None consente audit e lettura delle immagini senza Transformers.
    Con un tokenizer CLIP, input_ids ha lunghezza model_max_length.
    pixel_values ha shape (3, resolution, resolution), float32 in [-1, 1].
    """

    def __init__(self, data_dir: str | Path, tokenizer: Any = None,
                 resolution: int = 512, interpolation: str = "nearest"):
        if not isinstance(resolution, int) or resolution < 8 or resolution % 8:
            raise ValueError("resolution deve essere un intero positivo multiplo di 8.")
        modes = {"nearest": Image.Resampling.NEAREST, "lanczos": Image.Resampling.LANCZOS}
        if interpolation not in modes:
            raise ValueError("interpolation deve essere nearest oppure lanczos.")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.resolution = resolution
        self.interpolation = modes[interpolation]
        self.metadata_path = self.data_dir / "metadata.jsonl"
        self.records: list[ImageCaption] = []
        seen = set()
        with self.metadata_path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    name, text = row["file_name"], row["text"]
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("file_name vuoto o non testuale")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("text vuoto o non testuale")
                    path = (self.data_dir / name).resolve()
                    if Path(name).is_absolute() or not path.is_relative_to(self.data_dir):
                        raise ValueError("file_name deve rimanere nello split")
                    if path in seen:
                        raise ValueError(f"file immagine duplicato: {name}")
                    if not path.is_file():
                        raise ValueError(f"immagine mancante: {name}")
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{self.metadata_path}:{line_no}: {error}") from error
                self.records.append(ImageCaption(name, path, text.strip()))
                seen.add(path)
        if not self.records:
            raise ValueError(f"Dataset vuoto: {self.metadata_path}")
        self.input_ids = None
        if tokenizer is not None:
            max_length = tokenizer.model_max_length
            if not isinstance(max_length, int) or not 1 <= max_length <= 4096:
                raise ValueError("Il tokenizer deve definire un model_max_length valido.")
            texts = [r.text for r in self.records]
            tokens = tokenizer(texts, padding=False, truncation=False)["input_ids"]
            for record, ids in zip(self.records, tokens):
                if len(ids) > max_length:
                    raise ValueError(
                        f"Caption troppo lunga ({len(ids)} > {max_length} token): "
                        f"{record.file_name}. Accorciare la caption prima del training."
                    )
            self.input_ids = tokenizer(
                texts, padding="max_length", max_length=max_length,
                truncation=False, return_tensors="pt",
            )["input_ids"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        try:
            with Image.open(record.path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                width, height = image.size
                scale = self.resolution / min(width, height)
                size = (max(self.resolution, math.ceil(width * scale)),
                        max(self.resolution, math.ceil(height * scale)))
                image = image.resize(size, self.interpolation)
                left = (image.width - self.resolution) // 2
                top = (image.height - self.resolution) // 2
                image = image.crop((left, top, left+self.resolution, top+self.resolution))
                pixels = np.array(image, dtype=np.float32) / 127.5 - 1.0
        except (OSError, ValueError) as error:
            raise ValueError(f"Impossibile preparare {record.path}: {error}") from error
        item = {"pixel_values": torch.from_numpy(pixels).permute(2, 0, 1).contiguous()}
        if self.input_ids is not None:
            item["input_ids"] = self.input_ids[index]
        return item

    def audit(self) -> dict:
        """Decodifica tutti i file; firma anche byte, caption e nomi relativi.

        Gli hash RGB permettono di controllare overlap fra split anche quando
        lo stesso contenuto e salvato con nomi o codifiche differenti.
        """
        signature = hashlib.sha256()
        rgb_hashes = set()
        dimensions: dict[str, int] = {}
        for record in self.records:
            signature.update(json.dumps([record.file_name, record.text], ensure_ascii=False).encode())
            signature.update(hashlib.sha256(record.path.read_bytes()).digest())
            with Image.open(record.path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                key = f"{image.width}x{image.height}"
                dimensions[key] = dimensions.get(key, 0) + 1
                digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                if digest in rgb_hashes:
                    raise ValueError(f"Immagine duplicata nello split: {record.file_name}")
                rgb_hashes.add(digest)
        return {"count": len(self), "signature": signature.hexdigest(),
                "dimensions": dimensions, "rgb_hashes": sorted(rgb_hashes)}


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--interpolation", choices=["nearest", "lanczos"], default="nearest")
    args = parser.parse_args()
    dataset = PixelArtDataset(args.data_dir, resolution=args.resolution, interpolation=args.interpolation)
    result = dataset.audit()
    result.pop("rgb_hashes")
    # Verifica anche il preprocessing senza caricare un modello.
    for index in range(len(dataset)):
        assert dataset[index]["pixel_values"].shape == (3, args.resolution, args.resolution)
    print(json.dumps(result, indent=2))
