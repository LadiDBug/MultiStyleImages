"""Indice dei risultati salvati: nessun modello o accesso GPU."""

from dataclasses import dataclass
from pathlib import Path
import re

from demo.backend import CONDITIONS, ROOT, STYLES

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
DEFAULT_RESULTS = {
    **{f'baseline_seed_{seed}': f'outputs/baseline_seed_{seed}' for seed in (42, 123, 456)},
    **{f'rank_{rank}': f'outputs/rank_{rank}' for rank in (4, 8, 16, 32)},
}
STYLE_LABELS = {'gta5': 'GTA5', 'pixel_art': 'Pixel art', 'watercolor': 'Watercolor'}
MAP_LABELS = {'none': 'Nessuna', 'canny': 'Canny', 'depth': 'Depth', 'segmentation': 'Segmentation'}


@dataclass(frozen=True)
class SavedResult:
    key: str
    image_id: str
    experiment: str
    style: str
    condition: str
    seed: int
    path: Path
    run: str

    @property
    def label(self):
        return f'{self.experiment} · {STYLE_LABELS[self.style]} · {MAP_LABELS[self.condition]} · seed {self.seed}'


class TestGallery:
    def __init__(self, root=ROOT, *, image_paths=None, result_paths=None):
        self.root = Path(root).resolve()
        self.image_paths = self._paths({'our_ds': 'our_ds'} if image_paths is None else image_paths)
        self.result_paths = self._paths(DEFAULT_RESULTS if result_paths is None else result_paths)
        self.results = {}
        self.originals = {}
        self.warnings = []
        self._scan()

    def _paths(self, values):
        paths = {}
        for name, value in values.items():
            path = Path(value).expanduser()
            paths[name] = (path if path.is_absolute() else self.root / path).resolve()
        return paths

    @staticmethod
    def _inside(path, root):
        return path.resolve().is_relative_to(root)

    def _scan(self):
        datasets = {}
        for name, root in self.image_paths.items():
            if not root.is_dir():
                self.warnings.append(f'Dataset {name}: cartella non trovata ({root}).')
                continue
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and self._inside(path, root):
                    image_id = path.relative_to(root).as_posix()
                    datasets.setdefault(image_id, []).append(path.resolve())
        fallbacks = {}
        for experiment, root in self.result_paths.items():
            if not root.is_dir():
                self.warnings.append(f'Esperimento {experiment}: cartella non trovata ({root}).')
                continue
            count = 0
            # Supporta la radice dell'esportazione Kaggle, outputs/, il run o samples/.
            for images_dir in sorted(root.rglob('images')):
                if not images_dir.is_dir():
                    continue
                sample = images_dir.parent
                parts = sample.parts
                if 'samples' not in parts:
                    continue
                sample_index = len(parts) - 1 - parts[::-1].index('samples')
                image_id = Path(*parts[sample_index + 1:]).as_posix()
                if image_id == '.':
                    continue
                run = Path(*parts[:sample_index])
                for path in sorted(images_dir.glob('*/*/*')):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or not self._inside(path, root):
                        continue
                    style, condition = path.relative_to(images_dir).parts[:2]
                    match = re.search(r'_seed(\d+)$', path.stem)
                    if style not in STYLES or condition not in CONDITIONS or not match:
                        continue
                    key = f'{experiment}/{path.relative_to(root).as_posix()}'
                    self.results[key] = SavedResult(key, image_id, experiment, style, condition,
                                                    int(match[1]), path.resolve(), run.name)
                    count += 1
                original = sample / 'input_original.png'
                if original.is_file() and self._inside(original, root):
                    fallbacks.setdefault(image_id, original.resolve())
            if not count:
                self.warnings.append(f'Esperimento {experiment}: nessun risultato riconosciuto.')
        # Non associa per nome file quando più immagini hanno lo stesso nome.
        by_name = {}
        for image_id, paths in datasets.items():
            by_name.setdefault(Path(image_id).name, []).extend(paths)
        for image_id in sorted(set(datasets) | {r.image_id for r in self.results.values()}):
            exact = datasets.get(image_id, [])
            candidates = exact or by_name.get(Path(image_id).name, [])
            unique = set(candidates)
            self.originals[image_id] = next(iter(unique)) if len(unique) == 1 else fallbacks.get(image_id)

    @property
    def image_ids(self):
        return sorted(self.originals)

    @property
    def experiments(self):
        return list(dict.fromkeys(r.experiment for r in self.results.values()))

    @property
    def allowed_paths(self):
        # Autorizza soltanto le immagini indicizzate, mai l'intera esportazione con i pesi.
        return sorted({str(r.path) for r in self.results.values()} |
                      {str(p) for p in self.originals.values() if p is not None})

    def matching(self, image_id, experiments, styles, conditions):
        return [r for r in self.results.values() if r.image_id == image_id
                and r.experiment in (experiments or []) and r.style in (styles or [])
                and r.condition in (conditions or [])]

    def choices(self, image_id, experiments, styles, conditions):
        rows = self.matching(image_id, experiments, styles, conditions)
        labels = [r.label for r in rows]
        return [(r.label + (f' · {r.run}' if labels.count(r.label) > 1 else ''), r.key) for r in rows]

    def view(self, image_id, keys):
        rows = [self.results[key] for key in (keys or [])
                if key in self.results and self.results[key].image_id == image_id]
        original = self.originals.get(image_id)
        return (str(original) if original else None,
                [(str(r.path), f'{r.label} · {r.run}') for r in rows],
                f'{len(rows)} risultati selezionati per questa immagine.')
