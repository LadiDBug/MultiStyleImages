"""Build landscape prompts from semantic class IDs, without loading models.

Example (reuse the same IDs for the prompt and ControlNet):

    labels = segmentor.process_labels(input_img)
    builder = PromptBuilder(segmentor.model.config.id2label)
    features = builder.extract_features(labels)
    prompt = builder.build(labels, style="Cyberpunk")
    control_image = Image.fromarray(ADE20K_PALETTE[labels])

Area thresholds only filter the text; they never modify the segmentation map.
No object counts, physical attributes or spatial relations are inferred.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math
import re

import numpy as np

@dataclass(frozen=True)
class SemanticFeature:
    """Detected class; area_ratio is image coverage, not confidence."""

    class_id: int
    name: str
    pixel_count: int
    area_ratio: float


STYLE_PROMPTS = {
    "cyberpunk": "cyberpunk style",
    "gta v": "GTA V style",
    "pixel art": "pixel art style",
}


def normalize_style(style: str) -> str:
    """Validate and canonicalize a style for all builders and the baseline."""
    if not isinstance(style, str):
        raise ValueError("style must be a supported style name.")
    key = " ".join(style.lower().split())
    if key.endswith(" style"):
        key = key[:-6]
    if key not in STYLE_PROMPTS:
        raise ValueError(f"Unsupported style {style!r}. Choose from: {', '.join(STYLE_PROMPTS)}")
    return key


class PromptBuilder:
    """Compose a deterministic prompt from the largest detected regions.

    min_area_ratio is inclusive and defaults to 1% of image area.
    max_classes limits distinct names in the prompt, not tokenizer tokens.
    These are initial configurable heuristics, to validate on development data.
    If every class is filtered out, return the style-only baseline.
    """

    def __init__(
        self,
        id2label: Mapping[int | str, str],
        min_area_ratio: float = 0.01,
        max_classes: int = 8,
    ):
        if not math.isfinite(min_area_ratio) or not 0 <= min_area_ratio <= 1:
            raise ValueError("min_area_ratio must be a finite fraction in [0, 1].")
        if isinstance(max_classes, bool) or not isinstance(max_classes, int) or max_classes < 1:
            raise ValueError("max_classes must be a positive integer.")

        self.id2label = {}
        for key, value in id2label.items():
            # Hugging Face configs can have integer or JSON string keys.
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise ValueError("Class IDs must be integers or integer strings.")
            class_id = int(key)
            if class_id < 0 or class_id in self.id2label:
                raise ValueError(f"Invalid or duplicate class ID: {key!r}")
            if not isinstance(value, str):
                raise ValueError(f"Class {class_id} must have a text label.")
            # ADE20K labels may contain comma/semicolon-separated synonyms.
            name = re.split(r"[,;]", value, maxsplit=1)[0]
            name = " ".join(name.replace("_", " ").lower().split())
            if not name:
                raise ValueError(f"Class {class_id} has an empty label.")
            self.id2label[class_id] = name
        if not self.id2label:
            raise ValueError("id2label must not be empty.")
        self.min_area_ratio = min_area_ratio
        self.max_classes = max_classes

    def extract_features(self, labels: np.ndarray) -> list[SemanticFeature]:
        """Return all detected classes, sorted by area then ascending class ID.

        Input must be a nonempty H x W integer ID map, not an RGB map.
        Unmapped IDs fail explicitly rather than silently hiding scene content.
        """
        labels = np.asarray(labels)
        if labels.ndim != 2 or labels.size == 0:
            raise ValueError("Expected a nonempty 2D class-ID map (H, W).")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Class-ID map must contain integers.")
        ids, counts = np.unique(labels, return_counts=True)
        missing = [int(class_id) for class_id in ids if int(class_id) not in self.id2label]
        if missing:
            raise ValueError(f"Class IDs missing from id2label: {missing}")
        features = [
            SemanticFeature(
                class_id=int(class_id),
                name=self.id2label[int(class_id)],
                pixel_count=int(count),
                area_ratio=int(count) / labels.size,
            )
            for class_id, count in zip(ids, counts)
        ]
        return sorted(features, key=lambda feature: (-feature.pixel_count, feature.class_id))

    def select_features(self, labels: np.ndarray) -> list[SemanticFeature]:
        """Return the same ordered, filtered classes used by every backend."""
        selected = []
        names = set()
        for feature in self.extract_features(labels):
            if feature.area_ratio >= self.min_area_ratio and feature.name not in names:
                selected.append(feature)
                names.add(feature.name)
                if len(selected) == self.max_classes:
                    break
        return selected

    def build(self, labels: np.ndarray, style: str) -> str:
        """Return content + style text, preserving the model's class meanings.

        For example, 'water' stays 'water', not 'lake'; 'person' carries no
        claim about identity or number. Style cannot introduce scene objects.
        """
        style_key = normalize_style(style)
        names = [feature.name for feature in self.select_features(labels)]
        style_prompt = STYLE_PROMPTS[style_key]
        if not names:
            return style_prompt
        content = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return f"a landscape containing {content}, {style_prompt}"
