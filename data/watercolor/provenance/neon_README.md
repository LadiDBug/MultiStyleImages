---
license: apache-2.0
task_categories:
- text-to-image
language:
- en
tags:
- synthetic
- gpt-generated
- watercolor-anime
- watercolor
- anime
- image-caption-pairs
pretty_name: GPT Watercolor Anime Style Images
size_categories:
- n<1K
---

# GPT Watercolor Anime Style Images

## Dataset Description

This is a synthetic GPT-generated Watercolor Anime Style image dataset. It contains 150 image-caption pairs with transparent watercolor washes, soft ink linework, pale paper texture, muted colors, and traditional anime illustration scenes.

The images focus on soft watercolor washes, expressive linework, gentle lighting, quiet interiors, nature scenes, village streets, character studies, everyday activities, coastal outings, family moments, and calm storybook moods.

## Contents

```text
images/
metadata.jsonl
README.md
.gitattributes
```

## Fields

Each row in `metadata.jsonl` contains:

```text
file_name: relative path to the image
text: image caption string
```

Example caption style:

```text
watercolor anime, a garden table with tea cups and fresh flowers, soft daylight through leaves, cozy traditional anime illustration
```

## Usage

This dataset uses the Hugging Face image-folder convention. The `metadata.jsonl` file points to each image under `images/` and provides the matching caption, so the Dataset Viewer exposes the image and text columns with rendered thumbnails.
The dataset is intentionally kept in this simple layout so it can be loaded directly with the standard image-folder loader.

The intended trigger/caption prefix is:

```text
watercolor anime
```

## Intended Use

This dataset is intended for style experimentation, captioning experiments, image-generation research, and reproducible synthetic-dataset inspection. It is not intended to represent real artists, real people, factual events, or documentary subject matter.

## Limitations

The dataset is synthetic and stylistically narrow. Systems using it may overemphasize watercolor washes, ink outlines, muted palettes, glowing light, traditional clothing, bamboo groves, rainy streets, cozy interiors, and tranquil landscapes.

## License

This dataset is released under the Apache License 2.0.
