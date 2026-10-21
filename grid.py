from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# MODIFICA QUI
png = "0007x2"
run_dir = Path(
    "/kaggle/working/MultiStyleImages/outputs/20261021T095714_975703Z"
)
seed = 42

images_dir = run_dir / "samples" / f"{png}.png" / "images"
image_name = f"{png}_seed{seed}.png"

styles = {
    "gta5": "GTA V",
    "pixel_art": "Pixel Art",
    "watercolor": "Watercolor",
}

conditions = {
    "none": "None — baseline",
    "canny": "Canny",
    "depth": "Depth",
    "segmentation": "Segmentation",
}

for style_folder, style_title in styles.items():
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 12), constrained_layout=True
    )
    fig.suptitle(
        f"{style_title} — {png} — seed {seed}",
        fontsize=20,
        fontweight="bold",
    )

    for ax, (condition, title) in zip(axes.flat, conditions.items()):
        ax.set_title(title, fontsize=16)
        ax.axis("off")

        path = images_dir / style_folder / condition / image_name

        if not path.is_file():
            ax.text(
                0.5, 0.5,
                f"File non trovato:\n{path}",
                ha="center",
                va="center",
                wrap=True,
                transform=ax.transAxes,
                color="red",
            )
            continue

        with Image.open(path) as img:
            ax.imshow(ImageOps.exif_transpose(img).convert("RGB"))

    plt.show()