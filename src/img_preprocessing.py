"""
Image Preprocessing Module for MultiStyleImages.
Provides clean, artifact-free image enhancement designed specifically
for Semantic Segmentation (UperNet / ADE20K) and ControlNet Diffusion pipelines.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def _to_cv2_bgr(image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
    """Convert input (PIL Image, np.ndarray, file path) to an OpenCV BGR numpy array."""
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Impossibile caricare l'immagine da: {image}")
        return img_bgr
    elif isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image.copy()
    raise TypeError(f"Tipo di immagine non supportato: {type(image)}")


def _to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL Image in RGB format."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def autocrop_black_borders(img_bgr: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Rimuove eventuali bordi neri derivanti da screenshot o padding."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img_bgr[y0:y1, x0:x1]
    return img_bgr


def enhance_color_and_contrast_lab(
    img_bgr: np.ndarray,
    clahe_clip: float = 1.8,
    color_boost: float = 1.35,
    tile_grid: Tuple[int, int] = (6, 6)
) -> np.ndarray:
    """
    Separa ed esalta i canali cromatici in spazio LAB:
    - Canale L: CLAHE controllato per bilanciare luce/ombra senza bruciare i pixel.
    - Canali A e B: esaltazione del contrasto colore (separa il verde degli alberi dal marrone del terreno).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1. CLAHE controllato su L
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    l_enh = clahe.apply(l)

    # 2. Spinta di contrasto cromatico sui canali A e B (senza introdurre artefatti di frequenza)
    a_f = (a.astype(np.float32) - 128.0) * color_boost + 128.0
    b_f = (b.astype(np.float32) - 128.0) * color_boost + 128.0
    a_enh = np.clip(a_f, 0, 255).astype(np.uint8)
    b_enh = np.clip(b_f, 0, 255).astype(np.uint8)

    merged = cv2.merge([l_enh, a_enh, b_enh])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_min_size: int = 512,
    color_boost: float = 1.35,
    clahe_clip: float = 1.8,
    bilateral_smooth: bool = True,
    autocrop: bool = True,
    return_pil: bool = True
) -> Union[Image.Image, np.ndarray]:
    """
    Pipeline pulita per immagini a bassa risoluzione / degradate:
    
    1. Autocrop bordi scuri (se presenti).
    2. Upscaling Lanczos a dimensione target (>= 512px).
    3. Filtro Bilaterale cromatico morbido (unifica le chiazze di luce/foglie senza creare aloni).
    4. Separazione cromatica LAB (stacca alberi, fogliame e suolo).
    5. Micro-unsharp leggerissimo (senza artefatti ad anello).
    
    :param image: Immagine di input (PIL Image, np.ndarray o path).
    :param target_min_size: Dimensione minima desiderata per il lato più lungo.
    :param color_boost: Fattore di separazione cromatica (consigliato 1.2 - 1.5).
    :param clahe_clip: Contrasto locale di luminanza (consigliato 1.5 - 2.2).
    :param bilateral_smooth: Unifica le macchie di luce casuali.
    :param autocrop: Rimuove cornici nere da screenshot.
    :param return_pil: Se True restituisce una PIL.Image.
    """
    img = _to_cv2_bgr(image)

    # 1. Rimuove eventuali cornici nere
    if autocrop:
        img = autocrop_black_borders(img)

    # 2. Upscaling Lanczos
    h, w = img.shape[:2]
    max_side = max(h, w)
    if target_min_size is not None and max_side < target_min_size:
        scale = target_min_size / max_side
        new_w = max(64, int(w * scale))
        new_h = max(64, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Lisciatura bilaterale cromatica morbida (elimina le macchioline senza cancellare i contorni)
    if bilateral_smooth:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=55, sigmaSpace=10)

    # 4. Esaltazione contrasto e separazione cromatica LAB
    img = enhance_color_and_contrast_lab(
        img,
        clahe_clip=clahe_clip,
        color_boost=color_boost,
        tile_grid=(6, 6)
    )

    # 5. Micro-unsharp conservativo (molto lieve per evitare cerchi neri)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.2, blur, -0.2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)

    print("Preprocessing completato. Immagine pronta per la segmentazione semantica o il pipeline di diffusione.")
    if return_pil:
        return _to_pil_rgb(img)
    return img


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Migliora una singola immagine per la segmentazione semantica.")
    parser.add_argument("input_path", type=str, help="Percorso dell'immagine di input.")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Percorso di salvataggio dell'immagine migliorata.")
    parser.add_argument("--min_size", type=int, default=512, help="Dimensione minima (default: 512).")
    parser.add_argument("--color_boost", type=float, default=1.35, help="Fattore di separazione cromatica (default: 1.35).")
    parser.add_argument("--clahe_clip", type=float, default=1.8, help="Contrasto locale CLAHE (default: 1.8).")
    parser.add_argument("--no_smooth", action="store_true", help="Disabilita il smoothing bilaterale.")

    args = parser.parse_args()

    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Errore: il file {input_p} non esiste.")
        sys.exit(1)

    out_p = Path(args.output_path) if args.output_path else input_p.parent / f"{input_p.stem}_enhanced{input_p.suffix}"

    print(f"Elaborazione di: {input_p}")
    enhanced = preprocess_image(
        input_p,
        target_min_size=args.min_size,
        color_boost=args.color_boost,
        clahe_clip=args.clahe_clip,
        bilateral_smooth=not args.no_smooth,
        return_pil=True
    )
    enhanced.save(str(out_p))
    print(f"Immagine migliorata salvata in: {out_p}")