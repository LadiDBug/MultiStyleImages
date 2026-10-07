"""
Instance Segmentation Module for MultiStyleImages.
Differentiates individual object instances (e.g. Tree #1, Tree #2, Tree #3)
assigning each individual instance a distinct color and ID.
Includes optional preprocessing to clean and enhance low-resolution images.
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import cv2
import numpy as np
import torch
from PIL import Image

try:
    from segmentation import SemanticSegmentor
except ImportError:
    from src.segmentation import SemanticSegmentor

try:
    from img_preprocessing import preprocess_image
except ImportError:
    try:
        from src.img_preprocessing import preprocess_image
    except ImportError:
        preprocess_image = None


def generate_distinct_colors(num_colors: int = 512) -> np.ndarray:
    """Genera una palette di colori massimamente distinti per le istanze."""
    np.random.seed(42)
    colors = np.zeros((num_colors, 3), dtype=np.uint8)
    colors[0] = [30, 30, 30]  # Sfondo neutro
    
    for i in range(1, num_colors):
        hue = int((i * 137.5) % 180)  # Golden angle
        sat = np.random.randint(180, 255)
        val = np.random.randint(190, 255)
        hsv_pixel = np.uint8([[[hue, sat, val]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0, 0]
        colors[i] = rgb_pixel
    return colors


class InstanceSegmentor:
    """
    Segmentatore di Istanze.
    Identifica e separa i singoli tronchi e alberi attribuendo a ciascuna istanza
    una maschera e un colore univoco.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_instance_area: int = 100
    ):
        self.device = device
        self.min_instance_area = min_instance_area
        self.palette = generate_distinct_colors(512)
        
        print(f"Inizializzazione InstanceSegmentor su: {self.device}")
        self.semantic_segmentor = SemanticSegmentor(device=self.device)

    def process_instances(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        target_classes: Optional[List[int]] = None,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """
        Calcola la mappa di segmentazione per istanze (H, W).
        Ogni singolo oggetto ha un ID univoco (1, 2, 3, ... N).
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image.convert("RGB")

        # 1. Applicazione del preprocessing se richiesto
        if preprocess and preprocess_image is not None:
            pil_img = preprocess_image(pil_img, target_min_size=512, return_pil=True)

        orig_w, orig_h = pil_img.size
        
        # 2. Ottiene la mappa semantica di base
        semantic_labels = self.semantic_segmentor.process_labels(pil_img)
        lbl_h, lbl_w = semantic_labels.shape

        # Allinea rgb_arr alle dimensioni di semantic_labels
        if (orig_h, orig_w) != (lbl_h, lbl_w):
            resized_pil = pil_img.resize((lbl_w, lbl_h), resample=Image.Resampling.LANCZOS)
            rgb_arr = np.array(resized_pil)
        else:
            rgb_arr = np.array(pil_img)

        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

        if target_classes is None:
            target_classes = [4, 9, 17]  # 4=Tree, 9=Grass, 17=Plant

        instance_map = np.zeros((lbl_h, lbl_w), dtype=np.int32)
        instance_count = 0
        instance_metadata = {}

        # 3. Scomposizione delle aree alberate in singoli fusti/tronchi
        for class_id in target_classes:
            class_mask = (semantic_labels == class_id).astype(np.uint8) * 255
            if cv2.countNonZero(class_mask) == 0:
                continue

            # Riconoscimento delle linee verticali dei tronchi tramite gradiente di Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_mag = np.abs(sobel_x)
            sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Rimuove i bordi verticali dalla maschera per staccare i tronchi adiacenti
            edges_in_trees = cv2.bitwise_and(sobel_norm, sobel_norm, mask=class_mask)
            _, edge_thresh = cv2.threshold(edges_in_trees, 50, 255, cv2.THRESH_BINARY)
            
            separated_mask = cv2.subtract(class_mask, edge_thresh)
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
            cleaned_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, kernel_v)

            # Distance transform per trovare i centri di ciascun tronco
            dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
            thresh_val = max(1.0, 0.18 * dist_transform.max())
            _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, 0)
            sure_fg = np.uint8(sure_fg)

            num_labels, labels = cv2.connectedComponents(sure_fg)
            sure_bg = cv2.dilate(class_mask, np.ones((3, 3), np.uint8), iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)

            markers = labels.copy().astype(np.int32) + 1
            markers[unknown == 255] = 0

            markers = cv2.watershed(rgb_arr.copy(), markers)

            for marker_id in range(2, num_labels + 1):
                mask_inst = (markers == marker_id) & (class_mask == 255)
                area = np.sum(mask_inst)
                if area >= self.min_instance_area:
                    instance_count += 1
                    instance_map[mask_inst] = instance_count
                    
                    coords = np.argwhere(mask_inst)
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    
                    instance_metadata[instance_count] = {
                        "class_id": class_id,
                        "class_name": "tree" if class_id == 4 else "vegetation",
                        "area_pixels": int(area),
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
                    }

        # 4. Assegna istanze separate per le parti rimanenti (terreno / suolo)
        non_target_mask = (instance_map == 0) & (semantic_labels != 0)
        num_bg_labels, bg_labels = cv2.connectedComponents(non_target_mask.astype(np.uint8))
        for bg_id in range(1, num_bg_labels):
            bg_mask = (bg_labels == bg_id)
            if np.sum(bg_mask) > self.min_instance_area:
                instance_count += 1
                instance_map[bg_mask] = instance_count

        return instance_map, instance_metadata

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        target_classes: Optional[List[int]] = None,
        preprocess: bool = True
    ) -> Image.Image:
        """
        Ritorna la mappa visiva RGB dove ogni singolo albero/oggetto ha un colore distintivo diverso.
        """
        instance_map, _ = self.process_instances(image, target_classes=target_classes, preprocess=preprocess)
        
        h, w = instance_map.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_ids = np.unique(instance_map)
        for inst_id in unique_ids:
            if inst_id == 0:
                color_map[instance_map == 0] = [20, 20, 20]
            else:
                color_idx = inst_id % len(self.palette)
                color_map[instance_map == inst_id] = self.palette[color_idx]

        return Image.fromarray(color_map)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Esegue Instance Segmentation distinguendo ogni singolo albero/oggetto.")
    parser.add_argument("input_path", type=str, help="Percorso dell'immagine di input.")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Percorso di salvataggio della mappa di istanze.")
    parser.add_argument("--preprocess", action="store_true", default=True, help="Abilita il preprocessing (default: True).")
    parser.add_argument("--no_preprocess", action="store_false", dest="preprocess", help="Disabilita il preprocessing.")
    parser.add_argument("--min_area", type=int, default=100, help="Area minima in pixel per considerare un'istanza valida (default: 100).")

    args = parser.parse_args()

    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Errore: il file {input_p} non esiste.")
        sys.exit(1)

    if args.output_path:
        out_p = Path(args.output_path)
        if out_p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            out_p.mkdir(parents=True, exist_ok=True)
            out_p = out_p / f"{input_p.stem}_instances.png"
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_p = input_p.parent / f"{input_p.stem}_instances.png"

    print(f"Elaborazione Instance Segmentation (preprocess={args.preprocess}) per: {input_p}")
    segmentor = InstanceSegmentor(min_instance_area=args.min_area)
    instance_img = segmentor.process(input_p, preprocess=args.preprocess)
    instance_img.save(str(out_p))

    _, meta = segmentor.process_instances(input_p, preprocess=args.preprocess)
    print(f"✅ Trovate {len(meta)} istanze individuali distinte!")
    print(f"Mappa delle istanze salvata in: {out_p}")
