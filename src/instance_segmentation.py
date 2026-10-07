"""
Instance Segmentation Module for MultiStyleImages (ADE20K-Compliant).
Identifies individual object/tree instances and renders them using the official
ADE20K color palette with sharp instance-boundary demarcation, so ControlNet
recognizes them 100% as natural trees and foliage without confusing them with buildings or skyscrapers.
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import cv2
import numpy as np
import torch
from PIL import Image

try:
    from segmentation import SemanticSegmentor, ADE20K_PALETTE
except ImportError:
    from src.segmentation import SemanticSegmentor, ADE20K_PALETTE

try:
    from img_preprocessing import preprocess_image
except ImportError:
    try:
        from src.img_preprocessing import preprocess_image
    except ImportError:
        preprocess_image = None


class InstanceSegmentor:
    """
    Segmentatore di Istanze conforme alla palette ADE20K per ControlNet.
    Separa i singoli tronchi e alberi senza usare colori casuali:
    - Mantiene la classe semantica corretta (es. Albero = [4, 200, 3], Terra = [120, 120, 70]).
    - Traccia bordi di separazione netti tra le singole istanze per evitare che ControlNet
      fondi la foresta in un blocco unico o la scambi per edifici.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_instance_area: int = 80
    ):
        self.device = device
        self.min_instance_area = min_instance_area
        self.palette = ADE20K_PALETTE
        
        print(f"Inizializzazione InstanceSegmentor (ADE20K) su: {self.device}")
        self.semantic_segmentor = SemanticSegmentor(device=self.device)

    def process_instances(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        target_classes: Optional[List[int]] = None,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict]]:
        """
        Calcola sia la mappa delle istanze con ID univoci (1..N) sia la mappa semantica delle classi.
        
        :return: (instance_map, semantic_labels, instance_metadata)
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image.convert("RGB")

        # Preprocessing pulizia / contrasto / Lanczos
        if preprocess and preprocess_image is not None:
            pil_img = preprocess_image(pil_img, target_min_size=512, return_pil=True)

        orig_w, orig_h = pil_img.size
        semantic_labels = self.semantic_segmentor.process_labels(pil_img)
        lbl_h, lbl_w = semantic_labels.shape

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

        # Scomposizione delle aree alberate in singoli fusti/tronchi
        for class_id in target_classes:
            class_mask = (semantic_labels == class_id).astype(np.uint8) * 255
            if cv2.countNonZero(class_mask) == 0:
                continue

            # Gradiente verticale per isolare i tronchi
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_mag = np.abs(sobel_x)
            sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            edges_in_trees = cv2.bitwise_and(sobel_norm, sobel_norm, mask=class_mask)
            _, edge_thresh = cv2.threshold(edges_in_trees, 45, 255, cv2.THRESH_BINARY)
            
            separated_mask = cv2.subtract(class_mask, edge_thresh)
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
            cleaned_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, kernel_v)

            dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
            thresh_val = max(1.0, 0.15 * dist_transform.max())
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

        # Assegna il resto dello sfondo
        non_target_mask = (instance_map == 0) & (semantic_labels != 0)
        num_bg_labels, bg_labels = cv2.connectedComponents(non_target_mask.astype(np.uint8))
        for bg_id in range(1, num_bg_labels):
            bg_mask = (bg_labels == bg_id)
            if np.sum(bg_mask) > self.min_instance_area:
                instance_count += 1
                instance_map[bg_mask] = instance_count

        return instance_map, semantic_labels, instance_metadata

    def process(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        target_classes: Optional[List[int]] = None,
        preprocess: bool = True,
        boundary_width: int = 2
    ) -> Image.Image:
        """
        Genera la mappa conforme ad ADE20K:
        - Mantiene il colore semantico ADE20K per ogni classe (es. Verde per gli alberi).
        - Traccia linee di confine marcate tra i singoli alberi/istanze per istruire
          ControlNet a generare fusti e rami separati.
        """
        instance_map, semantic_labels, _ = self.process_instances(
            image, target_classes=target_classes, preprocess=preprocess
        )
        h, w = instance_map.shape

        # 1. Base ADE20K palette: ogni pixel riceve il colore della sua classe semantica corretta
        seg_color = self.palette[semantic_labels]  # Shape (H, W, 3)

        # 2. Estrazione dei contorni tra istanze adiacenti (per separare tronco da tronco)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_width, boundary_width))
        instance_boundaries = np.zeros((h, w), dtype=np.uint8)

        unique_instances = np.unique(instance_map)
        for inst_id in unique_instances:
            if inst_id == 0:
                continue
            inst_mask = (instance_map == inst_id).astype(np.uint8) * 255
            dilated = cv2.dilate(inst_mask, kernel, iterations=1)
            boundary = cv2.subtract(dilated, inst_mask)
            instance_boundaries = cv2.bitwise_or(instance_boundaries, boundary)

        # 3. I contorni tra alberi vengono colorati con il colore ADE20K del terreno/suolo o linea scura
        # In questo modo ControlNet vede una sequenza di fusti d'albero verdi separati da solchi naturali
        ground_color = self.palette[13]  # ADE20K Class 13: Earth / Ground [120, 120, 70]
        tree_mask = (semantic_labels == 4) | (semantic_labels == 9) | (semantic_labels == 17)
        tree_boundaries = (instance_boundaries == 255) & tree_mask

        # Applica il colore di demarcazione
        seg_color[tree_boundaries] = ground_color

        return Image.fromarray(seg_color.astype(np.uint8))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Esegue Instance Segmentation conforme alla palette ADE20K.")
    parser.add_argument("input_path", type=str, help="Percorso dell'immagine di input.")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Percorso di salvataggio della mappa di istanze.")
    parser.add_argument("--preprocess", action="store_true", default=True, help="Abilita il preprocessing (default: True).")
    parser.add_argument("--no_preprocess", action="store_false", dest="preprocess", help="Disabilita il preprocessing.")
    parser.add_argument("--min_area", type=int, default=80, help="Area minima in pixel (default: 80).")

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

    print(f"Elaborazione Instance Segmentation ADE20K per: {input_p}")
    segmentor = InstanceSegmentor(min_instance_area=args.min_area)
    instance_img = segmentor.process(input_p, preprocess=args.preprocess)
    instance_img.save(str(out_p))

    _, _, meta = segmentor.process_instances(input_p, preprocess=args.preprocess)
    print(f"✅ Trovate {len(meta)} istanze d'albero/oggetti distinte!")
    print(f"Mappa ADE20K salvata in: {out_p}")
