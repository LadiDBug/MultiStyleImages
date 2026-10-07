"""
Instance Segmentation Module for MultiStyleImages.
Differentiates individual object instances (e.g. Tree #1, Tree #2, Tree #3)
assigning each individual instance a distinct color and ID.
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


def generate_distinct_colors(num_colors: int = 256) -> np.ndarray:
    """Genera una palette di colori massimamente distinti per le istanze."""
    np.random.seed(42)
    colors = np.zeros((num_colors, 3), dtype=np.uint8)
    # Colore 0 = Sfondo / Neutro (Grigio scuro / nero)
    colors[0] = [30, 30, 30]
    
    # Genera colori brillanti e distinti nello spazio HSV
    for i in range(1, num_colors):
        hue = int((i * 137.5) % 180)  # Angolo aureo per massimizzare la distanza cromatica
        sat = np.random.randint(180, 255)
        val = np.random.randint(190, 255)
        hsv_pixel = np.uint8([[[hue, sat, val]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0, 0]
        colors[i] = rgb_pixel
    return colors


class InstanceSegmentor:
    """
    Segmentatore di Istanze.
    Identifica e separa i singoli oggetti (es. singoli tronchi d'albero, fusti e rami)
    attribuendo a ciascuna istanza una maschera e un colore univoco.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_instance_area: int = 150
    ):
        self.device = device
        self.min_instance_area = min_instance_area
        self.palette = generate_distinct_colors(512)
        
        print(f"Inizializzazione InstanceSegmentor su: {self.device}")
        self.semantic_segmentor = SemanticSegmentor(device=self.device)

    def process_instances(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        target_classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """
        Calcola la mappa di segmentazione per istanze (H, W).
        Ogni singolo oggetto ha un ID univoco (1, 2, 3, ... N).
        
        :param image: Immagine di input (PIL Image, np.ndarray, o Path).
        :param target_classes: Lista di classi semantiche da separare in istanze (default: [4] = Tree).
        :return: (instance_map, instance_metadata)
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size
        # 1. Ottiene la mappa semantica di base
        semantic_labels = self.semantic_segmentor.process_labels(pil_img)

        # Se target_classes non è specificato, separa la classe alberi (4) e vegetazione (9, 17)
        if target_classes is None:
            target_classes = [4, 9, 17]

        # Mappa finale delle istanze (0 = sfondo)
        instance_map = np.zeros((h, w), dtype=np.int32)
        instance_count = 0
        instance_metadata = {}

        # 2. Per ogni classe target, effettua la scomposizione per istanze (Watershed / Distance Transform)
        for class_id in target_classes:
            class_mask = (semantic_labels == class_id).astype(np.uint8) * 255
            if cv2.countNonZero(class_mask) == 0:
                continue

            # Applica una pulizia morfologica per isolare i rami/tronchi
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            # Distance Transform per trovare i centri/fusti di ogni singolo albero
            dist_transform = cv2.distanceTransform(opened_mask, cv2.DIST_L2, 5)
            
            # Soglia per determinare i nuclei certi di ogni albero
            threshold_val = 0.20 * dist_transform.max()
            _, sure_fg = cv2.threshold(dist_transform, threshold_val, 255, 0)
            sure_fg = np.uint8(sure_fg)

            # Trova le singole componenti connesse (ogni albero indipendente)
            num_labels, labels = cv2.connectedComponents(sure_fg)

            # Area sconosciuta (bordi tra alberi vicini)
            sure_bg = cv2.dilate(opened_mask, kernel, iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Prepara i marker per Watershed
            markers = labels.copy()
            markers = markers + 1
            markers[unknown == 255] = 0

            # Watershed su immagine a 3 canali
            rgb_arr = np.array(pil_img)
            markers = cv2.watershed(rgb_arr, markers)

            # Assegna un ID univoco a ogni istanza trovata
            for marker_id in range(2, num_labels + 1):
                mask_inst = (markers == marker_id)
                area = np.sum(mask_inst)
                if area >= self.min_instance_area:
                    instance_count += 1
                    instance_map[mask_inst] = instance_count
                    
                    # Calcola bounding box
                    coords = np.argwhere(mask_inst)
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    
                    instance_metadata[instance_count] = {
                        "class_id": class_id,
                        "class_name": "tree" if class_id == 4 else "vegetation",
                        "area_pixels": int(area),
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
                    }

        # Per il terreno e le altre classi non-target, assegna regioni di sfondo separate
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
        target_classes: Optional[List[int]] = None
    ) -> Image.Image:
        """
        Ritorna la mappa visiva RGB dove ogni singolo albero/oggetto ha un colore distintivo diverso.
        """
        instance_map, _ = self.process_instances(image, target_classes=target_classes)
        
        # Mappa gli ID delle istanze ai colori della palette
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
    parser.add_argument("--min_area", type=int, default=150, help="Area minima in pixel per considerare un'istanza valida (default: 150).")

    args = parser.parse_args()

    input_p = Path(args.input_path)
    if not input_p.exists():
        print(f"Errore: il file {input_p} non esiste.")
        sys.exit(1)

    out_p = Path(args.output_path) if args.output_path else input_p.parent / f"{input_p.stem}_instances.png"

    print(f"Elaborazione Instance Segmentation per: {input_p}")
    segmentor = InstanceSegmentor(min_instance_area=args.min_area)
    instance_img = segmentor.process(input_p)
    instance_img.save(str(out_p))

    # Stampa riepilogo
    _, meta = segmentor.process_instances(input_p)
    print(f"✅ Trovate {len(meta)} istanze individuali distinte!")
    print(f"Mappa delle istanze salvata in: {out_p}")
