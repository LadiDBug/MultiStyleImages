# This file is used to segment the original image into a segmentation map
# this part is important beacuse we want to modify the original image

from pathlib import Path
import torch 
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np 

# Official ADE20K palette from MMSegmentation:
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py
# UPerNet predicts IDs 0..149 (0 = wall), with no background entry or offset.
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)

class SemanticSegmentor:
    def __init__(
        self,
        model_name: str = "openmmlab/upernet-convnext-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Using device: {self.device} with model: {model_name}")

        # AutoImageProcessor downloads and prepares the model for semantic segmentation
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # Load the pre-trained UperNet model for semantic segmentation
        self.model = UperNetForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()

    def process(self, image: Image.Image) -> Image.Image:
        """Return the RGB ADE20K conditioning map expected by ControlNet."""
        pred_seg = self.process_labels(image)
        return Image.fromarray(ADE20K_PALETTE[pred_seg])

    def process_labels(self, image: Image.Image) -> np.ndarray:
        """Return original ADE20K IDs (H, W), with pre-upscaling for low-resolution inputs."""
        w, h = image.size

        # Pre-upscale small images (<512px) to 512px with Lanczos so ConvNeXt operates at optimal feature scale
        if max(w, h) < 512:
            scale = 512 / max(w, h)
            target_size = (max(64, int(w * scale)), max(64, int(h * scale)))
            proc_img = image.convert("RGB").resize(target_size, resample=Image.Resampling.LANCZOS)
        else:
            proc_img = image.convert("RGB")

        inputs = self.processor(images=proc_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits

        # Upsample the logits to match the processed image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=proc_img.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        return upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def _get_palette(self):
        return ADE20K_PALETTE.tolist()

if __name__ == "__main__":
    import argparse
    import sys
    from PIL import Image

    parser = argparse.ArgumentParser(description="Genera la mappa di segmentazione ADE20K per una singola immagine.")
    parser.add_argument("input_path", nargs="?", default="test_images/paesaggio-1024x683.png",
                        help="Percorso dell'immagine di input (default: test_images/paesaggio-1024x683.png).")
    parser.add_argument("-o", "--output_path", type=str, default=None,
                        help="Percorso del file in cui salvare la mappa di segmentazione.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Applica prima il preprocessing per migliorare l'immagine prima di segmentarla.")

    args = parser.parse_args()
    input_p = Path(args.input_path) if "Path" in globals() else sys.modules["pathlib"].Path(args.input_path)

    if not input_p.exists():
        print(f"Errore: Impossibile trovare l'immagine al percorso '{input_p}'!")
        sys.exit(1)

    out_p = Path(args.output_path) if args.output_path else input_p.parent / f"{input_p.stem}_seg_map.png"

    print(f"Caricamento dell'immagine: {input_p}...")
    input_img = Image.open(str(input_p)).convert("RGB")

    if args.preprocess:
        try:
            from img_preprocessing import preprocess_image
        except ImportError:
            from src.img_preprocessing import preprocess_image
        print("Applicazione del preprocessing anti-rumore / bilanciamento colore...")
        input_img = preprocess_image(input_img, return_pil=True)

    print("Inizializzazione del segmentatore UperNet...")
    segmentor = SemanticSegmentor()

    print("Elaborazione della mappa semantica...")
    output_seg = segmentor.process(input_img)

    output_seg.save(str(out_p))
    print(f"Fatto! La mappa semantica è stata salvata in: '{out_p}'")