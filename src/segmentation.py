# This file is used to segment the original image into a segmentation map
# this part is important beacuse we want to modify the original image

import torch 
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np 

class SemanticSegmentor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")

        # AutoImagePRocessor download and prepare the model for semantic segmentation
        self.processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        # Load the pre-trained UperNet model for semantic segmentation
        self.model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(self.device)

    def process(self, image: Image.Image) -> Image.Image:
        # Take a PIL img and convert it into a segmentation map
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits

        # Upsample the logits to match the original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, 
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )



        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # map the label to RGB colors
        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self._get_palette())

        for label, color in enumerate(palette):
            color_seg[pred_seg == label] = color

        return Image.fromarray(color_seg.astype(np.uint8))

    def _get_palette(self):
        # define a RGB palette for the segmentation map
        # only 150 colors, one for each class in the ADE20K dataset
        np.random.seed(42)
        return np.random.randint(0, 255, size=(150, 3), dtype=np.uint8).tolist()

if __name__ == "__main__":
    from PIL import Image
    import os

    print("Inizializzazione del segmentatore...")
    segmentor = SemanticSegmentor()

    # 1. Inserisci qui il percorso della tua foto sul PC
    # Esempio su Windows: "C:/Utenti/Nome/Desktop/mia_foto.jpg" oppure "mia_foto.jpg" se è nella stessa cartella
    image_path = "C:/Users/Ladybug/Desktop/paesaggio-1024x683.png"

    if not os.path.exists(image_path):
        print(f"Errore: Impossibile trovare l'immagine al percorso '{image_path}'!")
    else:
        print(f"Caricamento dell'immagine: {image_path}...")
        input_img = Image.open(image_path).convert("RGB")

        print("Elaborazione della mappa semantica...")
        output_seg = segmentor.process(input_img)

        # Salviamo la mappa semantica per verificarla
        output_filename = "test_segmentation_output.png"
        output_seg.save(output_filename)
        print(f"Fatto! La mappa semantica è stata salvata come '{output_filename}'")