import torch 
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ControlNet model for conditional image generation
# StableDiffusionControlNetPipeline is a pipeline that allos you to use ControlNet with Stable Diffusion for image generation tasks
# UniPCMultistepScheduler is a scheduler that helps to control the diffusion process during image generation

class StyleTransferPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        #download controlnet model for semantic segmentation
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )


        # Initialize the Stable Diffusion ControlNet pipeline with the pre-trained model and the ControlNet model
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )

        # Set the scheduler for the diffusion process to UniPCMultistepScheduler, which is a specific scheduler that helps control the diffusion process during image generation
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Move the pipeline to the specified device (GPU or CPU) in order to avoid memory issues
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(
            self,
            prompt: str,
            image: Image.Image,
            control_image: Image.Image,
            num_inference_steps: int = 20,
            guidance_scale: float = 7.5,
            controlnet_conditioning_scale: float = 1.0,
            seed: int = 42
            
    ) -> Image.Image:
        # we generate an image based on the prompt and the segmentation map
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        orig_size = image.size
        img_resized = image.resize((512, 512))
        control_resized = control_image.resize((512, 512))

        output = self.pipe(
            prompt=prompt,
            image=control_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        ).images[0]

        # Resize the output image back to the original size
        output_resized = output.resize(orig_size)

        return output_resized
if __name__ == "__main__":
    import os
    from PIL import Image
    from segmentation import SemanticSegmentor

    # 1. Percorso della foto di input sul tuo PC
    image_path = "C:\\Users\\Ladybug\\Desktop\\paesaggio-1024x683.png" # <--- Sostituisci con il nome della tua foto

    if not os.path.exists(image_path):
        print(f"Errore: File '{image_path}' non trovato!")
    else:
        print("1/3 Caricamento foto e calcolo mappa semantica...")
        input_img = Image.open(image_path).convert("RGB")
        
        # Segmentiamo l'immagine
        segmentor = SemanticSegmentor()
        seg_map = segmentor.process(input_img)
        
        print("2/3 Inizializzazione della Pipeline di Generazione...")
        pipeline = StyleTransferPipeline()

        print("3/3 Generazione dell'immagine in stile test...")
        # Proviamo a generare uno stile urbano generico prima di caricare le LoRA
        prompt = "a futuristic cyberpunk street scene with neon lights"
        
        output_img = pipeline.generate(
            prompt=prompt,
            image=input_img,
            controlnet_image=seg_map,
            num_inference_steps=20,
            controlnet_conditioning_scale=1.0,
            seed=42
        )

        output_path = "test_pipeline_output.png"
        output_img.save(output_path)
        print(f"Fatto! L'immagine generata è stata salvata come '{output_path}'")