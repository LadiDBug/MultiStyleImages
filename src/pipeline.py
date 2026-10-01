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
        else:
            self.pipe.to(self.device)

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
        """Generate from text + segmentation; image supplies the output size.

        This is ControlNet text-to-image, not img2img: the source photograph
        is represented by control_image, not used as initial diffusion latents.
        """
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

        orig_size = image.size
        # Nearest-neighbor preserves class colors instead of blending labels.
        control_resized = control_image.convert("RGB").resize(
            (512, 512), resample=Image.Resampling.NEAREST
        )

        output = self.pipe(
            prompt=prompt,
            image=control_resized,
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        ).images[0]

        # Resize the output image back to the original size
        output_resized = output.resize(orig_size, resample=Image.Resampling.LANCZOS)

        return output_resized
if __name__ == "__main__":
    import os
    from PIL import Image
    from segmentation import SemanticSegmentor

    # 1. Percorso della foto di input sul tuo PC
    image_path = "test_images/paesaggio-1024x683.png" # <--- Sostituisci con il nome della tua foto

    if not os.path.exists(image_path):
        print(f"Errore: File '{image_path}' non trovato!")
    else:
        print("1/3 Caricamento foto e calcolo mappa semantica...")
        input_img = Image.open(image_path).convert("RGB")
        
        # Segmentiamo l'immagine
        segmentor = SemanticSegmentor()
        seg_map = segmentor.process(input_img)
        seg_map.save("test_segmentation_output.png")
        
        print("2/3 Inizializzazione della Pipeline di Generazione...")
        pipeline = StyleTransferPipeline()

        print("3/3 Generazione dell'immagine in stile test...")
        # Style-only baseline: do not describe or change the scene content.
        prompt = "cyberpunk style"
        
        output_img = pipeline.generate(
            prompt=prompt,
            image=input_img,
            control_image=seg_map,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            seed=42
        )

        output_path = "test_pipeline_output.png"
        output_img.save(output_path)
        print(f"Fatto! L'immagine generata è stata salvata come '{output_path}'")
