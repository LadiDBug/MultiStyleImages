import os
import torch 
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, artifacts, flat colors, deformed, "
    "oversaturated, noisy, watermark, signature, text, bad anatomy"
)

def compute_optimal_dimensions(width: int, height: int, max_dim: int = 512) -> tuple[int, int]:
    """Calculate proportional dimensions aligned to multiples of 64 to preserve aspect ratio without distortion."""
    scale = max_dim / max(width, height)
    new_w = max(64, int((width * scale) // 64) * 64)
    new_h = max(64, int((height * scale) // 64) * 64)
    return new_w, new_h

class StyleTransferPipeline:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 1. Enhanced VAE for sharper details and vibrant colors
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=dtype
        )

        # 2. ControlNet model for semantic segmentation conditioning
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", 
            torch_dtype=dtype
        )

        # 3. Stable Diffusion ControlNet pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=dtype,
            safety_checker=None
        )

        # 4. UniPC scheduler for fast and stable multistep diffusion
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        control_image: Image.Image,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_end: float = 0.85,
        seed: int = 42
    ) -> Image.Image:
        """Generate from text + segmentation in Text-to-Image mode with aspect-ratio preservation."""
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

        orig_size = image.size
        # Compute optimal generation resolution matching input aspect ratio
        gen_w, gen_h = compute_optimal_dimensions(orig_size[0], orig_size[1], max_dim=512)

        # Nearest-neighbor preserves discrete class IDs/colors without interpolation artifacts
        control_resized = control_image.convert("RGB").resize(
            (gen_w, gen_h), resample=Image.Resampling.NEAREST
        )

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_resized,
            height=gen_h,
            width=gen_w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=0.0,
            control_guidance_end=control_guidance_end,
            generator=generator
        ).images[0]

        # Preserve native high quality (512px) if the input image was small;
        # only resize back if the input image was larger than the generated resolution.
        if orig_size[0] > gen_w or orig_size[1] > gen_h:
            return output.resize(orig_size, resample=Image.Resampling.LANCZOS)
        return output

if __name__ == "__main__":
    from segmentation import SemanticSegmentor

    # 1. Percorso della foto di input
    image_path = "test_images/paesaggio-1024x683.png"

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
        prompt = "cyberpunk style"
        
        output_img = pipeline.generate(
            prompt=prompt,
            image=input_img,
            control_image=seg_map,
            num_inference_steps=25,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            control_guidance_end=0.85,
            seed=42
        )

        output_path = "test_pipeline_output.png"
        output_img.save(output_path)
        print(f"Fatto! L'immagine generata è stata salvata come '{output_path}'")
