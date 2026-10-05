import gc
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

if __package__:
    from .prompt_builder import PromptBuilder, STYLE_PROMPTS, normalize_style
    from .segmentation import SemanticSegmentor, ADE20K_PALETTE
else:
    from prompt_builder import PromptBuilder, STYLE_PROMPTS, normalize_style
    from segmentation import SemanticSegmentor, ADE20K_PALETTE

SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

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
            SD_MODEL_ID,
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
def main(
    image_path="test_images/paesaggio-1024x683.png",
    style="Cyberpunk",
    automatic_prompt=True,
    min_area_ratio=0.01,
    max_classes=8,
):
    """Run the example; disable automatic_prompt for the style-only baseline."""
    import os

    style_key = normalize_style(style)
    if not os.path.exists(image_path):
        print(f"Errore: File '{image_path}' non trovato!")
    else:
        print("1/3 Caricamento foto e calcolo mappa semantica...")
        input_img = Image.open(image_path).convert("RGB")
        
        # One inference supplies both semantic text and the RGB control map.
        segmentor = SemanticSegmentor()
        labels = segmentor.process_labels(input_img)
        seg_map = Image.fromarray(ADE20K_PALETTE[labels])
        seg_map.save("test_segmentation_output.png")

        builder = PromptBuilder(
            segmentor.model.config.id2label,
            min_area_ratio=min_area_ratio,
            max_classes=max_classes,
        )
        prompt = (
            builder.build(labels, style=style_key)
            if automatic_prompt else STYLE_PROMPTS[style_key]
        )
        print("Classi rilevate (prima del filtraggio del prompt):")
        for feature in builder.extract_features(labels):
            print(f"  {feature.name}: {feature.area_ratio:.1%}")

        # Free segmentation weights before loading Stable Diffusion.
        del segmentor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        output_stem = "test_pipeline_auto" if automatic_prompt else "test_pipeline_output"
        print(f"Prompt: {prompt}")

        # Separate filenames keep automatic and baseline results comparable.
        with open(f"{output_stem}_prompt.txt", "w", encoding="utf-8") as prompt_file:
            prompt_file.write(prompt + "\n")
        
        print("2/3 Inizializzazione della Pipeline di Generazione...")
        pipeline = StyleTransferPipeline()

        print("3/3 Generazione dell'immagine in stile test...")
        output_img = pipeline.generate(
            prompt=prompt,
            image=input_img,
            control_image=seg_map,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            seed=42
        )

        output_path = f"{output_stem}.png"
        output_img.save(output_path)
        print(f"Fatto! L'immagine generata è stata salvata come '{output_path}'")


if __name__ == "__main__":
    main(
        image_path="test_images/paesaggio-1024x683.png",
        style="Cyberpunk",
        automatic_prompt=True,  # False: baseline with only the style prompt.
    )
