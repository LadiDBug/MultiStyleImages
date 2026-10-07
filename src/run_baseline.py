import os
import time
import json
import argparse
from pathlib import Path
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"--- {desc} ---")
        return iterable

from segmentation import SemanticSegmentor
try:
    from instance_segmentation import InstanceSegmentor
except ImportError:
    from src.instance_segmentation import InstanceSegmentor

from pipeline import StyleTransferPipeline

# Default styles for the baseline
DEFAULT_STYLES = {
    "cyberpunk": "cyberpunk style",
    "gta_5": "GTA V style",
    "pixel_art": "pixel art style"
}

def run_baseline_benchmark(
    input_dir: str,
    output_dir: str,
    styles: dict[str, str] = DEFAULT_STYLES,
    seg_type: str = "semantic",
    num_samples: int = None,
    seed: int = 42,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    control_guidance_end: float = 0.85,
    apply_preprocessing: bool = False,
    save_segmentation_maps: bool = True,
    device: str = None
):
    """Run baseline Text-to-Image style transfer on all images in input_dir with selectable segmentation type."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Gather valid images (supports both single file and directory)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if input_path.is_file():
        if input_path.suffix.lower() in valid_exts:
            image_files = [input_path]
        else:
            raise ValueError(f"File extension not supported: {input_path}")
    elif input_path.is_dir():
        image_files = sorted([
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_exts
        ])
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    if not image_files:
        raise ValueError(f"No valid images found in {input_path}")

    if num_samples is not None and num_samples > 0:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} images to process.")
    print(f"Segmentation type: {seg_type.upper()}")
    print(f"Target styles: {list(styles.keys())}")
    print(f"Output directory: {output_path}")

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if save_segmentation_maps:
        (output_path / "segmentation_maps").mkdir(parents=True, exist_ok=True)

    for style_name in styles.keys():
        (output_path / style_name).mkdir(parents=True, exist_ok=True)

    # 1. Initialize models
    segmentor_kwargs = {"device": device} if device else {}
    if seg_type.lower() == "instance":
        print("\n[1/3] Initializing InstanceSegmentor (Tree/Object Instance Separation)...")
        segmentor = InstanceSegmentor(**segmentor_kwargs)
    else:
        print("\n[1/3] Initializing SemanticSegmentor (UPerNet ADE20K)...")
        segmentor = SemanticSegmentor(**segmentor_kwargs)

    print("[2/3] Initializing StyleTransferPipeline (SD 1.5 + ControlNet Seg)...")
    pipeline_kwargs = {"device": device} if device else {}
    pipeline = StyleTransferPipeline(**pipeline_kwargs)

    if apply_preprocessing:
        try:
            from img_preprocessing import preprocess_image
        except ImportError:
            from src.img_preprocessing import preprocess_image
        print("Preprocessing module enabled for input images.")

    print("[3/3] Running benchmark across all images and styles...\n")
    metadata_log = {
        "benchmark_type": "baseline_style_only",
        "segmentation_type": seg_type,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "control_guidance_end": control_guidance_end,
        "styles": styles,
        "total_images": len(image_files),
        "processed_items": []
    }

    start_total_time = time.time()

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_name = img_file.stem
        input_img = Image.open(img_file).convert("RGB")

        if apply_preprocessing:
            input_img = preprocess_image(input_img, return_pil=True)
        
        t0 = time.time()
        # 1. Compute segmentation map once per image
        if seg_type.lower() == "instance":
            seg_map = segmentor.process(input_img, preprocess=apply_preprocessing)
        else:
            seg_map = segmentor.process(input_img)
        t_seg = time.time() - t0

        if save_segmentation_maps:
            seg_save_path = output_path / "segmentation_maps" / f"{img_name}_{seg_type}_seg.png"
            seg_map.save(seg_save_path)

        item_log = {
            "image_name": img_name,
            "original_path": str(img_file),
            "original_size": input_img.size,
            "segmentation_type": seg_type,
            "segmentation_time_sec": round(t_seg, 3),
            "generations": {}
        }

        # 2. Generate for each style
        for style_key, prompt_text in styles.items():
            t_gen_start = time.time()
            
            output_img = pipeline.generate(
                prompt=prompt_text,
                image=input_img,
                control_image=seg_map,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_end=control_guidance_end,
                seed=seed
            )
            
            t_gen = time.time() - t_gen_start
            
            save_dest = output_path / style_key / f"{img_name}_{style_key}.png"
            output_img.save(save_dest)

            item_log["generations"][style_key] = {
                "prompt": prompt_text,
                "output_file": str(save_dest.name),
                "generation_time_sec": round(t_gen, 3)
            }

        metadata_log["processed_items"].append(item_log)

    total_duration = time.time() - start_total_time
    metadata_log["total_elapsed_seconds"] = round(total_duration, 2)

    # Save complete run metadata
    metadata_file = output_path / "experiment_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_log, f, indent=2)

    print(f"\n Benchmark complete in {total_duration:.2f}s!")
    print(f"Results and metadata saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Style Transfer on Landscape Dataset")
    parser.add_argument("--input_dir", type=str, default="test_images", help="Path to input images directory or single image file")
    parser.add_argument("--output_dir", type=str, default="results/baseline", help="Directory to save generated images")
    parser.add_argument("--seg_type", type=str, choices=["semantic", "instance"], default="semantic",
                        help="Segmentation type to use: 'semantic' (ADE20K classes) or 'instance' (individual objects/trees)")
    parser.add_argument("--instance_seg", action="store_true", help="Shorthand to use instance segmentation instead of semantic")
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing to enhance input images before segmentation")
    parser.add_argument("--num_samples", type=int, default=None, help="Max number of images to process (e.g. 10 or 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for diffusion reproducibility")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG Guidance scale")
    parser.add_argument("--control_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--control_end", type=float, default=0.85, help="ControlNet guidance end ratio")

    args = parser.parse_args()

    chosen_seg_type = "instance" if args.instance_seg else args.seg_type

    run_baseline_benchmark(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seg_type=chosen_seg_type,
        styles=DEFAULT_STYLES,
        num_samples=args.num_samples,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_scale,
        control_guidance_end=args.control_end,
        apply_preprocessing=args.preprocess
    )
