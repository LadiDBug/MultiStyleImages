"""Runner iniziale: una fotografia, tutti gli stili e seed configurati."""

import argparse
import gc
import json
import torch
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.utils.config import load_experiment_config
from src.generation.pipeline import load_pipeline
from src.generation.generator import generate
from src.preprocessing.img_preprocess import preprocess_image
from src.preprocessing.canny import extract_canny



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        help="Prompt personalizzato: esegue un solo stile 'custom' senza modificare i TOML.",
    )
    parser.add_argument(
        "--negative-prompt",
        help="Negative prompt per --prompt; se omesso usa una stringa vuota.",
    )
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "-canny", "--canny", dest="condition", action="store_const", const="canny",
        help="Esegue solo ControlNet Canny, sostituendo le condizioni del TOML.",
    )
    condition_group.add_argument(
        "-segmentation", "--segmentation", dest="condition", action="store_const",
        const="segmentation",
        help="Esegue solo ControlNet Segmentation, sostituendo le condizioni del TOML.",
    )
    condition_group.add_argument(
        "-depth", "--depth", dest="condition", action="store_const", const="depth",
        help="Esegue solo ControlNet Depth, sostituendo le condizioni del TOML.",
    )
    condition_group.add_argument(
        "-none", "--none", dest="condition", action="store_const", const="none",
        help="Esegue solo la baseline, sostituendo le condizioni del TOML.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Usa la foto RGB originale, senza crop o resize, per tutte le condizioni.",
    )
    preparation_group = parser.add_mutually_exclusive_group()
    preparation_group.add_argument(
        "--prepare-canny-only",
        action="store_true",
        help="Salva foto preparata e mappa Canny senza caricare Stable Diffusion.",
    )
    preparation_group.add_argument(
        "--prepare-segmentation-only",
        action="store_true",
        help="Salva solo foto e mappe semantiche, senza caricare Stable Diffusion.",
    )
    preparation_group.add_argument(
        "--prepare-depth-only",
        action="store_true",
        help="Salva solo foto e mappa depth, senza caricare Stable Diffusion.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    preparation_condition = (
        "canny" if args.prepare_canny_only
        else "segmentation" if args.prepare_segmentation_only
        else "depth" if args.prepare_depth_only else None
    )
    if preparation_condition is not None and args.condition is not None:
        parser.error("Usare le opzioni --prepare-...-only senza flag di generazione.")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt non puo' essere vuoto.")
    if args.negative_prompt is not None and args.prompt is None:
        parser.error("--negative-prompt richiede --prompt.")
    if preparation_condition is not None and args.prompt is not None:
        parser.error("--prompt si usa per generare, non con --prepare-...-only.")
    config = load_experiment_config(root, condition=preparation_condition or args.condition)
    if args.prompt is not None:
        config["experiment"]["styles"] = ["custom"]
        config["styles"] = {
            "custom": {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt if args.negative_prompt is not None else "",
            }
        }
    input_dir = config["data"]["input_dir"]
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cartella input non trovata: {input_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        raise ValueError(f"Nessuna fotografia trovata in: {input_dir}")

    # Prima prova: solo la prima foto in ordine alfabetico.
    image_path = images[0]
    parameters = config["generation"]
    if args.no_preprocess:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    else:
        image = preprocess_image(
            image_path, width=parameters["width"], height=parameters["height"]
        )
    print(f"Foto selezionata: {image_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = config["data"]["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {
        **config,
        "input_image": str(image_path),
        "preprocessing_enabled": not args.no_preprocess,
        "input_size": image.size,
        "mode": f"prepare_{preparation_condition}_only" if preparation_condition else "generation",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # Conserva la foto esatta passata al modello per il confronto visivo.
    image.save(run_dir / ("input_original.png" if args.no_preprocess else "input_preprocessed.png"))

    conditions = config["experiment"]["conditions"]
    canny_image = None
    map_path = None
    condition_images = {}
    map_paths = {}
    labels_path = None
    depth_values_path = None
    if args.prepare_canny_only or "canny" in conditions:
        canny_parameters = config["preprocessing"]["canny"]
        canny_image = extract_canny(image, **canny_parameters)
        map_dir = run_dir / "maps" / "canny"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / f"{image_path.stem}.png"
        canny_image.save(map_path)
        condition_images["canny"] = canny_image
        map_paths["canny"] = map_path
        print(f"Mappa Canny salvata: {map_path}")

    if "segmentation" in conditions:
        # Import e caricamento del segmentatore solo quando viene richiesto.
        from src.preprocessing.semantic import SemanticSegmentor

        segmentation_config = config["preprocessing"]["segmentation"]
        print(f"Caricamento segmentatore: {segmentation_config['model_id']}")
        segmentor = SemanticSegmentor(
            model_name=segmentation_config["model_id"],
            device=config["model"]["device"],
        )
        try:
            labels = segmentor.process_labels(image)
            segmentation_image = segmentor.colorize_labels(labels)
            segmentation_dir = run_dir / "maps" / "segmentation"
            segmentation_dir.mkdir(parents=True, exist_ok=True)
            segmentation_path = segmentation_dir / f"{image_path.stem}.png"
            labels_path = segmentation_dir / f"{image_path.stem}_labels.png"
            segmentation_image.save(segmentation_path)
            Image.fromarray(labels).save(labels_path)
            condition_images["segmentation"] = segmentation_image
            map_paths["segmentation"] = segmentation_path
            print(f"Mappa semantica salvata: {segmentation_path}")
        finally:
            del segmentor
            gc.collect()
            if torch.device(config["model"]["device"]).type == "cuda":
                torch.cuda.empty_cache()

    if "depth" in conditions:
        import numpy as np
        from src.preprocessing.depth import DepthEstimator

        depth_config = config["preprocessing"]["depth"]
        print(f"Caricamento stimatore depth: {depth_config['model_id']}")
        estimator = DepthEstimator(
            model_name=depth_config["model_id"],
            device=config["model"]["device"],
        )
        try:
            depth_values = estimator.process_depth(image)
            depth_image = estimator.depth_to_image(depth_values)
            depth_dir = run_dir / "maps" / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            depth_path = depth_dir / f"{image_path.stem}.png"
            depth_values_path = depth_dir / f"{image_path.stem}_depth.npy"
            depth_image.save(depth_path)
            np.save(depth_values_path, depth_values, allow_pickle=False)
            condition_images["depth"] = depth_image
            map_paths["depth"] = depth_path
            print(f"Mappa depth salvata: {depth_path}")
        finally:
            del estimator
            gc.collect()
            if torch.device(config["model"]["device"]).type == "cuda":
                torch.cuda.empty_cache()

    if preparation_condition is not None:
        return

    model = config["model"]
    with (run_dir / "metadata.jsonl").open("a", encoding="utf-8") as metadata:
        for condition in conditions:
            checkpoint = config["controlnets"][condition] if condition != "none" else None
            control_kwargs = {}
            if condition != "none":
                control_kwargs = {
                    "condition_image": condition_images[condition],
                    "controlnet_conditioning_scale": config["controlnet"]["conditioning_scale"],
                }
            print(f"Caricamento pipeline: {condition}")
            pipe = load_pipeline(
                model_id=model["id"], device=model["device"], dtype=model["dtype"],
                condition=condition, controlnet_id=checkpoint,
            )
            try:
                for style in config["experiment"]["styles"]:
                    style_config = config["styles"][style]
                    output_dir = run_dir / "images" / style / condition
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for seed in config["experiment"]["seeds"]:
                        output_path = output_dir / f"{image_path.stem}_seed{seed}.png"
                        record = {
                            "image_id": image_path.name,
                            "input_path": str(image_path),
                            "style": style,
                            "condition": condition,
                            "seed": seed,
                            **style_config,
                            "model": model,
                            "scheduler": type(pipe.scheduler).__name__,
                            "generation": parameters,
                            "preprocessing_enabled": not args.no_preprocess,
                            "input_size": image.size,
                            "controlnet_checkpoint": checkpoint,
                            "controlnet_conditioning_scale": control_kwargs.get("controlnet_conditioning_scale"),
                            "condition_image_path": str(map_paths[condition]) if condition != "none" else None,
                            "canny_parameters": config["preprocessing"]["canny"] if condition == "canny" else None,
                            "segmentation_parameters": config["preprocessing"]["segmentation"] if condition == "segmentation" else None,
                            "segmentation_labels_path": str(labels_path) if condition == "segmentation" else None,
                            "depth_parameters": config["preprocessing"]["depth"] if condition == "depth" else None,
                            "depth_values_path": str(depth_values_path) if condition == "depth" else None,
                            "depth_normalization": "per_image_minmax_0_255_near_white" if condition == "depth" else None,
                            "output_path": str(output_path),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        started = perf_counter()
                        try:
                            print(
                                f"\nGenerazione: stile={style}, condizione={condition}, seed={seed}\n"
                                f"Prompt: {style_config['prompt']}\n"
                                f"Negative prompt: {style_config['negative_prompt']}\n"
                                f"Strength: {parameters['strength']}, "
                                f"guidance: {parameters['guidance_scale']}, "
                                f"ControlNet scale: {control_kwargs.get('controlnet_conditioning_scale')}\n"
                                f"Output: {output_path}",
                                flush=True,
                            )
                            output = generate(
                                pipe=pipe,
                                image=image,
                                prompt=style_config["prompt"],
                                negative_prompt=style_config["negative_prompt"],
                                seed=seed,
                                use_original_image=args.no_preprocess,
                                **parameters,
                                **control_kwargs,
                            )
                            output.save(output_path)
                        except Exception as error:
                            record.update(status="error", error=f"{type(error).__name__}: {error}")
                            raise
                        else:
                            record["status"] = "success"
                            print(f"Salvata: {output_path}")
                        finally:
                            record["duration_seconds"] = perf_counter() - started
                            metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
                            metadata.flush()
            finally:
                # Libera la pipeline prima di caricare la condizione successiva.
                del pipe
                gc.collect()
                if torch.device(model["device"]).type == "cuda":
                    torch.cuda.empty_cache()

    print(f"Run completato: {run_dir}")


if __name__ == "__main__":
    main()
