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
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "-canny", "--canny", dest="condition", action="store_const", const="canny",
        help="Esegue solo ControlNet Canny, sostituendo le condizioni del TOML.",
    )
    condition_group.add_argument(
        "-none", "--none", dest="condition", action="store_const", const="none",
        help="Esegue solo la baseline, sostituendo le condizioni del TOML.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Usa la foto RGB originale, senza crop o resize, solo per la baseline.",
    )
    parser.add_argument(
        "--prepare-canny-only",
        action="store_true",
        help="Salva foto preparata e mappa Canny senza caricare Stable Diffusion.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.prepare_canny_only and args.condition is not None:
        parser.error("Usare --prepare-canny-only senza -canny o -none.")
    config = load_experiment_config(root, condition=args.condition)
    if args.no_preprocess and (
        args.prepare_canny_only or config["experiment"]["conditions"] != ["none"]
    ):
        parser.error("Usare --no-preprocess insieme a --none, senza --prepare-canny-only.")
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
        "mode": "prepare_canny_only" if args.prepare_canny_only else "generation",
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
    if args.prepare_canny_only or "canny" in conditions:
        canny_parameters = config["preprocessing"]["canny"]
        canny_image = extract_canny(image, **canny_parameters)
        map_dir = run_dir / "maps" / "canny"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / f"{image_path.stem}.png"
        canny_image.save(map_path)
        print(f"Mappa Canny salvata: {map_path}")
    if args.prepare_canny_only:
        return

    model = config["model"]
    with (run_dir / "metadata.jsonl").open("a", encoding="utf-8") as metadata:
        for condition in conditions:
            checkpoint = config["controlnets"][condition] if condition != "none" else None
            control_kwargs = {}
            if condition == "canny":
                control_kwargs = {
                    "condition_image": canny_image,
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
                            "condition_image_path": str(map_path) if condition == "canny" else None,
                            "canny_parameters": config["preprocessing"]["canny"] if condition == "canny" else None,
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
