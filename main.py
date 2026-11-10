"""Runner img2img RQ1/RQ2: fotografie nelle sottocartelle, LoRA opzionale."""

import argparse
import gc
import hashlib
import json
import math
import torch
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.utils.config import load_experiment_config
from src.utils.lora_config import load_lora_config, training_arguments

ROOT = Path(__file__).resolve().parent


def _configure_lora(config, path, scale):
    """Aggiunge i riferimenti LoRA soltanto se richiesti esplicitamente."""
    if path is None:
        if scale is not None:
            raise ValueError("--lora-scale richiede --lora-path.")
        return
    scale = 1.0 if scale is None else scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("--lora-scale deve essere finita e non negativa.")
    if len(config["experiment"]["styles"]) != 1:
        raise ValueError(
            "Con --lora-path selezionare un solo stile in configs/main.toml "
            "(per questa LoRA: styles = ['pixel_art'])."
        )
    path = Path(path).expanduser().resolve()
    weights = path/"pytorch_lora_weights.safetensors" if path.is_dir() else path
    if not weights.is_file() or weights.suffix != ".safetensors":
        raise ValueError(f"Pesi LoRA .safetensors non trovati: {weights}")
    # Per gli export del nostro trainer, recupera rank e revisione immutabile.
    training_path = weights.parent/"run_config.json"
    if not training_path.is_file() and weights.parent.name.startswith("checkpoint-"):
        training_path = weights.parent.parent/"run_config.json"
    training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.is_file() else {}
    trained = training.get("config", {})
    if trained.get("model_id") and trained["model_id"] != config["model"]["id"]:
        raise ValueError("La LoRA e' stata addestrata con un backbone diverso da model.id.")
    revision = training.get("resolved_revision")
    if revision:
        config["model"] = {**config["model"], "revision": revision}
    config["lora"] = {
        "path": str(weights),
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "scale": float(scale),
        "rank": trained.get("rank"),
        "alpha": trained.get("alpha"),
        "base_model_revision": revision,
        "training_signature": training.get("signature"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "inference"],
                        help="train con --lora CONFIG; inference oppure nessun comando per generare.")
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--lora", type=Path,
                           help="Config LoRA in configs/, per esempio lora_rank4.toml.")
    lora_group.add_argument(
        "--lora-path", type=Path,
        help="Directory LoRA o file .safetensors locale; se omesso esegue come prima senza adattatore.",
    )
    parser.add_argument(
        "--lora-scale", type=float,
        help="Intensita' dell'adattatore (default 1.0); richiede --lora oppure --lora-path.",
    )
    parser.add_argument("--print-config", action="store_true",
                        help="Stampa la configurazione effettiva senza eseguire il run.")
    parser.add_argument("--check-data", action="store_true", help="Solo train: verifica il dataset senza addestrare.")
    parser.add_argument("--resume-from-checkpoint", help="Solo train: latest oppure percorso di un checkpoint.")
    parser.add_argument("--dataset-dir", type=Path,
                        help="Solo train: sostituisce il percorso del dataset per il run, senza modificare il TOML.")
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
    args = parser.parse_args(argv)
    root = ROOT
    preparation_condition = (
        "canny" if args.prepare_canny_only
        else "segmentation" if args.prepare_segmentation_only
        else "depth" if args.prepare_depth_only else None
    )
    if args.lora is not None and args.command is None:
        parser.error("Con --lora specificare train oppure inference.")
    if args.command == "train":
        if args.lora is None:
            parser.error("train richiede --lora CONFIG.toml.")
        if (preparation_condition or args.condition or args.prompt is not None
                or args.negative_prompt is not None or args.no_preprocess or args.lora_scale is not None):
            parser.error("I flag di generazione e preprocessing non si usano con train.")
        try:
            _, lora_config = load_lora_config(root, args.lora)
            if args.dataset_dir is not None:
                lora_config["data"]["input_dir"] = str(args.dataset_dir.expanduser().resolve())
            train_args = training_arguments(root, lora_config)
        except (ValueError, TypeError, OSError, KeyError) as error:
            parser.error(str(error))
        for flag in ("print_config", "check_data"):
            if getattr(args, flag):
                train_args.append("--" + flag.replace("_", "-"))
        if args.resume_from_checkpoint:
            train_args.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
        from src.training_lora.train import main as train_main

        train_main(train_args)
        return
    if args.check_data or args.resume_from_checkpoint or args.dataset_dir is not None:
        parser.error("--check-data, --dataset-dir e --resume-from-checkpoint si usano soltanto con train.")
    if preparation_condition is not None and args.condition is not None:
        parser.error("Usare le opzioni --prepare-...-only senza flag di generazione.")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt non puo' essere vuoto.")
    if args.negative_prompt is not None and args.prompt is None:
        parser.error("--negative-prompt richiede --prompt.")
    if preparation_condition is not None and args.prompt is not None:
        parser.error("--prompt si usa per generare, non con --prepare-...-only.")
    if preparation_condition is not None and (args.lora is not None or args.lora_path is not None or args.lora_scale is not None):
        parser.error("Le opzioni LoRA si usano per generare, non con --prepare-...-only.")
    lora_config = None
    try:
        if args.lora is not None:
            lora_config_path, lora_config = load_lora_config(root, args.lora)
        config = load_experiment_config(
            root, condition=preparation_condition or args.condition,
            experiment_overrides=lora_config["experiment"] if lora_config else None,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    if args.prompt is not None:
        config["experiment"]["styles"] = ["custom"]
        config["styles"] = {
            "custom": {
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt if args.negative_prompt is not None else "",
            }
        }
    try:
        lora_path = root / lora_config["data"]["output_dir"] if lora_config else args.lora_path
        _configure_lora(config, lora_path, args.lora_scale)
        if lora_config:
            for parameter in ("rank", "alpha"):
                actual = config["lora"][parameter]
                if actual is not None and actual != lora_config["parameters"][parameter]:
                    raise ValueError(f"I pesi LoRA non corrispondono a parameters.{parameter} del TOML selezionato.")
            config["lora"]["config_path"] = str(lora_config_path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False, default=str))
        return
    input_dir = config["data"]["input_dir"]
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cartella input non trovata: {input_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        raise ValueError(f"Nessuna fotografia trovata in: {input_dir}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    batch_dir = config["data"]["output_dir"] / run_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        **config,
        "input_images": [str(image.relative_to(input_dir)) for image in images],
        "preprocessing_enabled": not args.no_preprocess,
        "mode": f"prepare_{preparation_condition}_only" if preparation_condition else "generation",
    }
    (batch_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Trovate {len(images)} fotografie in {input_dir}")
    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(input_dir)
        # Mantiene sottocartelle ed estensione: foto.jpg e foto.png non collidono.
        image_run_dir = batch_dir / "samples" / relative_path
        image_run_dir.mkdir(parents=True, exist_ok=False)
        print(f"\nFoto {index}/{len(images)}: {relative_path}")
        _run_image(config, args, preparation_condition, image_path, image_run_dir, relative_path)
    print(f"Test completo terminato: {batch_dir}")


def _run_image(config, args, preparation_condition, image_path, run_dir, relative_path):
    """Esegue le condizioni della singola foto nella sua cartella dedicata."""
    from src.preprocessing.img_preprocess import preprocess_image

    parameters = config["generation"]
    if args.no_preprocess:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    else:
        image = preprocess_image(
            image_path, width=parameters["width"], height=parameters["height"]
        )
    print(f"Foto selezionata: {image_path}")

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
        from src.preprocessing.canny import extract_canny

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

    from src.generation.pipeline import load_pipeline
    from src.generation.generator import generate

    model = config["model"]
    lora_kwargs = {}
    if "lora" in config:
        lora_kwargs = {"lora_path": config["lora"]["path"], "lora_scale": config["lora"]["scale"]}
        if config["lora"]["base_model_revision"]:
            lora_kwargs["revision"] = config["lora"]["base_model_revision"]
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
                **lora_kwargs,
            )
            try:
                for style in config["experiment"]["styles"]:
                    style_config = config["styles"][style]
                    output_dir = run_dir / "images" / style / condition
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for seed in config["experiment"]["seeds"]:
                        output_path = output_dir / f"{image_path.stem}_seed{seed}.png"
                        record = {
                            "image_id": relative_path.as_posix(),
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
                            **({"lora": config["lora"]} if "lora" in config else {}),
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
