"""Runner iniziale: una fotografia, tutti gli stili e seed configurati."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.utils.config import load_experiment_config
from src.generation.pipeline import load_pipeline
from src.generation.generator import generate
from src.preprocessing.img_preprocess import preprocess_image



def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_experiment_config(root)
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
    image = preprocess_image(
        image_path, width=parameters["width"], height=parameters["height"]
    )
    print(f"Foto selezionata: {image_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = config["data"]["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {**config, "input_image": str(image_path)}
    (run_dir / "run_config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    # Conserva la foto esatta passata al modello per il confronto visivo.
    image.save(run_dir / "input_preprocessed.png")

    model = config["model"]
    pipe = load_pipeline(
        model_id=model["id"], device=model["device"], dtype=model["dtype"]
    )

    with (run_dir / "metadata.jsonl").open("a", encoding="utf-8") as metadata:
        for style in config["experiment"]["styles"]:
            style_config = config["styles"][style]
            output_dir = run_dir / "images" / style / "none"
            output_dir.mkdir(parents=True, exist_ok=True)
            for seed in config["experiment"]["seeds"]:
                output_path = output_dir / f"{image_path.stem}_seed{seed}.png"
                record = {
                    "image_id": image_path.name,
                    "input_path": str(image_path),
                    "style": style,
                    "condition": "none",
                    "seed": seed,
                    **style_config,
                    "model": model,
                    "scheduler": type(pipe.scheduler).__name__,
                    "generation": parameters,
                    "output_path": str(output_path),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                started = perf_counter()
                try:
                    output = generate(
                        pipe=pipe,
                        image=image,
                        prompt=style_config["prompt"],
                        negative_prompt=style_config["negative_prompt"],
                        seed=seed,
                        **parameters,
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

    print(f"Run completato: {run_dir}")


if __name__ == "__main__":
    main()
