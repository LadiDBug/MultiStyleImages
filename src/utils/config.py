import math
import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Legge un singolo file TOML senza modificarne il contenuto."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_experiment_config(
    project_root: str | Path | None = None,
    *,
    condition: str | None = None,
) -> dict:
    """Carica la configurazione img2img, senza caricare modelli.

    Restituisce model, generation, preprocessing, experiment, data e styles.
    Se sono richiesti canny, depth o segmentation, include controlnet e controlnets.
    condition, se specificata, sostituisce le condizioni del TOML per il run.
    I percorsi in data sono oggetti Path assoluti; non vengono create cartelle.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    config_dir = root / "configs"
    base = load_config(config_dir / "base.toml")
    main = load_config(config_dir / "main.toml")
    styles = load_config(config_dir / "styles.toml")["styles"]

    experiment = main["experiment"]
    if condition is not None:
        experiment["conditions"] = [condition]
    selected_styles = experiment["styles"]
    if not isinstance(selected_styles, list) or not selected_styles:
        raise ValueError("experiment.styles deve essere una lista non vuota.")
    for style in selected_styles:
        if not isinstance(style, str) or style not in styles:
            raise ValueError(f"Stile non definito in styles.toml: {style!r}")
        for field in ("prompt", "negative_prompt"):
            if not isinstance(styles[style].get(field), str):
                raise ValueError(f"styles.{style}.{field} deve essere una stringa.")

    conditions = experiment["conditions"]
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(
            not isinstance(condition, str)
            or condition not in {"none", "canny", "depth", "segmentation"}
            for condition in conditions
        )
        or len(conditions) != len(set(conditions))
    ):
        raise ValueError(
            "experiment.conditions deve essere una lista non vuota, "
            "senza duplicati, contenente solo 'none', 'canny', 'depth' o 'segmentation'."
        )

    control_config = {}
    structural_conditions = [value for value in conditions if value != "none"]
    if structural_conditions:
        controlnet = base.get("controlnet", {})
        scale = controlnet.get("conditioning_scale")
        if (
            type(scale) not in (int, float)
            or not math.isfinite(scale)
            or scale < 0
        ):
            raise ValueError(
                "controlnet.conditioning_scale deve essere un numero finito non negativo."
            )
        checkpoints = load_config(config_dir / "models.toml").get("controlnets", {})
        selected_checkpoints = {}
        for structural_condition in structural_conditions:
            checkpoint = checkpoints.get(structural_condition)
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(
                    f"models.toml deve definire controlnets.{structural_condition} non vuoto."
                )
            selected_checkpoints[structural_condition] = checkpoint
        control_config = {
            "controlnet": controlnet,
            "controlnets": selected_checkpoints,
        }

    seeds = experiment["seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        raise ValueError("experiment.seeds deve contenere interi non negativi.")

    # Il preprocessing resta separato dai parametri passati a generate().
    preprocessing = base["preprocessing"]
    for learned_condition in ("segmentation", "depth"):
        if learned_condition in conditions:
            preprocessor_config = preprocessing.get(learned_condition, {})
            model_id = preprocessor_config.get("model_id")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError(
                    f"preprocessing.{learned_condition}.model_id "
                    "deve essere un checkpoint non vuoto."
                )
    canny = preprocessing["canny"]
    low = canny.get("low_threshold")
    high = canny.get("high_threshold")
    if (
        type(low) is not int
        or type(high) is not int
        or not 0 <= low < high <= 255
    ):
        raise ValueError(
            "preprocessing.canny: le soglie devono essere interi "
            "con 0 <= low_threshold < high_threshold <= 255."
        )

    data = dict(main["data"])
    for field in ("input_dir", "output_dir"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{field} deve essere un percorso non vuoto.")
        data[field] = (root / value).resolve()

    return {
        "model": base["model"],
        "generation": base["generation"],
        "preprocessing": preprocessing,
        "experiment": experiment,
        "data": data,
        "styles": styles,
        **control_config,
    }

