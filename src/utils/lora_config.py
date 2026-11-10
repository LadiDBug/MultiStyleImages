"""Adatta i config LoRA del progetto alle CLI di training e inferenza."""

from pathlib import Path

from src.utils.config import load_config


def load_lora_config(root: Path, requested: Path) -> tuple[Path, dict]:
    path = requested.expanduser()
    if not path.is_absolute():
        path = root / "configs" / path if path.parent == Path(".") else root / path
    path = path.resolve()
    content = load_config(path)
    for section in ("experiment", "data", "parameters"):
        if not isinstance(content.get(section), dict):
            raise ValueError(f"{path}: manca la sezione [{section}].")
    for key in ("input_dir", "output_dir"):
        value = content["data"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: data.{key} deve essere un percorso non vuoto.")
    parameters = content["parameters"]
    rank, alpha = parameters.get("rank"), parameters.get("alpha")
    if type(rank) is not int or rank <= 0 or type(alpha) is not int or alpha != rank:
        raise ValueError(f"{path}: rank deve essere positivo e alpha deve essere uguale a rank.")
    return path, content


def training_arguments(root: Path, content: dict) -> list[str]:
    """Traduce lo schema [parameters]/[data], mantenendo il trainer esistente."""
    from src.training_lora.train import TrainingConfig

    base = load_config(root / "configs" / "base.toml")
    values = {
        "dataset_dir": str((root / content["data"]["input_dir"]).resolve()),
        "output_dir": str((root / content["data"]["output_dir"]).resolve()),
        "model_id": base["model"]["id"],
    }
    if "revision" in base["model"]:
        values["revision"] = base["model"]["revision"]
    seeds = content["experiment"].get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("experiment.seeds deve contenere interi non negativi.")
    values["seed"] = seeds[0]
    for key, value in content["parameters"].items():
        if key not in TrainingConfig.__dataclass_fields__ or key in {
            "dataset_dir", "train_dir", "validation_dir", "output_dir",
        }:
            raise ValueError(f"Parametro di training non supportato: parameters.{key}")
        values[key] = value
    arguments = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if key == "gradient_checkpointing":
            if type(value) is not bool:
                raise ValueError("parameters.gradient_checkpointing deve essere booleano.")
            arguments.append(flag if value else "--no-gradient-checkpointing")
        elif key == "target_modules":
            if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
                raise ValueError("parameters.target_modules deve essere una lista di stringhe.")
            arguments.extend([flag, *value])
        else:
            arguments.extend([flag, str(value)])
    return arguments
