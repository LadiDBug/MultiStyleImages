import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Legge un singolo file TOML senza modificarne il contenuto."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_experiment_config(project_root: str | Path | None = None) -> dict:
    """Carica la configurazione della baseline, senza caricare modelli.

    Restituisce le sezioni model, generation, experiment, data e styles.
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
    selected_styles = experiment["styles"]
    if not isinstance(selected_styles, list) or not selected_styles:
        raise ValueError("experiment.styles deve essere una lista non vuota.")
    for style in selected_styles:
        if not isinstance(style, str) or style not in styles:
            raise ValueError(f"Stile non definito in styles.toml: {style!r}")
        for field in ("prompt", "negative_prompt"):
            if not isinstance(styles[style].get(field), str):
                raise ValueError(f"styles.{style}.{field} deve essere una stringa.")

    if experiment["conditions"] != ["none"]:
        raise ValueError('La baseline richiede conditions = ["none"].')

    seeds = experiment["seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        raise ValueError("experiment.seeds deve contenere interi non negativi.")

    data = dict(main["data"])
    for field in ("input_dir", "output_dir"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{field} deve essere un percorso non vuoto.")
        data[field] = (root / value).resolve()

    return {
        "model": base["model"],
        "generation": base["generation"],
        "experiment": experiment,
        "data": data,
        "styles": styles,
    }

