# file con il main a cui passare un config e si avvia la modalità richiesta

# deve fare le seguenti operazioni
#1. leggere le config toml
#2. selezionare lo stile
#3. selezionare il condizionamento
# 4. avviare la generazione e salvare img

# implemento la condition = none, ovvero la baseline

from pathlib import Path

from src.utils.config import load_config
from src.generation.pipeline import load_pipeline
from src.generation.generator import generate



def main():
    root = Path(__file__).resolve().parent

    base = load_config(root / "configs/base.toml")
    styles = load_config(root / "configs/styles.toml")

    style = "cyberpunk"
    style_config = styles["styles"][style]

    pipe = load_pipeline(
        model_id=base["model"]["id"]
    )

    # Qui caricheremo la fotografia
    # e chiameremo generate(...)


if __name__ == "__main__":
    main()