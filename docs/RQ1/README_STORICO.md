> **Archivio del README precedente, non guida operativa corrente.** Per struttura e comandi vedere [README principale](../../README.md); per LoRA Pixel Art, dataset e risultati RQ2 vedere [RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md). Le indicazioni seguenti su LoRA future o assenti descrivono una versione superata.

# MultiStyleImages

Per lo stato aggiornato della RQ1 vedere [STATO_PROGETTO_RQ1.md](STATO_PROGETTO_RQ1.md).
L'evaluation separata (audit, CLIP-I, CLIP-T, consistenza semantica) e i comandi
attuali sono descritti in [EVALUATION_RQ1.md](EVALUATION_RQ1.md).
Le istruzioni M1 sottostanti descrivono una versione precedente del runner;
per la CLI di generazione corrente usare `python main.py --help`.

## Baseline RQ1: Stable Diffusion 1.5 img2img (M1)

Il nuovo entry point usa `condition = "none"`: foto RGB + prompt, senza
ControlNet, segmentazione, LoRA o VAE aggiuntivi. Il caricamento segue
la [API Diffusers img2img](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img).
I file in `configs/` sono TOML; `base.toml` contiene modello e parametri,
`styles.toml` i prompt. `models.toml` è riservato alla futura fase ControlNet.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --dry-run
.\.venv\Scripts\python.exe main.py
```

Il default elabora una foto in `src/test_images`, stile cyberpunk, seed 42,
30 step, guidance 7.5 e strength 0.65. Il primo avvio scarica SD 1.5 nella
cache locale `.cache/huggingface`; serve accesso a Internet. `device = "auto"`
sceglie CUDA se disponibile, altrimenti CPU (molto più lenta).

```powershell
.\.venv\Scripts\python.exe main.py --input-dir data/raw --style all --limit 0 --device cuda
```

`--config`, `--styles-config`, `--output-dir`, `--seed`, `--style` e `--limit`
permettono di selezionare un run. I percorsi nei TOML sono relativi alla
radice del repository; quelli passati da CLI alla directory corrente.
`--limit 0` elabora tutte le immagini della cartella, senza ricorsione.

Il preprocessing minimo applica orientamento EXIF, conversione RGB e crop
centrale con resize Lanczos a 512×512: evita deformazioni, ma taglia le parti
periferiche delle foto panoramiche. Non applica miglioramenti di colore/contrasto.
La stessa procedura dovrà essere mantenuta per il confronto con ControlNet.

Ogni run salva in `outputs/rq1/<timestamp>/`:

- `images/<style>/none/<image_id>_seed<seed>.png`;
- `run_config.json`, con configurazione effettiva e input;
- `metadata.jsonl`, con parametri, hash input, seed, tempi, versioni ed errori.

La revision predefinita `main` è mobile: per lo studio definitivo impostare
un commit del modello in `model.revision`. Il seed è ripetibile nei limiti
di hardware e versioni software; non garantisce uguaglianza tra CPU e GPU.
Questa è M1: prompt da validare nel pilot, nessuna metrica o ripresa automatica.

Gli script storici `src/run_baseline.py`, `src/pipeline.py` e
`src/segmentation.py` restano disponibili. Usano segmentazione e ControlNet
text-to-image: le istruzioni seguenti riguardano quel prototipo, non M1.
Il nuovo motore riusabile è in `src/generation/`; il runner è `src/experiment.py`.
Per il prototipo storico servono anche `opencv-python` e `tqdm`.

### Eseguire il prototipo storico con ControlNet

```powershell
python src/run_baseline.py --input_dir src/test_images --output_dir results/baseline --seed 42 --steps 25
```

Per una prova sulla prima immagine della cartella, in ordine alfabetico:

```powershell
python src/run_baseline.py --input_dir src/test_images --output_dir results/test --num_samples 1 --seed 42 --steps 25
```

Per scegliere una foto specifica, inserirla da sola in una cartella e
passare quella cartella a `--input_dir`.

Parametri disponibili:

- `--num_samples`: numero massimo di immagini.
- `--seed`: seed della generazione, default 42.
- `--steps`: step di diffusione, default 25.
- `--guidance_scale`: guidance del prompt, default 7.5.
- `--control_scale`: intensità di ControlNet, default 1.0.
- `--control_end`: frazione della generazione fino alla quale applicare ControlNet, default 0.85.

### Eseguire solo la segmentazione

```powershell
python src/segmentation.py "src/test_images/paesaggio-1024x683.png"
```

Con preprocessing:

```powershell
python src/segmentation.py "src/test_images/paesaggio-1024x683.png" --preprocess
```

La mappa viene salvata accanto alla foto con suffisso `_seg_map.png`.

## Risultati

La baseline salva nella directory di output:

- immagini generate nelle cartelle `cyberpunk`, `gta_5` e `pixel_art`;
- mappe semantiche nella cartella `segmentation_maps`;
- parametri e tempi in `experiment_metadata.json`.

Usare cartelle separate per input e output. La directory di input deve
contenere solo fotografie da elaborare, perché anche le mappe PNG vengono
considerate immagini di input.
