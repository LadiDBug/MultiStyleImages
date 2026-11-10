# MultiStyleImages

Progetto di trasferimento di stile da fotografie con Stable Diffusion 1.5
img2img, ControlNet e adattatori LoRA Pixel Art.

- **RQ1:** confronto fra `none`, `canny`, `depth` e `segmentation`, sugli stili
  configurati e con seed controllati.
- **RQ2:** training di LoRA Pixel Art a rank 4, 8, 16 e 32 e successiva inferenza
  con lo stesso motore di generazione.

## Struttura attuale

```text
MultiStyleImages/
├── main.py                     # Entry point: generazione, training e inferenza LoRA
├── README.md                   # Struttura e comandi attuali
├── configs/
│   ├── main.toml               # Foto, risultati ed esperimento di generazione RQ1
│   ├── base.toml               # SD, device di inferenza, generazione e preprocessing
│   ├── models.toml             # Checkpoint ControlNet
│   ├── styles.toml             # Prompt e negative prompt degli stili
│   ├── evaluation.toml         # Protocollo delle metriche
│   ├── lora_rank4.toml         # Dataset, directory pesi e parametri rank 4
│   ├── lora_rank8.toml
│   ├── lora_rank16.toml
│   └── lora_rank32.toml
├── docs/
│   ├── Resoconto_RQ1.md        # Sintesi RQ1: implementazione, prove e lavoro restante
│   ├── Resoconto_RQ2.md        # Sintesi RQ2: dataset, quattro training e confronto
│   ├── RQ1/                   # Evaluation, stato e riferimenti storici RQ1
│   └── RQ2/                   # Protocollo LoRA, roadmap e architettura condivisa
├── src/
│   ├── generation/            # Caricamento SD/ControlNet/LoRA e generazione
│   ├── preprocessing/         # Preparazione foto, Canny, depth e segmentazione
│   ├── training_lora/         # Dataset immagini/caption e trainer LoRA
│   ├── evaluation/            # Audit e metriche dei risultati
│   └── utils/                 # Lettura e adattamento delle configurazioni
├── data/
│   ├── test_images/           # Fotografie di inferenza nel main.toml attuale
│   └── pixel_art/             # train/, validation/ e Dataset_LoRA.md
├── lora/                      # Pesi LoRA, checkpoint e log del training
├── outputs/                   # Run di generazione, creati quando eseguiti
├── tests/                     # Test offline
└── requirements.txt
```

Gli script del prototipo storico sono stati rimossi; il flusso corrente
parte da `main.py`. Le directory degli artefatti vengono
create o popolate dai rispettivi run; non tutte sono necessariamente presenti.
La [descrizione del dataset](data/pixel_art/Dataset_LoRA.md) contiene fonti,
criteri di selezione e inventario completo. In `data/pixel_art/` gli split
contengono le immagini e i metadata; il documento Markdown non e letto dal trainer.

## Ambiente

Usare Python **3.11 o successivo**. Dalla radice del progetto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Il file requirements include tutte le dipendenze dirette del codice corrente:
PyTorch, Diffusers, Transformers, Accelerate, PEFT, Hugging Face Hub,
Safetensors, Packaging, NumPy, Pillow e OpenCV. Non servono comandi pip
aggiuntivi per LoRA o Canny. Le dipendenze transitive sono risolte da pip.
Le versioni di Diffusers, Transformers e PEFT sono fissate a quelle dei run
verificati; gli altri intervalli rispettano i vincoli del trainer. Non e un
lock completo dell'ambiente: le versioni effettive restano nei run_config.json.

Usare una build PyTorch compatibile con CUDA o Apple Silicon; per versioni
e combinazioni ufficiali vedere [PyTorch](https://pytorch.org/get-started/previous-versions/).
Non sono richiesti xformers, bitsandbytes, timm, datasets o pyarrow per il
codice corrente: il dataset viene letto da immagini e JSONL, non dai Parquet
sorgenti. I modelli configurati usano le implementazioni native Transformers.
Il progetto importa tomllib direttamente e richiede Python 3.11+: installare
tomli non rende compatibili gli interpreti precedenti.

Le dipendenze di generazione sono caricate quando necessarie: il training
e --help non importano Canny. Il primo caricamento dei modelli richiede
accesso a Hugging Face e spazio per la cache; i pesi non sono inclusi nei
requirements. Non e stata eseguita una nuova installazione o un training
per il riordino della documentazione.

Verifica dell'ambiente locale al 10 novembre: i requisiti diretti del file
sono soddisfatti e i 16 test offline passano. `pip check` segnala pero un
conflitto preesistente con `datasets 5.1.0`, che richiede Hugging Face Hub
>=1.31, mentre il trainer richiede Hub <1. Il progetto non importa datasets:
per eseguire anche quel pacchetto serve un ambiente separato compatibile.
Per un'installazione nuova usare un virtualenv pulito; non sono stati
rimossi o cambiati pacchetti nell'ambiente esistente.

## Training LoRA dal main

```bash
python3 main.py --lora lora_rank4.toml train
```

Il nome semplice viene cercato in `configs/`. E accettato anche il percorso
`configs/lora_rank4.toml` o un percorso assoluto. Per gli altri rank basta
scegliere `lora_rank8.toml`, `lora_rank16.toml` o `lora_rank32.toml`.

| Campo nel config LoRA | Significato durante il training |
|---|---|
| `data.input_dir` | Directory dataset con `train/` e `validation/` |
| `data.output_dir` | Directory dei pesi, checkpoint e log |
| `parameters.rank`, `alpha`, `max_train_steps` | Parametri del trainer |
| `experiment.seeds[0]` | Seed del training; `parameters.seed`, se presente, prevale |

Gli altri parametri mantengono i default del trainer, oppure possono essere
specificati nella sezione `[parameters]` con i nomi di `TrainingConfig`.
Il modello base viene da `base.toml`. I percorsi nei TOML sono relativi alla
radice del repository. Nessun config viene riscritto dal programma.

I quattro TOML LoRA indicano `data/pixel_art`, che coincide con il dataset
locale attuale. Il comando di training funziona quindi senza override del
percorso. Se necessario, `--dataset-dir` consente di selezionarlo esplicitamente:

```bash
python3 main.py --lora lora_rank4.toml train --dataset-dir data/pixel_art
```

Per stampare i parametri o controllare il dataset senza addestrare:

```bash
python3 main.py --lora lora_rank4.toml train --print-config
python3 main.py --lora lora_rank4.toml train --dataset-dir data/pixel_art --check-data
```

Per riprendere un checkpoint compatibile nella stessa directory di output:

```bash
python3 main.py --lora lora_rank4.toml train --dataset-dir data/pixel_art --resume-from-checkpoint latest
```

L'output contiene `pytorch_lora_weights.safetensors`, `run_config.json`, log
CSV, checkpoint e `training_summary.json` a completamento. Una directory
non vuota non viene sovrascritta: serve un checkpoint compatibile per il resume.
Il trainer seleziona automaticamente CUDA, MPS o CPU; l'implementazione attuale
usa FP32 su MPS/CPU. Il trainer diretto `python3 -m src.training_lora.train`
resta disponibile e la sua opzione `--config` usa lo schema TOML specifico
descritto nel modulo, diverso dai config LoRA adattati dal main.

## Inferenza con il LoRA selezionato

Dopo aver completato il training:

```bash
python3 main.py --lora lora_rank4.toml inference
```

Il programma carica i pesi da `data.output_dir` del config LoRA e usa il suo
`experiment` per stile, condizioni e seed. Le **fotografie** vengono invece
da `data.input_dir` di `configs/main.toml`; i **risultati** vanno nel suo
`data.output_dir`. Il dataset e la directory dei pesi non vengono usati come
cartella di fotografie o come output delle immagini generate.

Modello e parametri di generazione restano quelli di `base.toml`, checkpoint
ControlNet quelli di `models.toml`, prompt quelli di `styles.toml`. Se il
training registra una revisione immutabile del backbone, viene riutilizzata.
Il device di inferenza resta quello di `base.toml` (attualmente `cuda`);
la selezione automatica del trainer e separata da questa impostazione.

Si possono usare i flag di generazione gia disponibili:

```bash
python3 main.py --lora lora_rank4.toml inference --canny --lora-scale 0.8
```

LoRA rank/alpha vengono confrontati con i metadati del training, se presenti.
Con un adattatore deve essere selezionato un solo stile. `none` indica assenza
di ControlNet: in questa modalita il LoRA rimane comunque attivo. Per il
confronto senza LoRA occorre eseguire un run separato senza adattatore.

## Generazione e comandi

```bash
python3 main.py
python3 main.py --canny
python3 main.py --prepare-canny-only
python3 main.py --lora-path lora/pixel_art_rank_4 --lora-scale 1.0
```

Senza `--lora` e `--lora-path`, `main.py` usa l'esperimento originale in
`configs/main.toml`, senza adattatore. Anche il precedente `--lora-path`
rimane disponibile e richiede un solo stile nel config o un `--prompt`
personalizzato. Tutte le opzioni sono elencate da `python3 main.py --help`.

La generazione elabora ricorsivamente le fotografie e crea una cartella
con timestamp sotto l'output configurato. Contiene `run_config.json` e,
per ciascuna foto, immagine di input, mappe, output per stile/condizione/seed
e `metadata.jsonl`. I riferimenti e l'hash dei pesi LoRA sono inclusi quando
l'adattatore e attivo.

## Valutazione e documentazione

La valutazione e separata dal training e dalla generazione: istruzioni e
comandi sono in [guida alla valutazione](docs/RQ1/EVALUATION_RQ1.md).
Il [documento del dataset](data/pixel_art/Dataset_LoRA.md) elenca tutte le
400 immagini con origine e formato; il trainer legge immagini e
`metadata.jsonl` degli split, non la documentazione.
Per confrontare i rank, mantenere fissi dataset, split e parametri di training.

- [Resoconto RQ1](docs/Resoconto_RQ1.md)
- [Resoconto RQ2](docs/Resoconto_RQ2.md)
- [Protocollo e stato RQ2 Pixel Art](docs/RQ2/RQ2_PIXEL_ART.md)
- [Diagrammi dell'architettura RQ1/RQ2](docs/RQ2/ARCHITETTURA.md)
- [Stato e registro degli esperimenti](docs/RQ1/STATO_PROGETTO_RQ1.md)
- [Roadmap RQ1/RQ2](docs/RQ2/ROADMAP_MultiStyleImages_RQ1_RQ2.md)
- [Riferimento storico RQ1](docs/RQ1/RQ1_PROJECT_REFERENCE.md)
- [README precedente](docs/RQ1/README_STORICO.md)

Le parti storiche conservano decisioni e comandi delle versioni precedenti;
per la CLI corrente fare riferimento a questo README e a `main.py --help`.
