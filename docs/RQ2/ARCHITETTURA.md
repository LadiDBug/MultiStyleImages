# Architettura del progetto — RQ1 e RQ2

Riferimento al codice attuale, 10 novembre 2026. RQ1 e RQ2 condividono il
motore di generazione img2img. RQ2 aggiunge il training degli adattatori LoRA
e il loro caricamento opzionale; i componenti di training e inferenza restano
separati. Le frecce tratteggiate indicano contributi opzionali o una fase
esplicitamente ancora da implementare.

Per protocollo e stato aggiornato dei rank vedere [RQ2_PIXEL_ART.md](RQ2_PIXEL_ART.md).
I quattro rank 4/8/16/32 sul dataset corrente sono completati e verificati;
il pilota rank 8 storico usa un'altra selezione e resta distinto. Vedere
il [resoconto RQ2](../Resoconto_RQ2.md). Il dataset runtime e in `data/pixel_art/` (360/40),
la descrizione e l'inventario in [Dataset_LoRA.md](../../data/pixel_art/Dataset_LoRA.md).

## 1. Vista complessiva e configurazioni

```mermaid
flowchart TB
    USER["CLI: main.py"] --> MODE{"Modalita"}
    MODE -->|"nessun comando / inference senza LoRA"| RQ1["RQ1: esperimento originale"]
    MODE -->|"--lora CONFIG train"| TRAIN["RQ2: training LoRA"]
    MODE -->|"--lora CONFIG inference"| RQ2["RQ2: generazione con LoRA"]

    MAINCFG["configs/main.toml<br/>Fotografie e risultati<br/>Esperimento RQ1"] --> RQ1
    MAINCFG -->|"solo percorsi foto e risultati"| RQ2
    LORACFG["configs/lora_rank4,8,16,32.toml<br/>experiment / data / parameters"] --> ADAPT["src/utils/lora_config.py<br/>Risoluzione config e adattamento parametri"]
    ADAPT --> TRAIN
    ADAPT -->|"stile, condizioni, seed e directory pesi"| RQ2
    BASE["base.toml<br/>Modello SD, generazione e preprocessing"] --> TRAIN
    BASE --> GEN["Motore img2img condiviso<br/>src/generation/"]
    STYLE["styles.toml: prompt<br/>models.toml: checkpoint ControlNet"] --> GEN

    DATA["Dataset Pixel Art<br/>train + validation<br/>Immagini e metadata.jsonl"] --> TRAIN
    TRAIN --> WEIGHTS["Directory data.output_dir del config LoRA<br/>Adapter, checkpoint, configurazione e loss"]
    WEIGHTS --> RQ2
    RQ1 -->|"senza adattatore"| GEN
    RQ2 -->|"con adattatore selezionato"| GEN
    LEGACY["--lora-path<br/>Compatibilita CLI precedente"] -.-> GEN
    PHOTO["Fotografie di inferenza<br/>data.input_dir di main.toml"] --> GEN
    GEN --> RUN["Run con timestamp<br/>Input, mappe, immagini e metadati"]
    RUN --> EVAL["Valutazione separata<br/>src/evaluation/evaluate.py"]
    ECFG["configs/evaluation.toml"] --> EVAL
    EVAL --> METRICS["Metriche per immagine<br/>Riepiloghi e confronti fra condizioni"]
    METRICS -.-> RANKS["Confronto aggregato fra rank RQ2<br/>Da implementare"]
```

`src/utils/config.py` carica e valida l'esperimento di generazione. In RQ2
sostituisce soltanto la sezione `experiment` con quella del config LoRA:
foto e risultati rimangono quelli di `main.toml`. Per il training,
`data.input_dir` del config LoRA e invece il dataset e `data.output_dir`
e la directory dei pesi. `parameters` viene tradotta nella CLI del trainer.
Il seed di training e il primo di `experiment.seeds`, salvo `parameters.seed`.

Il CSV di selezione e i manifest del dataset descrivono costruzione e
provenienza: il trainer carica direttamente gli split e i loro metadata.
Le fotografie usate per i test di inferenza costituiscono un insieme distinto.

## 2. Generazione condivisa: RQ1 e inferenza RQ2

```mermaid
flowchart TB
    INPUT["Scansione ricorsiva fotografie<br/>main.py"] --> PREP{"Preprocessing fotografico"}
    PREP -->|"default"| RESIZE["EXIF, RGB, resize e crop centrale<br/>src/preprocessing/img_preprocess.py"]
    PREP -->|"--no-preprocess"| RGB["RGB alle dimensioni originali"]
    RESIZE --> IMAGE["Stessa fotografia per tutte le condizioni"]
    RGB --> IMAGE
    IMAGE --> CONDITION{"Condizione selezionata<br/>Un ramo per generazione"}
    CONDITION -->|"none"| NONE["Img2img senza ControlNet"]
    CONDITION -->|"canny"| CANNY["canny.py<br/>Estrazione bordi"]
    CONDITION -->|"depth"| DEPTH["depth.py<br/>MiDaS: Intel/dpt-hybrid-midas"]
    CONDITION -->|"segmentation"| SEG["semantic.py<br/>UPerNet ConvNeXt / ADE20K"]
    CANNY --> MAP["Mappa RGB allineata alla fotografia"]
    DEPTH --> MAP
    SEG --> MAP
    MAP --> CN["ControlNet specifico della condizione<br/>Checkpoint da models.toml"]
    NONE --> PIPE["pipeline.py: load_pipeline<br/>SD 1.5 img2img / ControlNet img2img"]
    CN --> PIPE
    IMAGE --> PIPE
    ADAPTER["RQ2: adapter LoRA locale<br/>Verifica metadati, hash e revisione base"] -.->|"attach_lora e scala adattatore"| PIPE
    TEXT["styles.toml<br/>Prompt e negative prompt"] --> LOOP["generate(): foto x stile x condizione x seed<br/>Step, strength, CFG e scala ControlNet"]
    PIPE --> LOOP
    LOOP --> SD["Diffusers<br/>CLIP testo + VAE immagine<br/>U-Net, scheduler e denoising<br/>Decodifica VAE"]
    SD --> OUT["Output per foto<br/>input, maps, images/stile/condizione<br/>run_config.json e metadata.jsonl"]
    MAP --> OUT
    MAP -.->|"--prepare-...-only: termina qui"| PREPONLY["Solo input e mappe<br/>Nessun caricamento Stable Diffusion"]
```

Le condizioni vengono eseguite separatamente: il codice non combina Canny,
depth e segmentation in un Multi-ControlNet. La fotografia preparata e
condivisa; il modello di preprocessing appreso viene liberato prima di
caricare la pipeline di generazione. Si carica una pipeline per condizione
e fotografia, riutilizzandola per gli stili e i seed di quella condizione.

RQ1 usa gli stili di `main.toml`, attualmente GTA V, Pixel Art e Watercolor,
senza LoRA. In RQ2 viene caricato un adattatore per run e un solo stile.
La condizione `none` con LoRA significa **senza ControlNet, con LoRA attivo**;
la baseline senza adattatore richiede un run separato.

L'inferenza usa il device e la precisione indicati in `base.toml` e lo
scheduler del checkpoint caricato. I flag di preparazione salvano input e
mappe senza Stable Diffusion. Il seed e ricreato per ogni generazione.

## 3. Training RQ2: componenti congelati e parametri aggiornati

```mermaid
flowchart TB
    CFG["main.py --lora CONFIG train<br/>rank, alpha, step e percorsi"] --> TRAINER["src/training_lora/train.py<br/>TrainingConfig, audit e revisioni"]
    DATA["Dataset<br/>Immagini Pixel Art + caption"] --> DS["dataset.py: PixelArtDataset<br/>RGB, nearest resize, crop centrale<br/>Tensori normalizzati e tokenizzazione CLIP"]
    DS --> TR["Split train<br/>DataLoader e sampler con seed"]
    DS --> VAL["Split validation separato<br/>Nessun aggiornamento pesi"]
    TRAINER --> MODELS["SD 1.5 preaddestrato"]
    MODELS --> VAE["VAE congelato<br/>Immagine verso distribuzione latente"]
    MODELS --> CLIP["Text encoder CLIP congelato<br/>Caption verso embedding"]
    MODELS --> UNET["U-Net base congelata<br/>LoRA addestrabili su attenzione<br/>to_q, to_k, to_v, to_out.0"]
    TR -->|"immagini"| VAE
    TR -->|"caption tokenizzate"| CLIP
    VAE --> LATENT["Campionamento latenti<br/>Rumore e timestep casuali<br/>DDPMScheduler.add_noise"]
    LATENT --> UNET
    CLIP --> UNET
    UNET --> LOSS["Loss MSE denoising<br/>Target epsilon o v_prediction"]
    LOSS --> OPT["Accelerate + AdamW + scheduler LR<br/>Accumulo gradienti, clipping<br/>Gradient checkpointing"]
    OPT -->|"Aggiorna solo matrici LoRA"| UNET
    VAL --> CHECK["validation_loss<br/>Stessi componenti congelati<br/>Rumore e timestep con seed fissato"]
    UNET --> CHECK
    CHECK --> LOG["Metriche CSV<br/>Training loss e validation loss"]
    OPT --> SAVE["Checkpoint riprendibili<br/>Adapter + optimizer, scheduler e RNG<br/>Posizione del sampler e firma dati"]
    SAVE -->|"resume compatibile"| TRAINER
    OPT --> EXPORT["Export finale<br/>pytorch_lora_weights.safetensors<br/>run_config.json e training_summary.json"]
    EXPORT --> INFER["main.py --lora CONFIG inference<br/>Caricamento sul backbone SD compatibile"]
```

Il training non usa fotografie di test, mappe ControlNet o metriche di stile.
La validation loss misura il denoising su dati separati; non dimostra da sola
la qualita del trasferimento Pixel Art.

Nella revisione locale a 400 immagini lo split e 360 train / 40 validation.
I default del trainer sono risoluzione 512, batch 1, accumulo 4, AdamW con
learning rate 1e-4, 1000 aggiornamenti e seed 42. I config rank impostano
alpha uguale al rank. VAE ed embedding testuali vengono ancora calcolati a
ogni batch: non e implementata una cache dei latenti o degli embedding.

Accelerate sceglie automaticamente CUDA, MPS o CPU. L'implementazione attuale
usa FP32 su MPS/CPU e applica la mixed precision richiesta su CUDA. Questo
comportamento e separato dalla scelta del device per la generazione.

## 4. Valutazione delle immagini e confronto degli esperimenti

```mermaid
flowchart TB
    RUN["Un run di generazione<br/>Fotografie preparate + output + metadati"] --> AUDIT["records.py: audit_run<br/>Completezza e coerenza<br/>foto / stile / condizione / seed"]
    CFG["evaluation.toml<br/>Modelli valutatori e testi stile"] --> EVAL["evaluate.py<br/>Valutazione separata dalla generazione"]
    AUDIT -->|"Output validi"| EVAL
    EVAL --> CI["style.py: CLIP-I<br/>Similarita globale input / output"]
    EVAL --> CT["style.py: CLIP-T<br/>Output / descrizione dello stile"]
    EVAL --> SEM["content.py: SegFormer ADE20K<br/>Input / output su griglia comune<br/>mIoU semantica"]
    EVAL --> DINO["dino.py: DINOv2<br/>Similarita visiva input / output"]
    CI --> PER["per_image_metrics.csv"]
    CT --> PER
    SEM --> PER
    DINO --> PER
    PER --> AGG["Blocchi foto / stile / seed completi<br/>Media seed per foto, poi media foto<br/>Intervalli bootstrap per foto"]
    AGG --> SUMMARY["summary_metrics.csv<br/>paired_vs_none.csv<br/>Configurazioni e riepiloghi JSON"]
    SUMMARY -.-> RQ2["Confronto fra run senza LoRA e rank 4/8/16/32<br/>Aggregatore RQ2 dedicato ancora da implementare"]
```

CLIP-I e DINO-I misurano similarita visiva, CLIP-T l'allineamento al testo
di valutazione dello stile: non sono percentuali di qualita. SegFormer e
un valutatore distinto dall'UPerNet usato per il conditioning.

Il comando di evaluation opera su **un singolo run** ed e separato dal main.
Le metriche sono riutilizzabili sugli output RQ2, ma i riepiloghi esistenti
confrontano le condizioni con `none` all'interno di quel run. Se il run usa
un LoRA, anche `none` usa quel LoRA: `paired_vs_none.csv` non e quindi un
confronto contro il modello senza adattatore.

Il confronto automatico appaiato fra rank, con controlli espliciti su dataset,
training e LoRA, non e ancora implementato: `src/evaluation/lora_eval.py`
e attualmente vuoto. Il ramo tratteggiato rappresenta questo lavoro futuro.
Il device automatico dell'evaluator seleziona CUDA oppure CPU, non MPS.

## Comandi di accesso ai flussi

```bash
# RQ1: esperimento configurato, senza adattatore
python3 main.py

# RQ2: training, con percorso locale del dataset
python3 main.py --lora lora_rank4.toml train --dataset-dir data/pixel_art

# RQ2: inferenza con i pesi del rank selezionato
python3 main.py --lora lora_rank4.toml inference

# Valutazione separata di un run
python3 -m src.evaluation.evaluate --run-dir outputs/NOME_RUN
```

I config non vengono riscritti. Per gli altri rank cambia il nome del TOML.
Il dataset attuale e in `data/pixel_art`, come indicato nei TOML. La
directory contiene i due split con immagini e metadata e il documento
`Dataset_LoRA.md` con fonti e selezione. Le fotografie di inferenza
rimangono quelle configurate in `main.toml`.
