# RQ1 — Project Reference & Implementation Roadmap

> **Scopo del documento:** riferimento persistente per riprendere il progetto in sessioni future, ricostruire le decisioni, evitare di cambiare inconsapevolmente protocollo sperimentale e guidare l'implementazione.
>
> **Stato del corpo RQ1:** riferimento storico del 13 ottobre 2026, non descrizione della CLI corrente. Il piano originario menzionava Cyberpunk e YAML; gli esperimenti RQ1 correnti usano GTA V, Pixel Art e Watercolor e config TOML.
>
> **Aggiornamento RQ2 al 10 novembre 2026:** la fase e attiva e usa esclusivamente **LoRA Pixel Art**. Dataset pronto (400 immagini, 360/40), trainer e inferenza integrati nel main; tutti i rank 4/8/16/32 completati e verificati sul dataset corrente, distinti dal vecchio pilota rank 8. Riferimento operativo: [RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md).

## 0. Contesto e motivazione

Progetto magistrale di **Deep Learning, image-to-image style transfer**: trasformare fotografie reali in tre stili visivi distinti:

1. **GTA V** — resa da videogioco 3D semi-realistica;
2. **Cyberpunk** — estetica futuristica/neon (specificare se ci si riferisce al genere o precisamente a Cyberpunk 2077);
3. **Pixel Art** — stilizzazione a pixel, con semplificazione di colori e dettagli.

Feedback del professore: l'assemblaggio di Stable Diffusion, ControlNet, prompt engineering e LoRA **non è sufficiente**. Serve una domanda scientifica chiara, ipotesi testabili, confronto controllato, ablation study e metriche distinte per contenuto/struttura e stile.

Il progetto complessivo prevede:
- **RQ1 (focus attuale):** tipo di condizionamento strutturale (nessuno, edges, depth, segmentation) in tre stili.
- **RQ2 (fase attiva, protocollo aggiornato separato):** influenza del **rank LoRA Pixel Art** sul trade-off fra contenuto e stile, con confronto off/4/8/16/32; vedere [RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md).

## 1. Research Question

> **RQ1: How do different structural conditioning methods (semantic segmentation, edge maps, and depth maps) affect content preservation and style fidelity across three visually distinct styles: GTA V, Cyberpunk, and Pixel Art?**

**Fattori variati:**
- `condition ∈ {none, canny, depth, segmentation}`;
- `style ∈ {gta5, cyberpunk, pixel_art}`.

**Output:** immagini stilizzate, metriche di conservazione del contenuto, metriche di fedeltà allo stile, confronti qualitativi.

**Ipotesi da verificare, NON conclusioni:**
- Canny tende a vincolare maggiormente contorni e geometria locale;
- Depth tende a vincolare relazioni spaziali/globali;
- Segmentation tende a vincolare layout e classi semantiche;
- L'efficacia di una mappa cambia a seconda dello stile: pixel art può richiedere una perdita intenzionale dei dettagli, per cui preservare tutti i bordi non è necessariamente ottimale.

## 2. Disegno sperimentale

| Condition | GTA V | Cyberpunk | Pixel Art |
|---|---|---|---|
| `none` (baseline) | ✓ | ✓ | ✓ |
| `canny` | ✓ | ✓ | ✓ |
| `depth` | ✓ | ✓ | ✓ |
| `segmentation` | ✓ | ✓ | ✓ |

**12 configurazioni** prima di moltiplicare per numero di immagini e seed. Esempio preliminare: 50 immagini × 12 configurazioni × 2 seed = **1.200 generazioni**. Prima fare un *pilot* piccolo (es. 10 immagini × 12 × 1 seed).

### Cosa controllare

- Stesso backbone Stable Diffusion, famiglia di scheduler e procedura img2img;
- stessi input, split, crop/resize, risoluzione, prompt per stile, seed abbinati per confronto, inference steps, guidance e denoising strength;
- **stessa LoRA fissata all'interno di ciascuno stile**, presente tanto in `none` quanto nei rami ControlNet, *se* si adotta l'opzione LoRA;
- stesso valore iniziale di `controlnet_conditioning_scale` nei tre rami, registrando che valori identici non implicano necessariamente forza effettiva identica.

**Caveat metodologico:** passando da Canny a Depth o Segmentation cambiano anche i **checkpoint ControlNet**, non soltanto le mappe. I risultati caratterizzano le *configurazioni di condizionamento* e non l'effetto causale isolato della sola rappresentazione. Dichiararlo nei limiti.

### Come ottenere i tre stili

- **Baseline tecnica iniziale:** prompt-only per verificare il funzionamento della pipeline.
- **Studio definitivo:** decidere e validare se prompt-only è sufficiente o se servono **tre LoRA stilistiche fisse** (una per stile), compatibili con lo stesso backbone. La fedeltà ai tre stili deve essere recognoscibile nel pilot.
- Non confondere RQ1 con RQ2: durante RQ1 non modificare rank/pesi della LoRA al variare del conditioning.

## 3. Baseline: perché è semplice e dove vive

La baseline **non è una nuova architettura da addestrare**: è **Stable Diffusion img2img senza ControlNet** (`condition=none`). Riceve una foto reale e un prompt di stile e produce un'immagine. Più avanti può utilizzare la LoRA fissa dello stile, ma senza mappa.

**Baseline tecnica:** una foto → img2img → output salvato. Serve a verificare caricamento del modello, generazione, seed e salvataggio.

**Baseline sperimentale:** dataset → tutti e tre gli stili → stesso adattamento stilistico delle condizioni ControlNet → parametri fissati → output + metadati. È il confronto scientifico valido per RQ1.

**Non creare una cartella o implementazione `baseline/` separata**: la baseline è `condition: none` della stessa pipeline condivisa. L'essenzialità del codice non rende banale la ricerca: il contributo è nel confronto controllato e nell'evaluation, non nel numero di righe della baseline.

## 4. Architettura repository (solo RQ1)

```text
image-to-image/
├── configs/
│   ├── base.yaml             # backbone, inferenza, seed, risoluzione
│   ├── models.yaml           # checkpoint ControlNet
│   ├── styles.yaml           # prompt e LoRA fissa opzionale per stile
│   └── rq1.yaml              # matrice condition × style
├── data/                     # non versionare dataset pesanti
│   ├── raw/
│   ├── processed/
│   │   ├── canny/
│   │   ├── depth/
│   │   └── segmentation/
│   └── references/
│       ├── gta5/
│       ├── cyberpunk/
│       └── pixel_art/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── canny.py
│   │   ├── depth.py
│   │   └── segmentation.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── pipeline.py       # carica SD img2img o SD+ControlNet img2img
│   │   └── generator.py      # inferenza tramite interfaccia unificata
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── content.py
│   │   ├── style.py
│   │   └── evaluator.py
│   └── utils/
│       ├── config.py
│       └── seed.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_rq1.py
│   └── evaluate_rq1.py
├── notebooks/
│   └── kaggle_rq1.ipynb     # orchestration/setup, NON duplicare codice
├── outputs/                 # non versionare grandi output
├── tests/
│   ├── test_preprocessing.py
│   └── test_config.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Responsabilità

- **`preprocessing/*`**: da immagine RGB a mappa, con risultati in cache; non deve importare Stable Diffusion.
- **`generation/pipeline.py`**: costruire/caricare la pipeline giusta e l'eventuale LoRA.
- **`generation/generator.py`**: funzione `generate(...)`, con mappa opzionale, stessi parametri per tutti i rami.
- **`scripts/prepare_data.py`**: preprocessare input e produrre mappe una sola volta.
- **`scripts/run_rq1.py`**: iterare su condizioni, stili, input e seed; salvare output e metadata.
- **`evaluation/*`**: calcolare metriche senza modificare la generazione.
- **`notebooks/kaggle_rq1.ipynb`**: installare, montare i dati, lanciare gli script; NON copiare la business logic nel notebook.

### Logica di generazione condivisa (pseudocodice)

```python
for condition in conditions:
    pipe = load_pipeline(condition)      # none = img2img; altri = ControlNet img2img
    for style in styles:
        set_fixed_style(pipe, style)     # opzionale LoRA, non deve contaminare altri stili
        for image in dataset:
            condition_image = load_map(image, condition)  # None se condition == 'none'
            for seed in seeds:
                output = generate(pipe, image, condition_image, style.prompt, seed)
                save_image_and_metadata(output, condition, style, image, seed)
```

**Attenzione:** scaricare/disattivare correttamente la LoRA precedente quando si cambia stile; una soluzione alternativa è separare i run per stile per ridurre errori di stato e memoria GPU.

## 5. Configurazioni iniziali (da validare nel pilot)

```yaml
# configs/rq1.yaml
experiment: rq1_structural_conditioning
styles: [gta5, cyberpunk, pixel_art]
conditions: [none, canny, depth, segmentation]
seeds: [42, 123, 456]
```

```yaml
# configs/base.yaml — valori di PARTENZA, non parametri scientificamente ottimizzati
model_id: stable-diffusion-v1-5/stable-diffusion-v1-5
height: 512
width: 512
steps: 30
guidance_scale: 7.5
strength: 0.65
controlnet_conditioning_scale: 0.8
```

Checkpoints candidati SD 1.5 (verificare versioni e compatibilità reale con preprocessing):

```yaml
# configs/models.yaml
controlnets:
  canny: lllyasviel/control_v11p_sd15_canny
  depth: lllyasviel/control_v11f1p_sd15_depth
  segmentation: lllyasviel/control_v11p_sd15_seg
```

**Criticità importante:** ControlNet Seg usa tipicamente palette/classi ADE20K; non è sicuro passare direttamente mappe Cityscapes con colori arbitrari. Per Depth il preprocessore deve corrispondere alla distribuzione di mappe attesa dal ControlNet (es. MiDaS per alcune versioni), anche se nel futuro si potrà testare Depth Anything. Verificare ciò nel pilot.

## 6. Dataset e preprocessing

- **Input reali:** valutare Cityscapes (utile perché ha GT semantico, con vincoli/licenze e split da rispettare) o BDD100K. Avviare con poche immagini e poi costruire un sottoinsieme rappresentativo.
- **Riferimenti di stile:** raccolte lecite di screenshot/reference GTA V, Cyberpunk e Pixel Art, separate dagli input, con diversità di scene.
- **Canny:** `cv2.Canny()` con soglie configurabili; registrarle.
- **Depth:** stimatore monoculare e normalizzazione documentati.
- **Segmentation:** segmentatore pretrained + mapping classi/palette compatibile col checkpoint.
- Salvare mappe su disco e ricaricarle, senza rieseguire i preprocessori per ogni stile.

## 7. Evaluation: il principale contenuto di ricerca

Misurare **due dimensioni separatamente**; non confondere metrica con verità assoluta.

| Dimensione | Candidate metrics | Avvertenze |
|---|---|---|
| Preservazione semantica | mIoU / consistenza semantica su classi comuni | Segmentatore di valutazione indipendente dal preprocessore; domain shift sugli output stilizzati |
| Preservazione dei bordi | Edge F1 con tolleranza spaziale | Pixel Art perde intenzionalmente dettagli, quindi bordi diversi non implicano qualità inferiore |
| Preservazione spaziale | Correlazione depth / ranking delle distanze | Stime di profondità su stili diversi possono avere bias |
| Fedeltà allo stile | Similarità CLIP con reference per stile, o valutatore di stile separato | CLIP misura anche semantica/composizione, non solo stile |
| Validazione percettiva | Human evaluation su subset, criteri e scala definiti | Randomizzazione, istruzioni coerenti, idealmente più valutatori |

Per ogni configurazione: aggregare per immagine e riportare media, dispersione/intervalli di confidenza. Seed e immagini generati dallo stesso input non sono osservazioni completamente indipendenti. Visualizzare **trade-off content preservation vs style fidelity** e risultati disaggregati per stile, non soltanto una classifica globale.

**Tabella risultati prevista:** righe per `none/canny/depth/segmentation`, colonne mIoU, Edge F1, Depth correlation e Style Score, ripetuta o filtrabile per stile. Affiancare una griglia comparativa `input | none | canny | depth | segmentation` per ciascuno dei tre stili.

## 8. Output, tracciamento e riproducibilità

```text
outputs/rq1/
├── images/
│   ├── gta5/{none,canny,depth,segmentation}/
│   ├── cyberpunk/{none,canny,depth,segmentation}/
│   └── pixel_art/{none,canny,depth,segmentation}/
├── metadata.csv
├── metrics.csv
├── run_config.yaml
└── figures/
```

Nome output esemplificativo: `city_001_seed42.png`.

Ogni generazione deve conservare: `image_id`, style, condition, seed, prompt/negative prompt, model e revision, eventuale LoRA/checkpoint/scale, ControlNet checkpoint, strength, guidance scale, steps, scheduler, dimensioni, percorso output, data e possibilmente commit Git. Salvare anche gli errori e poter riprendere run interrotte.

## 9. Milestone operative (ordine consigliato)

### M1 — Baseline tecnica (PR/commit iniziale)

- [ ] Creare cartelle, `.gitignore`, `requirements.txt`, YAML essenziali.
- [ ] `src/generation/pipeline.py`: caricare `StableDiffusionImg2ImgPipeline` senza ControlNet.
- [ ] `src/generation/generator.py`: generare un'immagine in modo deterministico per seed (nei limiti hardware/software).
- [ ] `scripts/run_rq1.py`: eseguire il caso `style=cyberpunk, condition=none, limit=1`.
- [ ] Salvare PNG e almeno i metadati minimi.

**Definition of Done:** una foto realistica in input, un output img2img valido, modello e parametri registrati, nessuna mappa. Il codice è volutamente breve.

### M2 — Baseline sperimentale

- [ ] Spostare prompt/parametri nei file YAML.
- [ ] Elaborare un set di immagini per **tutti e tre gli stili**.
- [ ] Validare la riconoscibilità dei target: decidere prompt-only vs LoRA stilistiche fisse.
- [ ] Bloccare per il confronto la scelta stilistica all'interno di ciascuno stile.

### M3 — Preprocessing strutturale

- [ ] Implementare e verificare Canny.
- [ ] Implementare e verificare Depth.
- [ ] Implementare e verificare Segmentation, class mapping e palette.
- [ ] Mettere in cache le mappe, con naming coerente con gli input.

### M4 — ControlNet

- [ ] Aggiungere un loader opzionale ControlNet a `pipeline.py`.
- [ ] `generator.py` accetta `condition_image=None` oppure una mappa valida.
- [ ] Pilot delle quattro condizioni sulla stessa fotografia con stessi parametri.
- [ ] Validare VRAM, preprocessing e casi di incompatibilità.

### M5 — RQ1 completa ed evaluation

- [ ] Eseguire matrice 4 condizioni × 3 stili sul dataset e seed selezionati.
- [ ] Implementare content metrics, style metrics e validazione umana.
- [ ] Salvare CSV, CI/varianza, scatter dei trade-off e griglie di output.
- [ ] Scrivere limiti e conclusioni (senza inventare risultati).

## 10. Branch e convenzioni Git

- Branch RQ1: `experiment/conditioning-ablation`.
- Mantenere `main` funzionante; PR piccole per baseline, mappe, ControlNet, multi-style, evaluation.
- Non committare dataset, checkpoint e grandi output. Conservare invece **manifest**, script di download/preparazione, config e piccoli file di test.
- Kaggle è l'ambiente previsto per le run GPU; usare il codice Python del repository come sorgente unica, notebook solo come launcher.

## 11. Decisioni chiuse vs aperte

### Già stabilito

- RQ1 riguarda **4 condizioni × 3 stili**.
- Stili: GTA V, Cyberpunk, Pixel Art.
- La baseline è **SD img2img senza ControlNet**, non un modello custom.
- Repository modulare, configurazioni esterne, riuso del motore di inferenza.
- La valutazione separa content preservation e style fidelity.
- Il codice RQ2 e ora implementato separatamente: trainer `src/training_lora/`, main `--lora CONFIG train/inference`, LoRA Pixel Art soltanto sulla U-Net. Non cambiare il protocollo RQ1 per introdurre i confronti fra rank.

### Da definire / verificare

1. Dataset input definitivo, licenze, split e dimensione dello studio.
2. Dataset reference per stile e procedura di style evaluation.
3. Prompt-only vs tre LoRA fisse per RQ1; verificare che i checkpoint non alterino il backbone.
4. Checkpoint ControlNet, loro versioni e preprocessing perfettamente compatibile.
5. Risoluzione/crop (512×512 può distorcere immagini widescreen: scegliere una procedura coerente).
6. Metriche esatte e protocollo human evaluation; possibilità di confound nei metric extractor.
7. Numero di seed, budget Kaggle, strategia di aggregazione, eventuali analisi di sensitività su conditioning scale.

## 12. Contesto aggiornato per riprendere il progetto

Leggere prima [STATO_PROGETTO_RQ1.md](STATO_PROGETTO_RQ1.md) e
[RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md), poi il README principale e
[ARCHITETTURA.md](../RQ2/ARCHITETTURA.md). RQ1 e img2img senza LoRA, con GTA V,
Pixel Art e Watercolor e quattro condizioni; l'utente riferisce run con
seed 42/123/456, senza certificazione nuova degli aggregati in questo aggiornamento.
RQ2 usa Pixel Art, dataset in data/pixel_art con split 360/40 e supporto in
data/pixel_art/Dataset_LoRA.md. I quattro rank correnti sono completati; il
vecchio pilota rank 8 su altra selezione non e confrontabile direttamente.
Proseguire con inferenza e confronti appaiati; lora_eval.py e ancora vuoto.
Il main e la CLI corrente sono nel README principale; i percorsi YAML e i
pseudocodici del corpo storico sopra non sono comandi da eseguire oggi.
