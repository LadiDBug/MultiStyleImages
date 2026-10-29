# MultiStyleImages — Memoria del progetto e prossimi passi RQ1

> **Aggiornato al:** 29 ottobre 2026.  
> **Scopo:** conservare il contesto necessario per riprendere il progetto, distinguendo implementazione, prove disponibili e lavoro ancora da svolgere.  
> **Snapshot del repository:** branch `conditioning-ablation`, base commit `593d702` (`add placeholder files for RQ2`), più implementazione evaluation nella working tree non ancora committata.  
> **Fonti:** codice, configurazioni, cronologia Git, artefatti locali e aggiornamenti dell'utente. Verifiche offline dell'evaluation eseguite; nessuna nuova generazione o valutazione con pesi preaddestrati eseguita in questa sessione.

## 1. Da dove ripartiamo

Il progetto dispone di una pipeline modulare per confrontare **Stable Diffusion 1.5 img2img senza controllo strutturale** con tre varianti **ControlNet img2img: Canny, Depth e Segmentation**. Il runner gestisce più fotografie, stili e seed, salva gli input effettivi, le mappe, le configurazioni e i metadati delle generazioni.

La configurazione attuale seleziona **GTA V, Pixel Art e Watercolor**. La strategia stilistica implementata è basata sui prompt: il loader corrente non carica LoRA.

**La RQ1 non è ancora conclusa scientificamente.** L'evaluation è ora implementata e verificata con test offline: audit del run, CLIP-I, CLIP-T, consistenza semantica con SegFormer, similarità visiva DINOv2 (`dino_i`), CSV, aggregazione per input e confronti appaiati con `none`. Non sono ancora disponibili metriche su output reali. La guida completa è [EVALUATION_RQ1.md](EVALUATION_RQ1.md).

**Aggiornamento dell'utente:** le foto sono state caricate in `raw` ed è in corso su Kaggle la generazione per tutti gli stili, senza preprocessing fotografico (`--no-preprocess`). La configurazione effettiva e la completezza del run verranno verificate sui file scaricati; non è stata effettuata un'ispezione diretta della sessione Kaggle. Il TOML locale punta ancora a `data/test_images`, mentre qui i dati sono in `data/raw`: questa differenza locale non dimostra un problema nel run Kaggle. La prossima azione è completare/scaricare il run ed eseguire l'audit, poi un primo calcolo delle metriche.

## 2. Research question e obiettivo scientifico

La domanda centrale è: **come cambia il compromesso tra preservazione del contenuto e fedeltà allo stile quando si usa un diverso condizionamento strutturale?**

Una formulazione coerente con gli stili attualmente configurati è:

> How do different structural conditioning methods (semantic segmentation, edge maps, and depth maps) affect content preservation and style fidelity across GTA V, Pixel Art, and Watercolor?

Questa formulazione fotografa la configurazione corrente; va confermata nel protocollo finale perché il documento storico `RQ1_PROJECT_REFERENCE.md` menziona ancora **Cyberpunk** al posto di Watercolor.

La foto deve cambiare resa visiva mantenendo soggetti e organizzazione della scena. Queste due esigenze possono entrare in tensione: vincolare molto la struttura potrebbe limitare alcune trasformazioni stilistiche; lasciare più libertà potrebbe alterare oggetti e disposizione spaziale. È un'ipotesi di lavoro da misurare, non un risultato già ottenuto.

I sistemi da confrontare sono:

| Condizione | Informazione disponibile al generatore | Ruolo nel confronto |
|---|---|---|
| `none` | Foto iniziale + prompt | Baseline img2img senza ControlNet |
| `canny` | Foto iniziale + prompt + mappa dei bordi | Controllo di contorni e dettagli locali |
| `depth` | Foto iniziale + prompt + profondità relativa | Controllo delle relazioni spaziali |
| `segmentation` | Foto iniziale + prompt + mappa semantica | Controllo del layout delle classi |

Anche `none` riceve la fotografia iniziale: l'assenza di ControlNet non significa assenza di informazione sul contenuto.

Ogni ramo strutturale usa un checkpoint ControlNet diverso. Le conclusioni riguarderanno quindi i **sistemi di condizionamento concretamente utilizzati**, comprensivi di preprocessore e checkpoint, e non il solo formato della mappa isolato da ogni altra differenza.

Con tre stili e quattro condizioni ci sono **12 combinazioni per fotografia e seed**. RQ1 non richiede training: l'obiettivo corrente è confrontare modelli preaddestrati a parametri di inferenza controllati. Lo studio del rango LoRA appartiene a RQ2 e resta successivo.

## 3. Cosa è stato costruito finora

### 3.1 Dal prototipo storico al motore img2img

Nel repository convivono due percorsi:

- **Prototipo storico:** `src/run_baseline.py`, `src/pipeline.py`, `src/segmentation.py` e `src/img_preprocessing.py`. Usa segmentazione e `StableDiffusionControlNetPipeline`, quindi un percorso ControlNet text-to-image. Include anche scelte differenti, come un VAE aggiuntivo e scheduler UniPC.
- **Percorso attuale RQ1:** `main.py`, `src/generation/`, `src/preprocessing/` e `src/utils/config.py`. Usa img2img sia per la baseline sia per i rami ControlNet.

Per proseguire la RQ1, il riferimento operativo è il secondo percorso. Gli output dei due motori non sono direttamente intercambiabili in uno stesso confronto: cambiano componenti e modalità con cui la fotografia entra nella generazione.

La cronologia Git recente mostra modifiche a prompt, guidance, strength, intensità ControlNet e soglie Canny, oltre all'estensione del runner a tutte le fotografie della cartella. Queste modifiche documentano sviluppo e taratura esplorativa; non costituiscono una selezione quantitativa dei parametri migliori.

### 3.2 Configurazioni separate per responsabilità

| File | Funzione attuale |
|---|---|
| `configs/base.toml` | Backbone, device, precisione, dimensioni, inferenza e preprocessori |
| `configs/main.toml` | Stili, condizioni, seed e percorsi dell'esperimento |
| `configs/models.toml` | Checkpoint ControlNet per ciascuna condizione |
| `configs/styles.toml` | Prompt e negative prompt per stile |
| `src/utils/config.py` | Caricamento TOML, validazione e risoluzione dei percorsi |

Il loader controlla, tra le altre cose, che gli stili esistano, le condizioni siano supportate e prive di duplicati, i seed siano interi non negativi, i checkpoint richiesti siano definiti e le soglie Canny siano valide. I percorsi nei TOML vengono risolti rispetto alla radice del repository.

### 3.3 Preparazione condivisa delle fotografie

`src/preprocessing/img_preprocess.py` implementa orientamento EXIF, conversione RGB, crop centrale e resize Lanczos alle dimensioni configurate. Non aggiunge correzioni di colore o contrasto.

Nel percorso standard il runner estrae tutte le mappe dalla **stessa fotografia preparata** fornita a img2img. Questo mantiene allineati crop, dimensioni e contenuto tra le condizioni. Il crop evita deformazioni del rapporto d'aspetto ma può eliminare soggetti periferici: va mantenuto coerente e controllato nel pilot.

È disponibile anche `--no-preprocess`: converte l'input in RGB e lascia gli adattamenti dimensionali a Diffusers. Non esegue il percorso EXIF/crop/resize condiviso. È una variante esplorativa da registrare; per il confronto principale bisogna scegliere una sola procedura e mantenerla uguale per tutti i metodi.

### 3.4 Canny

`src/preprocessing/canny.py` converte la foto RGB in scala di grigi e applica OpenCV Canny. Restituisce un'immagine RGB con bordi bianchi su sfondo nero, con le stesse dimensioni della foto.

Le soglie sono configurabili. Esiste la modalità `--prepare-canny-only`, che salva input e mappa senza caricare Stable Diffusion. In `outputs/` sono presenti mappe Canny e configurazioni storiche: questa è la prova locale più concreta del lavoro di preprocessing già svolto.

### 3.5 Segmentazione semantica

`src/preprocessing/semantic.py` implementa `SemanticSegmentor` con `openmmlab/upernet-convnext-base` e le **150 classi ADE20K**. I logits vengono riportati alle dimensioni dell'input prima dell'argmax. Gli ID `0..149` sono convertiti nella palette ADE20K senza offset aggiuntivo.

Il runner salva sia la mappa RGB per ControlNet sia gli ID di classe in un PNG separato. Esiste la modalità `--prepare-segmentation-only`. Il codice controlla il numero di classi, ma la compatibilità effettiva dell'intera catena preprocessore–ControlNet deve essere verificata nel pilot.

### 3.6 Profondità

`src/preprocessing/depth.py` implementa `DepthEstimator` con `Intel/dpt-hybrid-midas`. La predizione è riportata alle dimensioni dell'input e conservata come array float32.

Si tratta di **profondità inversa relativa**, non di distanze in metri. La mappa per ControlNet usa normalizzazione min-max per immagine a `0..255`, con vicino chiaro e lontano scuro, replicata su tre canali RGB. Il runner salva anche i valori in `.npy`, utili per analisi successive. Esiste la modalità `--prepare-depth-only`.

Anche qui l'implementazione non sostituisce la verifica visiva e end-to-end: orientamento della profondità, normalizzazione e risposta del checkpoint devono essere controllati su scene diverse.

### 3.7 Generazione condivisa

`src/generation/pipeline.py` seleziona:

- `StableDiffusionImg2ImgPipeline` per `none`;
- `StableDiffusionControlNetImg2ImgPipeline` e relativo ControlNet per gli altri tre rami.

Il loader gestisce device e precisione e segnala CUDA non disponibile o float16 richiesto su CPU. **Non carica LoRA e non forza attualmente UniPC:** la riga che sostituisce lo scheduler è commentata. Viene usato lo scheduler caricato dal checkpoint, il cui nome è poi registrato nei metadati.

`src/generation/generator.py` riceve la pipeline già caricata, verifica foto e mappe, applica i parametri e restituisce l'immagine. Crea un nuovo `torch.Generator` su CPU per ogni chiamata, inizializzato con il seed scelto, così lo stato casuale non dipende dalle chiamate precedenti. La riproducibilità va comunque verificata nello stesso ambiente software e hardware.

### 3.8 Runner e tracciamento

`main.py` cerca ricorsivamente le immagini nella cartella input e nelle sottocartelle. Per ogni fotografia prepara input e mappe, poi itera su condizioni, stili e seed. Consente di selezionare una sola condizione da CLI e di usare un prompt personalizzato con stile `custom`.

La struttura attesa dal runner corrente è:

```text
outputs/<timestamp_UTC>/
  run_config.json
  samples/<percorso_relativo_della_foto_con_estensione>/
    run_config.json
    input_preprocessed.png              # oppure input_original.png
    maps/canny/<nome>.png
    maps/segmentation/<nome>.png
    maps/segmentation/<nome>_labels.png
    maps/depth/<nome>.png
    maps/depth/<nome>_depth.npy
    images/<stile>/<condizione>/<nome>_seed42.png
    metadata.jsonl
```

Le mappe sono prodotte solo per le condizioni richieste. Le modalità di sola preparazione terminano prima della generazione e non producono il relativo `metadata.jsonl`.

I record di generazione includono identità e percorso input, stile, condizione, seed, prompt, modello, nome scheduler, parametri, riferimenti alle mappe, percorso output, timestamp, durata e stato. Gli errori dentro la chiamata di generazione/salvataggio vengono registrati e rilanciati: **il batch si interrompe**, non passa automaticamente al campione successivo. Errori precedenti, come il caricamento di un modello, non hanno la stessa copertura nel JSONL.

Le pipeline vengono liberate tra condizioni. Tuttavia, essendo il ciclo esterno sulle fotografie, i modelli vengono ricaricati per ogni foto: è semplice da seguire, ma può diventare costoso su un benchmark grande. Le mappe sono riutilizzate tra stili e seed della stessa foto nello stesso run, non ricaricate automaticamente da una cache persistente tra run.

## 4. Configurazione effettiva al 29 ottobre 2026

Questi sono **valori correnti, non parametri dimostrati ottimali**:

| Voce | Valore |
|---|---|
| Backbone | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Device / dtype | `cuda` / `float16` |
| Dimensioni con preprocessing | `512 × 512` |
| Inference steps | `50` |
| Guidance scale | `7.5` |
| Img2img strength | `0.5` |
| ControlNet conditioning scale | `0.8` |
| Soglie Canny | `100`, `200` |
| Segmentatore | `openmmlab/upernet-convnext-base` |
| Stimatore depth | `Intel/dpt-hybrid-midas` |
| ControlNet Canny | `lllyasviel/control_v11p_sd15_canny` |
| ControlNet Depth | `lllyasviel/control_v11f1p_sd15_depth` |
| ControlNet Segmentation | `lllyasviel/control_v11p_sd15_seg` |
| Stili | `gta5`, `pixel_art`, `watercolor` |
| Condizioni | `none`, `canny`, `depth`, `segmentation` |
| Seed | `[42]` |
| Input configurato | `data/test_images` — assente localmente |
| Output configurato | `outputs/` |
| Nome esperimento | `rq1_none_vs_canny` — non descrive più tutte le condizioni |

I prompt correnti descrivono rispettivamente resa di gioco GTA V, pixel art a 16 bit e acquerello rappresentativo. Sono definiti per stile e riutilizzati senza modificarli tra condizioni: questo è il comportamento da mantenere nel confronto principale.

## 5. Dati, prove locali e stato delle milestone

### Dati presenti

L'inventario locale aggiornato di `data/raw` contiene **108 file immagine: 97 PNG e 11 JPG**, nella radice, senza le precedenti sottocartelle di categoria. Il precedente inventario di 86 foto per categorie è stato sostituito dai dati caricati dall'utente. È un conteggio dei file, non una certificazione di unicità, validità, licenza o rappresentatività; per sapere quali foto sta elaborando Kaggle farà fede il suo `run_config.json`.

Non è presente un manifest che identifichi queste immagini come test set definitivo o le separi dalla validazione. `data/processed` contiene soltanto `.gitkeep`; anche `src/test_images` contiene soltanto `.gitkeep`. La roadmap proponeva 100 fotografie, ma tale benchmark congelato non è attestato dalla copia locale.

Le configurazioni watercolor/LoRA riguardano il lavoro preparatorio per RQ2 e non dimostrano che il runner RQ1 utilizzi adattatori stilistici. La precedente directory locale `data/watercolor` non è presente nell'inventario aggiornato.

### Artefatti presenti

In `outputs/` risultano sei directory di run del **15 ottobre 2026**, con fotografie preprocessate e `run_config.json`; quattro contengono anche una mappa Canny. Non risultano immagini generate nelle sottocartelle `images/`, JSONL di generazione o metriche.

Le configurazioni storiche differiscono da quelle correnti: per esempio usavano anche Cyberpunk, 30 step e altri valori di strength/soglie. Non vanno interpretate come output del protocollo attuale. La mancanza di risultati locali non esclude esecuzioni su Kaggle o altre macchine: tali prove vanno recuperate e associate alle rispettive configurazioni prima di usarle nel report.

| Area | Stato verificabile |
|---|---|
| Architettura modulare e TOML | Implementati |
| Baseline img2img | Implementata; validazione end-to-end da documentare |
| Canny | Implementato; mappe locali presenti |
| Depth e Segmentation | Implementati; nessun relativo artefatto locale trovato |
| Runner 4 condizioni × 3 stili | Implementato; run Kaggle in corso riferito dall'utente, senza preprocessing; configurazione effettiva da acquisire |
| Tracciamento per generazione | Implementato; nessun record locale trovato |
| Dataset test e validation congelati | Non documentati con manifest |
| Evaluation RQ1 | Implementata in `content.py`, `style.py`, `dino.py`, `records.py`, `evaluate.py`; protocollo in `configs/evaluation.toml`, guida in `EVALUATION_RQ1.md` |
| Verifica evaluation | Test offline di formule, audit, aggregazione e forward con piccoli modelli casuali; pesi preaddestrati ancora da verificare |
| Evaluation RQ2 | `lora_eval.py` rimane vuoto; fuori dall'attuale lavoro |
| Metriche e conclusioni RQ1 | Non disponibili localmente |
| RQ2 | File preparatori presenti; non necessaria per completare RQ1 |

**Non è ancora giustificato affermare quale conditioning preservi meglio contenuto o stile.**

## 6. Disallineamenti e limiti da tenere presenti

1. **Documentazione storica:** il README descrive una CLI precedente (`--dry-run`, `--style`, `--input-dir`, ecc.) e cita `src/experiment.py`, assente. Per i comandi attuali fanno fede `main.py` e il suo help.
2. **Definizione di baseline:** una roadmap chiama baseline il prototipo semantico; nel motore img2img corrente la baseline è `none`. Specificare sempre quale percorso si intende.
3. **Stili:** il riferimento storico contiene Cyberpunk; i TOML correnti contengono Watercolor. Confermare la scelta finale prima del benchmark e aggiornare i documenti.
4. **Dipendenze:** Canny importa `cv2`, ma `requirements.txt` non include `opencv-python`. Il loader usa direttamente `tomllib`, quindi richiede Python 3.11+ nello stato attuale, pur essendo elencato `tomli` per versioni precedenti.
5. **Riproducibilità:** le versioni delle librerie non sono tutte fissate; il loader di generazione non passa revisioni immutabili ai checkpoint. I metadati correnti non includono hash input, commit Git o inventario delle versioni installate.
6. **Scheduler:** viene registrato il nome della classe, non l'intera configurazione. Non attribuire al motore corrente l'UniPC usato dal prototipo.
7. **Robustezza e costo:** non esistono ripresa automatica, riuso persistente delle mappe o gestione completa degli errori di setup. Misurare il costo su poche immagini prima di scalare.
8. **Dati e versionamento:** controllare provenienza, duplicati e split; `.gitignore` ignora globalmente i PNG e `outputs/`. Gli artefatti pesanti devono essere conservati separatamente, con manifest e riferimenti sufficienti a ritrovarli.

## 7. Prossimi passi, in ordine operativo

### Passo 1 — Rendere coerente il punto di partenza

- [ ] Confermare gli stili finali: la proposta coerente con i TOML attuali è GTA V, Pixel Art, Watercolor.
- [ ] Impostare un input esistente in `configs/main.toml`; per il primo test usare una cartella dedicata con una sola fotografia, senza spostare gli originali.
- [ ] Rinominare l'esperimento, per esempio `rq1_structural_conditioning`.
- [ ] Verificare Python 3.11+, dipendenze effettive e disponibilità CUDA nell'ambiente che eseguirà i test.
- [ ] Allineare README e riferimenti alla CLI corrente, distinguendo prototipo e motore img2img.

**Completato quando:** configurazione, input e ambiente sono coerenti e i comandi disponibili sono documentati correttamente.

### Passo 2 — Verificare tutte le mappe senza generazione

- [ ] Eseguire Canny, Depth e Segmentation sulla stessa fotografia preparata.
- [ ] Controllare dimensioni, allineamento, contorni, palette, ID semantici e orientamento della profondità.
- [ ] Ripetere su poche scene diverse, includendo paesaggi e scene urbane.
- [ ] Conservare input, mappe, label, profondità float e configurazioni.

**Completato quando:** ogni preprocessore produce artefatti validi e interpretabili e non perde l'allineamento con la foto usata dal generatore.

### Passo 3 — Smoke test end-to-end e pilot stilistico

- [ ] Generare su una foto le quattro condizioni per i tre stili: **12 output**, seed 42 e parametri correnti come punto di partenza.
- [ ] Verificare immagini salvate, metadati `success`, assenza di output anomali e coerenza del confronto.
- [ ] Ripetere una configurazione nello stesso ambiente per verificare la riproducibilità osservata.
- [ ] Misurare durata totale e memoria disponibile prima di aumentare il numero di immagini.
- [ ] Estendere a un piccolo validation set separato dal futuro test, per verificare riconoscibilità dello stile e qualità del controllo.

La priorità è validare il percorso **prompt-only già implementato**. Se uno stile è poco riconoscibile, rivedere il prompt sul validation set prima di congelarlo. Eventuali LoRA fisse introdurrebbero un cambiamento di protocollo da documentare e applicare ugualmente a tutte le condizioni di quello stile; non sono un prerequisito attuale.

**Completato quando:** tutte le combinazioni funzionano e i tre target stilistici sono sufficientemente distinguibili per rendere sensata l'evaluation.

### Passo 4 — Congelare dataset e protocollo

- [ ] Verificare provenienza, utilizzo consentito, integrità e duplicati delle fotografie.
- [ ] Separare validation/pilot e test, mantenendo le categorie rilevanti.
- [ ] Salvare manifest con ID univoco, percorso relativo, categoria, split e hash.
- [ ] Fissare stili, prompt, preprocessing, checkpoint/revisioni, scheduler/config, precisione, step, CFG, strength, control scale e seed.
- [ ] Conservare config, commit e ambiente associati al protocollo finale.

Un primo seed è sufficiente per il debug; per lo studio valutare più seed compatibilmente con il budget. Il costo è `N immagini × 4 condizioni × 3 stili × S seed`: ad esempio **50 immagini × 12 × 3 = 1.800 output**, come scenario di pianificazione, non esecuzione già avvenuta.

**Completato quando:** si può elencare esattamente cosa generare senza prendere nuove decisioni guardando i risultati del test.

### Passo 5 — Implementare l'evaluation separata

**Codice implementato il 29 ottobre 2026; esecuzione sugli output reali ancora da fare.** `src/evaluation/content.py`, `style.py`, `dino.py`, `records.py` ed `evaluate.py` leggono il run esistente senza rigenerare immagini. Il protocollo è in `configs/evaluation.toml`:

| Dimensione | Misura proposta | Cautela interpretativa |
|---|---|---|
| Contenuto globale | CLIP-I, similarità foto/output | Non garantisce identità degli oggetti o geometria |
| Similarità visiva complementare | DINO-I (`dino_i`), coseno dei token CLS DINOv2 | Sensibile anche ai cambiamenti stilistici; non misura direttamente geometria o stile |
| Allineamento stilistico | CLIP-T con descrizioni di stile fisse | È un proxy; può riflettere anche il contenuto |
| Layout semantico | mIoU tra segmentazioni di input/output | È consistenza fra predizioni, non accuratezza rispetto a ground truth |
| Verifica percettiva | Piccola valutazione umana cieca | Separare stile, contenuto e artefatti |

Decisioni implementate: CLIP `openai/clip-vit-base-patch32`, coseni non scalati; descrizioni stilistiche brevi e fisse distinte dai prompt di generazione; mIoU con `nvidia/segformer-b2-finetuned-ade-512-512`, diverso dall'UPerNet del conditioning, sulle 150 classi ADE20K. I logits di input e output sono interpolati prima dell'argmax su una griglia comune con rapporto d'aspetto dell'input, lato lungo massimo 512. Classi assenti in entrambe escluse, presenti in una sola penalizzate con zero IoU. La classe 0 non è un background da escludere automaticamente. Restano bias ADE20K e domain shift stilistico.

Edge F1 con tolleranza e correlazione/ranking depth sono estensioni utili, successive alle metriche principali. Anche queste possono dipendere dal preprocessore e penalizzare trasformazioni stilistiche intenzionali.

- [x] Implementare CSV per singolo output con ID, categoria, stile, condizione, seed, dimensioni, hash e stato.
- [x] Implementare riepiloghi per stile/condizione/categoria, dispersione e conteggi.
- [x] Usare blocchi foto/stile/seed completi; mediare prima i seed per foto, poi le foto, e calcolare differenze appaiate rispetto a `none`.
- [x] Implementare intervalli bootstrap per input (2.000 ricampionamenti, seed 2026); con un solo input non stimarli.
- [x] Configurare checkpoint e testi di evaluation separati dalla generazione.
- [ ] Congelare revisioni immutabili dei valutatori e confermare definitivamente i testi prima del confronto finale (inizialmente revision `main`).
- [ ] Verificare caricamento dei checkpoint preaddestrati e produrre i primi CSV reali.

**Completato quando:** una piccola cartella di output produce CSV per immagine e riepiloghi verificabili senza nuova generazione.

La modalità `--audit-only` controlla la completezza senza caricare i modelli. I percorsi assoluti Kaggle vengono ricostruiti dalla struttura locale del run; non serve modificare i JSON dopo il trasferimento. L'audit confronta manifest, parametri, record e geometrie; il default blocca il calcolo se mancano output. `--allow-partial` è esplicito e i riepiloghi escludono comunque i blocchi incompleti. Ogni invocazione crea una nuova evaluation; non esiste ripresa automatica. Vedere la guida per comandi e dettagli.

### Passo 6 — Benchmark definitivo, analisi e report

- [ ] Valutare se aggiungere ripresa dei run e riuso dei modelli/mappe prima di eseguire il batch grande.
- [ ] Eseguire la matrice completa congelata e registrare eventuali combinazioni mancanti.
- [ ] Calcolare le metriche e produrre griglie `input | none | canny | depth | segmentation` per ciascuno stile.
- [ ] Visualizzare fedeltà stilistica contro preservazione del contenuto, con risultati distinti per stile e categoria.
- [ ] Esaminare casi di successo e fallimento con criteri dichiarati, evitando di selezionare solo esempi favorevoli.
- [ ] Scrivere conclusioni limitate ai checkpoint, stili e dati effettivamente studiati.

Solo dopo il confronto principale, valutare un'ablation della `conditioning_scale` su validation e un'eventuale estensione al test secondo un protocollo già fissato. Lo studio del rank LoRA resta separato, nella RQ2.

## 8. Comandi attuali per riprendere

Eseguire dalla radice del repository **dopo aver corretto `data.input_dir` nei TOML**. Il runner processa tutte le immagini trovate ricorsivamente: per uno smoke test selezionare una cartella piccola. Ogni invocazione crea una nuova directory timestamp; non riprende automaticamente quella precedente.

```powershell
# Visualizzare la CLI effettiva (richiede gli import iniziali del programma).
.\.venv\Scripts\python.exe main.py --help

# Solo preparazione: nessun caricamento di Stable Diffusion.
.\.venv\Scripts\python.exe main.py --prepare-canny-only
.\.venv\Scripts\python.exe main.py --prepare-depth-only
.\.venv\Scripts\python.exe main.py --prepare-segmentation-only

# Una condizione; esegue tutti gli stili e seed configurati.
.\.venv\Scripts\python.exe main.py --none
.\.venv\Scripts\python.exe main.py --canny
.\.venv\Scripts\python.exe main.py --depth
.\.venv\Scripts\python.exe main.py --segmentation

# Intera matrice definita in configs/main.toml.
.\.venv\Scripts\python.exe main.py
```

Non usare i flag della vecchia CLI del README senza verificarli: il runner attuale non offre `--dry-run`, `--limit`, `--input-dir`, `--style` o `--seed`. Input, stili e seed si modificano nei TOML; `--prompt` e `--negative-prompt` servono per prove personalizzate.

## 9. Memoria da aggiornare dopo ogni sessione

### Sessione del 29 ottobre 2026 — Evaluation RQ1

- **Aggiornamento dati/run:** utente segnala caricamento foto in raw e generazione Kaggle in corso, tutti gli stili, senza preprocessing. Inventario locale attuale: 108 immagini. Nessun output Kaggle reale ancora valutato.
- **File implementati:** `src/evaluation/content.py` (mIoU e SegFormer), `style.py` (embedding CLIP), `records.py` (audit e trasferimento percorsi), `evaluate.py` (CLI, metriche, CSV e aggregazione); `configs/evaluation.toml`; test in `tests/test_evaluation.py` e `tests/test_evaluation_models.py`.
- **Documentazione:** creata `EVALUATION_RQ1.md`, aggiunto riferimento nel README e aggiornato questo stato senza modificare la generazione in corso.
- **Verifiche:** **14 test passati**: formule, aggregazione appaiata, run sintetico trasferito, audit CLI, geometrie incoerenti, intero flusso di evaluation/CSV con valutatori deterministici e forward di piccoli CLIP/SegFormer casuali senza download. Nessuna stima reale di qualità ottenuta da questi test. Comando: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- **Ambiente locale rilevato:** Python 3.11.9, Torch 2.14.1+cu126, Transformers 5.19.0, Diffusers 0.41.0, NumPy 2.4.6, Pillow 12.3.0; CUDA disponibile. Versioni diverse da parte dei vincoli storici di `requirements.txt`; nessuna installazione effettuata.
- **Limiti aperti:** checkpoint preaddestrati non caricati in questa sessione; revisioni `main` da congelare; nessuna cache/ripresa persistente dell'evaluation; grafici, griglie e valutazione umana da aggiungere dopo i primi risultati.
- **Prossima azione:** scaricare l'intera directory del run Kaggle, eseguire `python -m src.evaluation.evaluate --run-dir <RUN> --audit-only`, verificare matrice e geometrie, poi calcolare le metriche sul run. Se 108 input, tre stili, quattro condizioni e un seed sono confermati dal manifest, attesi 1.296 output.

### Aggiornamento del 29 ottobre 2026 — Integrazione DINO

- **Richiesta:** aggiungere un secondo valutatore del contenuto complementare a CLIP-I, mantenendo l'integrazione circoscritta.
- **Modello scelto:** DINOv2 ViT-B/14, checkpoint `facebook/dinov2-base`, revision iniziale `main`. Nessun training; parametri congelati e inferenza senza gradienti. La scelta attuale non usa DINOv3.
- **Definizione della metrica:** `dino_i` è il coseno non scalato fra i token CLS finali di originale e output (768 componenti per il checkpoint scelto). Non si mediano i patch token e non si eseguono confronti locali. Il processore del checkpoint applica il proprio resize/crop/normalizzazione a entrambe le immagini.
- **File modificati:** sezione `[dino]` e lista metriche in `configs/evaluation.toml`; nuovo `src/evaluation/dino.py`; collegamento in `evaluate.py`, aggiornamento dei due file di test e della guida. I commenti dell'utente nei file esistenti sono conservati.
- **Orchestrazione:** il modello viene caricato solo se richiesto; l'embedding originale viene riutilizzato tra output consecutivi della stessa foto. Punteggi individuali, riepiloghi, differenze da `none` e metadati dei valutatori includono DINO. Si può eseguire soltanto DINO con `--metrics dino_i`, senza caricare CLIP o SegFormer.
- **Verifiche:** **16 test offline passati**, inclusi estrazione del CLS con un piccolo DINOv2 casuale, esecuzione di solo DINO e integrazione nei CSV/confronti appaiati con valutatori deterministici. Checkpoint DINOv2 preaddestrato non scaricato né eseguito in questa sessione; nessuna dipendenza aggiunta.
- **Interpretazione:** DINO-I confronta rappresentazioni visive; serve a verificare se i risultati sul contenuto concordano con CLIP-I. Non è una misura pura del contenuto né una percentuale, e i valori assoluti non sono calibrati rispetto a CLIP-I. Rimangono da verificare il caricamento dei pesi reali e il costo GPU.
- **Prossima azione:** audit del run Kaggle e primo calcolo delle quattro metriche; congelare le revisioni dei modelli prima del confronto definitivo.

Per mantenere utile questo documento, aggiungere una voce solo quando esiste un cambiamento o una prova concreta:

```text
Data:
Obiettivo della sessione:
File modificati / commit:
Config e ambiente usati:
Dataset / split / manifest:
Run e percorso artefatti:
Verifiche eseguite e loro esito:
Risultati osservati (con riferimenti, separati dalle interpretazioni):
Problemi aperti:
Prossima azione concreta:
```

### Prompt per una nuova sessione

> Leggi `STATO_PROGETTO_RQ1.md` e `EVALUATION_RQ1.md`, verificando cambiamenti del codice rispetto allo snapshot. Prosegui con RQ1: SD 1.5 img2img, condizioni `none`, `canny`, `depth`, `segmentation`; stili GTA V, Pixel Art, Watercolor; nessuna LoRA nel loader corrente. L'utente ha avviato Kaggle per tutti gli stili senza preprocessing. L'evaluation ora include audit, CLIP-I, CLIP-T, mIoU con SegFormer diverso dall'UPerNet del conditioning e DINO-I (`dino_i`, coseno dei CLS di `facebook/dinov2-base`), CSV e confronti appaiati con bootstrap per input. I 16 test offline passano, ma mancano caricamento dei pesi preaddestrati e metriche reali sugli output Kaggle. Localmente raw contiene 108 immagini; la matrice effettiva farà fede sul manifest Kaggle. Acquisisci il run, esegui prima l'audit e poi l'evaluation, conservando configurazioni e aggiornando questa memoria. Procedi per piccoli passi spiegabili, senza introdurre altre metriche o file non necessari. Non presentare test sintetici come risultati scientifici e mantieni RQ2 separata.
