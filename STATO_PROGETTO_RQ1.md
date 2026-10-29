# MultiStyleImages — Memoria del progetto e prossimi passi RQ1

> **Aggiornato al:** 29 ottobre 2026.  
> **Scopo:** conservare il contesto necessario per riprendere il progetto, distinguendo implementazione, prove disponibili e lavoro ancora da svolgere.  
> **Snapshot del repository:** branch `conditioning-ablation`, HEAD `09948b4`; stesso commit registrato nell'evaluation reale su Kaggle.
> **Fonti:** codice, configurazioni, cronologia Git, aggiornamenti dell'utente e risultati Kaggle scaricati in `evaluation_firstTry/`. Prima evaluation reale completata su `olanda.png`; analisi dei file effettuata localmente, senza rieseguire le metriche.

## 1. Da dove ripartiamo

Il progetto dispone di una pipeline modulare per confrontare **Stable Diffusion 1.5 img2img senza controllo strutturale** con tre varianti **ControlNet img2img: Canny, Depth e Segmentation**. Il runner gestisce più fotografie, stili e seed, salva gli input effettivi, le mappe, le configurazioni e i metadati delle generazioni.

La configurazione attuale seleziona **GTA V, Pixel Art e Watercolor**. La strategia stilistica implementata è basata sui prompt: il loader corrente non carica LoRA.

**La parte tecnica ha superato una prima prova end-to-end reale.** Su Kaggle sono state generate le 12 combinazioni di `olanda.png` (tre stili, quattro condizioni, seed 42, senza preprocessing fotografico esplicito) e calcolate tutte e quattro le metriche con i checkpoint preaddestrati. Audit: 12 output pronti, 0 non validi; evaluation: 12 successi e 3 blocchi completi. Gli artefatti sono in [evaluation_firstTry/20261029T134635_359265Z](evaluation_firstTry/20261029T134635_359265Z/). Completata anche l'ispezione visiva esplorativa dell'originale e di tutti i 12 output, distinta da una valutazione umana cieca. La RQ1 non è ancora conclusa scientificamente: riconoscibilità degli stili da consolidare, benchmark completo, analisi aggregata e report ancora da completare. La guida tecnica è [EVALUATION_RQ1.md](EVALUATION_RQ1.md).

**Prossima azione concreta, concordata a fine giornata:** recuperare/verificare il run sulle **108 foto**, eseguire prima l'audit e poi l'evaluation con le quattro metriche, quindi iniziare tabelle, analisi e scrittura della parte RQ1. Non cambiare ora prompt, parametri o metriche e non rigenerare automaticamente il batch: l'utente riferisce circa 15 ore per un seed e il completamento del run grande non è ancora attestato dai file locali. La scarsa riconoscibilità di Pixel Art e i limiti stilistici osservati nel pilot devono essere dichiarati e verificati su altri esempi, senza occultarli. Dopo RQ1, passare a RQ2 con una LoRA Watercolor e confronto dei rank. `configs/main.toml` locale punta a `data/prova_eval`: è l'input della prova singola, non del benchmark completo. Il valutatore legge il manifest del run indicato in `--run-dir`, non questo percorso TOML.

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

Le soglie sono configurabili. Esiste la modalità `--prepare-canny-only`, che salva input e mappa senza caricare Stable Diffusion. Oltre alle mappe storiche in `outputs/`, la prova Kaggle del 29 ottobre attesta il completamento del ramo Canny su una foto; ispezione visiva della mappa e dell'output ancora da documentare.

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
| Input configurato | `data/prova_eval` — prova su una fotografia; tornare a `data/raw` per il dataset completo |
| Output configurato | `outputs/` |
| Nome esperimento | `rq1_none_vs_canny` — non descrive più tutte le condizioni |

I prompt correnti descrivono rispettivamente resa di gioco GTA V, pixel art a 16 bit e acquerello rappresentativo. Sono definiti per stile e riutilizzati senza modificarli tra condizioni: questo è il comportamento da mantenere nel confronto principale.

## 5. Dati, prove locali e stato delle milestone

### Dati presenti

L'inventario locale aggiornato di `data/raw` contiene **108 file immagine: 97 PNG e 11 JPG**, nella radice, senza le precedenti sottocartelle di categoria. Il precedente inventario di 86 foto per categorie è stato sostituito dai dati caricati dall'utente. È un conteggio dei file, non una certificazione di unicità, validità, licenza o rappresentatività; per sapere quali foto sta elaborando Kaggle farà fede il suo `run_config.json`.

Non è presente un manifest che identifichi queste immagini come test set definitivo o le separi dalla validazione. `data/processed` contiene soltanto `.gitkeep`; anche `src/test_images` contiene soltanto `.gitkeep`. La roadmap proponeva 100 fotografie, ma tale benchmark congelato non è attestato dalla copia locale.

Le configurazioni watercolor/LoRA riguardano il lavoro preparatorio per RQ2 e non dimostrano che il runner RQ1 utilizzi adattatori stilistici. La precedente directory locale `data/watercolor` non è presente nell'inventario aggiornato.

### Artefatti presenti

Nello snapshot iniziale, `outputs/` conteneva soltanto prove del 15 ottobre, senza metriche finali. **Aggiornamento:** in `evaluation_firstTry/` sono ora disponibili gli artefatti della prova reale Kaggle su `olanda.png`: cartella `20261029T134520_300187Z` con il solo audit e cartella `20261029T134635_359265Z` con audit e evaluation completa. Il run di generazione associato è `20261029T133513_530561Z`. I CSV attestano 12 output valutati; le immagini originali/generate non sono contenute nei CSV e la loro verifica visiva resta un'attività separata.

Le configurazioni storiche differiscono da quelle correnti: per esempio usavano anche Cyberpunk, 30 step e altri valori di strength/soglie. Non vanno interpretate come output del protocollo attuale. La mancanza di risultati locali non esclude esecuzioni su Kaggle o altre macchine: tali prove vanno recuperate e associate alle rispettive configurazioni prima di usarle nel report.

| Area | Stato verificabile |
|---|---|
| Architettura modulare e TOML | Implementati |
| Baseline img2img | Eseguita e valutata su `olanda.png`, nei tre stili |
| Canny, Depth e Segmentation | Eseguiti e valutati sulla stessa foto; verifica visiva ancora da documentare |
| Runner 4 condizioni × 3 stili | Prova reale completata: 12 output, seed 42, senza preprocessing fotografico esplicito |
| Tracciamento per generazione | Audit reale completo: tutti i 12 record richiesti superano i controlli |
| Dataset test e validation congelati | Non documentati con manifest |
| Evaluation RQ1 | Implementata in `content.py`, `style.py`, `dino.py`, `records.py`, `evaluate.py`; protocollo in `configs/evaluation.toml`, guida in `EVALUATION_RQ1.md` |
| Verifica evaluation | 16 test offline documentati; quattro metriche reali riuscite sui 12 output Kaggle |
| Evaluation RQ2 | `lora_eval.py` rimane vuoto; fuori dall'attuale lavoro |
| Metriche e conclusioni RQ1 | CSV reali disponibili per una foto; nessuna conclusione generale ancora giustificata |
| RQ2 | File preparatori presenti; non necessaria per completare RQ1 |

**Si possono descrivere i risultati della singola foto; non è ancora giustificato affermare quale conditioning sia generalmente migliore.**

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
- [x] Impostare l'input di prova: `data/prova_eval` su Kaggle, con `olanda.png`; configurazione acquisita nel manifest reale.
- [ ] Rinominare l'esperimento, per esempio `rq1_structural_conditioning`.
- [x] Verificare operativamente l'ambiente: generazione ed evaluation reale su CUDA completate; versioni dei valutatori registrate in `evaluation_config.json`.
- [ ] Allineare README e riferimenti alla CLI corrente, distinguendo prototipo e motore img2img.

**Completato quando:** configurazione, input e ambiente sono coerenti e i comandi disponibili sono documentati correttamente.

### Passo 2 — Verificare tutte le mappe senza generazione

- [ ] Eseguire Canny, Depth e Segmentation sulla stessa fotografia preparata.
- [ ] Controllare dimensioni, allineamento, contorni, palette, ID semantici e orientamento della profondità.
- [ ] Ripetere su poche scene diverse, includendo paesaggi e scene urbane.
- [ ] Conservare input, mappe, label, profondità float e configurazioni.

**Completato quando:** ogni preprocessore produce artefatti validi e interpretabili e non perde l'allineamento con la foto usata dal generatore.

### Passo 3 — Smoke test end-to-end e pilot stilistico

- [x] Generare su una foto le quattro condizioni per i tre stili: **12 output**, seed 42, run `20261029T133513_530561Z`, senza preprocessing fotografico esplicito.
- [x] Verificare presenza/leggibilità, metadati e coerenza tecnica: audit 12 pronti, 0 non validi.
- [ ] Verificare visivamente originali/output e mappe: riconoscibilità dello stile, oggetti, posizione e artefatti. L'audit non sostituisce questo controllo.
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

**Pianificazione discussa con l'utente:** per 108 fotografie e seed 42 sono attesi 1.296 output; l'utente riferisce circa 15 ore di generazione. Tre seed sull'intero dataset porterebbero a 3.888 output e circa 45 ore, a tempi proporzionali. Proposta ancora da confermare: confronto principale su 108 foto con un seed e controllo di stabilità su un sottoinsieme di 15–20 foto scelto prima dell'analisi, con altri due seed. Non sono esperimenti già eseguiti. Run separati richiedono un'unione verificata: l'orchestratore attuale valuta un run alla volta.

**Categorie:** i dati non sono organizzati in categorie equilibrate; l'utente stima circa tre foto di animali. Mantenere come analisi principale la media globale con uguale peso per foto (`category=__all__`), senza pesi aggiuntivi per categoria. Eventuali categorie servono a un'analisi esplorativa, riportando sempre le numerosità. Non spostare immagini o modificare gli ID dei run già generati. La composizione sbilanciata limita la generalizzazione e va dichiarata.

**Completato quando:** si può elencare esattamente cosa generare senza prendere nuove decisioni guardando i risultati del test.

### Passo 5 — Implementare l'evaluation separata

**Codice implementato e prima esecuzione reale completata il 29 ottobre 2026.** `src/evaluation/content.py`, `style.py`, `dino.py`, `records.py` ed `evaluate.py` leggono il run esistente senza rigenerare immagini. Su `olanda.png` tutte e quattro le metriche sono state calcolate per tutti i 12 output. Il protocollo è in `configs/evaluation.toml`:

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
- [x] Verificare caricamento dei checkpoint preaddestrati e produrre i primi CSV reali: eseguito su Kaggle, artefatti in `evaluation_firstTry/20261029T134635_359265Z/`.

**Milestone tecnica raggiunta:** una piccola cartella di output ha prodotto CSV per immagine e riepiloghi verificabili senza nuova generazione. Rimangono verifica visiva, valutazione del dataset completo e analisi scientifica.

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

### Aggiornamento del 29 ottobre 2026 — Prima prova reale e lettura dei risultati

Questa voce aggiorna e supera le indicazioni precedenti «checkpoint non caricati» e «nessun CSV reale»; le voci precedenti documentano lo stato al momento dell'implementazione.

- **Run generazione:** `20261029T133513_530561Z`; input `olanda.png`, 1217×694; output 1216×688, quattro condizioni × tre stili × seed 42. `--no-preprocess`, 50 step configurati, CFG 7,5, strength 0,5, ControlNet scale 0,8, scheduler PNDM.
- **Artefatti verificati:** `evaluation_firstTry/20261029T134635_359265Z/`, letti insieme all'utente file per file: audit, riepiloghi, punteggi individuali, differenze appaiate, configurazione, valutatori e manifest. Nessuna immagine è stata ispezionata visivamente durante questa lettura dei CSV.
- **Esito:** audit 12 attesi/pronti e 0 non validi; metriche 12 successi, 3 blocchi foto/stile/seed completi e 12 output nei riepiloghi. Sono state eseguite CLIP-I, CLIP-T, mIoU e DINO-I con pesi preaddestrati.
- **Ambiente evaluation Kaggle:** Torch 2.11.0+cu128, Transformers 5.16.1, Pillow 12.3.0, NumPy 2.1.3, huggingface-hub 1.29.0; CUDA; commit `09948b473e10971eefc0456d890f3bae8e9c3f3c`.
- **Revisioni risolte:** CLIP `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; SegFormer `de01bae28967510f9ddd496c60a969357195400c`; DINOv2 `f9e44c814b77203eaa57a6bdbbd535f21ede1415`. Sono documentate in `evaluators.json`; i TOML richiedono ancora `main`.
- **Osservazioni sulla sola foto:** Canny ha DINO-I massimo nei tre stili e anche CLIP-I e mIoU massimi in GTA V e Pixel Art; Watercolor favorisce Segmentation su mIoU e CLIP-I. CLIP-T massimo: Depth in GTA V, `none` in Pixel Art e Canny in Watercolor. In Watercolor, Canny supera `none` su tutte e quattro le metriche. Non sono conclusioni generalizzabili né una selezione definitiva del metodo.
- **Limiti interpretativi:** con una sola foto non si stimano deviazione standard fra input e bootstrap; campi vuoti corretti. I gruppi `__all__` e `unassigned` coincidono, senza duplicare gli input nella stessa media. CLIP usa crop centrale 224×224 dopo resize lato corto 224; DINO crop 224×224 dopo resize lato corto 256. SegFormer ridimensiona a 512×512 e i logits sono riportati sulla griglia comune 512×292. Questi preprocessori sono attivi anche se il preprocessing fotografico di generazione è disattivato.
- **Ordine delle prossime attività:** (1) confronto visivo della foto e dei 12 output, (2) conferma del protocollo e identificazione di pilot/test, (3) recupero/completamento e audit delle 1.296 generazioni del dataset se confermate, (4) evaluation completa, (5) grafici e griglie, analisi e report, con eventuale controllo su più seed compatibile con il budget. Nessun nuovo modulo o metrica necessario per il prossimo passo.

Per mantenere utile questo documento, aggiungere una voce solo quando esiste un cambiamento o una prova concreta:

**Confronto visivo GTA V, 29 ottobre:** ricevuto `generated_images(12).zip` ed estratto, senza modificare gli originali, in `outputs/confronto_olanda_20261029/`. Verificati gli hash dei 12 output rispetto al CSV: tutti corrispondono. Ispezionati soltanto originale e quattro output GTA V. Canny mantiene meglio ponte, tenda a strisce, fila di biciclette, ringhiera e disposizione dei tavoli; restano deformazioni di dettagli e persone. `none` e Depth alterano maggiormente biciclette, ringhiere, facciate e vegetazione; Segmentation mantiene il contesto del canale ma perde buona parte della ringhiera e altera i dettagli. L'ordine qualitativo di preservazione è compatibile con i maggiori CLIP-I/DINO-I/mIoU di Canny per GTA V. Tutti gli output presentano texture artificiali e distorsioni; il CLIP-T maggiore di Depth non dimostra da solo una resa GTA V visivamente migliore. Osservazioni esplorative dell'assistente su un solo input/seed, non valutazione umana cieca. Pixel Art e Watercolor ancora da ispezionare visivamente; nessun cambiamento a codice o parametri.

**Confronto visivo Pixel Art, 29 ottobre:** ispezionati originale e quattro output in `outputs/confronto_olanda_20261029/samples/olanda.png/images/pixel_art/`. Tutti mantengono una resa dettagliata vicina alla fotografia/illustrazione con texture granulari; non emerge chiaramente una griglia di pixel regolare, una palette limitata o una semplificazione coerente da videogioco 16-bit. Canny conserva meglio ponte, tenda centrale a strisce, ringhiera, biciclette e persone ai tavoli; ciò è coerente con i massimi CLIP-I (0,956), DINO-I (0,867) e mIoU (0,316). Negli altri rami diversi dettagli sono reinventati: ringhiera decorata, tronco sostituito da elemento simile a un lampione, biciclette e arredi alterati. CLIP-T massimo di `none` (0,200 contro 0,195 Canny) non certifica riuscita Pixel Art e la differenza è piccola, senza evidenza di stabilità. Un alto punteggio di contenuto può accompagnare una stilizzazione debole. Osservazione esplorativa dell'assistente su una sola foto/seed; finire Watercolor e discutere la validità stilistica del pilot prima di modificare o congelare parametri. Nessun cambiamento a codice, prompt o configurazioni.

**Confronto visivo Watercolor, 29 ottobre:** ispezionati originale e tutti e quattro gli output Watercolor. `none` presenta superfici e contorni pittorici ma reinventa vegetazione lungo il bordo del canale, biciclette e ringhiera; Depth conserva l'impianto generale ma mantiene diverse reinvenzioni, inclusa la tenda centrale senza strisce. Canny conserva meglio ponte, tenda a strisce, fila delle biciclette, ringhiera e persone ai tavoli, pur con distorsioni. Segmentation conserva le grandi regioni ma altera gli oggetti minuti e perde tratti della ringhiera. Watercolor appare più chiaramente pittorico del corrispondente Pixel Art, senza garantire una resa ad acquerello pienamente convincente o una separazione netta fra i tre target. I punteggi della singola foto: Canny massimo CLIP-T (0,271) e DINO-I (0,832), Segmentation massimo CLIP-I (0,938) e mIoU (0,264). Il massimo CLIP-I di Segmentation non implica migliore preservazione di ogni oggetto: l'ispezione locale favorisce Canny su diversi dettagli. Non possiamo attribuire le differenze di ranking a una causa certa dai soli punteggi aggregati. Risultati visivi e metriche sostengono Canny come candidato interessante nel pilot, senza eleggerlo vincitore generale. Ora tutti i 12 output sono stati ispezionati. Nessun cambiamento a codice, prompt o parametri; prossimo passo: discutere/validare riconoscibilità stilistica prima del protocollo definitivo.

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

> Leggi `STATO_PROGETTO_RQ1.md`, soprattutto la sezione «Chiusura del 29 ottobre 2026», e `EVALUATION_RQ1.md`. Abbiamo completato generazione/evaluation reale su `olanda.png`: 12 output, tre stili (GTA V, Pixel Art, Watercolor), quattro condizioni (`none`, `canny`, `depth`, `segmentation`), seed 42, senza preprocessing fotografico esplicito. Audit ed evaluation tutti riusciti; metriche CLIP-I, CLIP-T, mIoU SegFormer e DINO-I. CSV/JSON in `evaluation_firstTry/20261029T134635_359265Z/`; immagini in `outputs/confronto_olanda_20261029/`. Abbiamo letto tutti i file e ispezionato tutti i 12 output: Canny preserva diversi dettagli, ma la riuscita stilistica è limitata, soprattutto Pixel Art. Nessuna conclusione generale da una foto. Decisione finale: evaluation del run completo sulle 108 foto e scrittura della parte RQ1, poi RQ2 LoRA solo Watercolor, pilota rank 8 e confronto rank 4/8/16/32 a training controllato. Non avviare LoRA prima di chiudere l'analisi RQ1. Riparti verificando cartella, manifest e stato del run completo: 1.296 output attesi se 108 foto × 3 stili × 4 condizioni × 1 seed sono confermati. Non rigenerare automaticamente: costo riferito circa 15 ore. Usa media globale a pari peso per foto, senza pesi per categorie sbilanciate. Più seed su un sottoinsieme sono soltanto una proposta da confermare. Non cambiare codice o protocollo senza richiesta. L'utente preferisce piccoli passi spiegati uno alla volta; aggiorna questa memoria man mano.

### Chiusura del 29 ottobre 2026 — Punto di ripartenza

Questa è la decisione più recente e prevale sulle precedenti indicazioni operative nel diario.

**Completato e verificato:**

- Implementazione evaluation con CLIP-I, CLIP-T, mIoU SegFormer e DINO-I; ultima suite documentata: 16 test offline passati.
- Prova end-to-end Kaggle su una fotografia: tutti e 12 gli output generati, validi e valutati; tre blocchi completi.
- Lettura guidata di tutti i nove file della evaluation completa: audit e riepilogo, manifest, configurazione, valutatori, punteggi individuali, medie e differenze dalla baseline.
- Ispezione visiva esplorativa dell'originale e dei 12 output, stile per stile. Nessun nuovo parametro selezionato sulla base dei risultati; nessuna valutazione umana cieca ancora effettuata.
- Confermati i percorsi locali: `evaluation_firstTry/20261029T134635_359265Z/` per CSV/JSON; `outputs/confronto_olanda_20261029/` per run completo estratto. ZIP sorgente ricevuto: `C:/Users/Ladybug/Downloads/generated_images(12).zip`. Gli hash dei 12 output estratti corrispondono al CSV.

**Decisioni più recenti:**

1. Completare l'evaluation sulle 108 foto e scrivere la parte RQ1 sul confronto fra conditioning, dichiarando il limite di fedeltà stilistica della strategia prompt-only. Le conclusioni dovranno dipendere dai dati completi, non dalla sola foto del pilot.
2. Non aggiungere altre metriche o modificare ora la pipeline. Usare analisi globale con uguale peso per fotografia; categorie eventualmente esplorative, senza pesi aggiuntivi (circa tre esempi animali secondo l'utente).
3. Dopo l'analisi RQ1, RQ2 su un solo stile, **Watercolor**: verificare prima un pilota LoRA rank 8, poi confronto no-LoRA e rank 4/8/16/32 su stesso dataset/split e conditioning fisso. Dataset finale, budget di training e conditioning RQ2 ancora da fissare. Non variare contemporaneamente rank e strategia di training nel confronto principale; variazione della durata di training eventualmente successiva. La LoRA è un'ipotesi da verificare per la fedeltà stilistica, non una garanzia di qualità estetica.
4. Un seed per il confronto principale è il piano attuale; tre seed sull'intero dataset non sono stati decisi. Proposta futura: 15–20 foto selezionate con criteri espliciti e altri due seed per stabilità, compatibilmente con il budget.

**Da fare all'apertura della prossima sessione, una cosa alla volta:**

1. Identificare la directory del run completo su Kaggle e verificarne il `run_config.json`: non confonderlo con la prova `20261029T133513_530561Z`. Il completamento delle 108 foto non è ancora verificato qui.
2. Eseguire `python -m src.evaluation.evaluate --run-dir "outputs/<RUN_COMPLETO>" --audit-only`. Attesi 1.296 output e 0 non validi se il manifest conferma 108 input, tre stili, quattro condizioni e seed 42. Se incompleto, capire cosa manca prima di lanciare nuove generazioni.
3. Se audit completo, eseguire `python -m src.evaluation.evaluate --run-dir "outputs/<RUN_COMPLETO>" --device cuda`. Ogni invocazione crea una cartella timestamp di evaluation; non serve spostare le immagini, serve indicare il run giusto.
4. Conservare l'intero run e i risultati; leggere prima conteggi/errori, poi tabella globale `category=__all__`, confronti appaiati e campione visivo vario. Se `olanda.png` è inclusa nel benchmark, dichiarare che è già stata usata come pilot/ispezione e mantenere trasparente l'uso dei dati.
5. Preparare tabelle/grafici e testo RQ1 solo dopo i risultati completi; discutere limiti, un solo seed, composizione del dataset e riuscita stilistica. Nessuna scrittura del paper completata finora.

**Preferenza di lavoro:** l'utente chiede spiegazioni graduali e un file/attività alla volta; evitare nuove strutture e modifiche multiple non necessarie. Questa chiusura aggiorna soltanto la documentazione, senza eseguire generazioni, metriche o training.
