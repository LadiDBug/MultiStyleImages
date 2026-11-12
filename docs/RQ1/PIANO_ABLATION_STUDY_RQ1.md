# RQ1 — Piano di ablation study e registro degli esperimenti

**Aggiornato al 12 novembre 2026.** Documento operativo da aggiornare dopo ogni evaluation e decisione. Le caselle indicano attività effettivamente concluse; una generazione presente non equivale a un risultato analizzato.

L'obiettivo è rispondere a questa domanda: **a parità di foto, stile e seed, quanto cambia il compromesso fra contenuto e stile aggiungendo Canny, depth o segmentation a SD 1.5 img2img?** Prima analizziamo i risultati già prodotti in [`result_rq1/`](../../result_rq1/); nuove generazioni servono solo a colmare lacune o verificare un'ipotesi precisa emersa dall'analisi.

Il protocollo tecnico delle metriche e dei file è in [EVALUATION_RQ1.md](EVALUATION_RQ1.md). Qui teniamo l'ordine del lavoro, i confronti da completare, le decisioni e le evidenze. **Aggiornamento dopo l'evaluation Kaggle:** l'analisi quantitativa A01–A05 è completata in [ANALISI_RISULTATI_RQ1_COMMON104.md](ANALISI_RISULTATI_RQ1_COMMON104.md); restano ispezione visiva, analisi per scena e giudizi umani. I conteggi dell'inventario iniziale più avanti descrivono lo stato prima dell'evaluation.

**Griglie e modulo disponibili, preparazione autorizzata:** [galleria locale](../../result_evaluation_rq1/visual_review_common104_v1/index.html) e [criteri/istruzioni](GRIGLIE_ISPEZIONE_RQ1_COMMON104.md). Sono state create 18 griglie casuali e 12 diagnostiche, tutte con seed 42 e provenienza verificata. L'utente ha scelto di annotare tutte le 30 griglie nell'HTML: quattro domande standard, salvataggio locale, revisori distinti, divisione A/B ed esportazione/importazione JSON. Nessun giudizio viene assegnato automaticamente; i dati compilati andranno raccolti dai revisori.

## Decisioni operative concordate

- **Analisi principale sulle stesse 104 foto complete**, per tutti gli stili, condizioni e seed: 104 × 3 × 4 × 3 = **3.744 output da valutare**. Non usare le 105 disponibili in alcuni stili per la tabella principale.
- **Nessuna nuova generazione per recuperare le foto mancanti.** Escludere `0600x2.png`, `lsdir_0149.png`, `lsdir_0405.png` e `lsdir_0411.png` dall'analisi principale. Conservare gli artefatti originali e dichiarare nel report che il sottoinsieme comune deriva dalla completezza degli output, non dai punteggi ottenuti.
- **Non usare la GPU locale.** Le inferenze di evaluation ed eventuali generazioni si eseguono su Kaggle; nessuna di queste esecuzioni è autorizzata automaticamente.
- Procedere un passo alla volta. Prima di ogni passaggio operativo successivo, descrivere attività e risultato atteso e **attendere l'OK dell'utente**. Quando serve Kaggle, segnalare esplicitamente la necessità del notebook prima di preparare o avviare quel passaggio.
- Preparazione, lancio Kaggle e analisi aggregata autorizzati dall'utente e completati. **3.744 output valutati**, tre seed completi, dati verificati e risultati aggregati in `result_evaluation_rq1/analysis_common104_v1/`. Nessuna inferenza locale eseguita; il prossimo passo visivo resta da autorizzare separatamente.

L'audit sui 108 input resta la provenienza del dataset. Il benchmark analizzato sarà quello sulle 104 foto comuni; eventuali errori di metrica vanno documentati e discussi prima di cambiare ulteriormente il sottoinsieme.

## 1. Stato di partenza: che cosa abbiamo davvero

Inventario ricavato dai manifest e dai `metadata.jsonl` locali, seguito da **audit effettivo tramite `audit_run` sui cinque run**, senza inferenza. I cinque manifest dichiarano gli stessi parametri di modello, generazione, prompt, preprocessori e ControlNet; cambiano seed e liste di input. Tutti i record trovati dichiarano `success` e `PNDMScheduler`. Non risultano CSV di evaluation in `result_rq1/` alla data di questo documento.

| ID artefatto | Directory sotto `result_rq1/` | Seed | Foto nel manifest | Record di successo / output attesi | Stato iniziale |
|---|---|---:|---:|---:|---|
| G42-A | `baseline_seed_42/test_1` | 42 | 60 | 720 / 720 | Audit: 720 validi; da valutare |
| G42-B | `baseline_seed_42/test_2` | 42 | 48 | 576 / 576 | Audit: 576 validi; da valutare |
| G123 | `baseline_seed_123` | 123 | 108 | 1.281 / 1.296 | Audit: 1.280 validi, 16 non validi/mancanti |
| G456-A | `baseline_seed_456/test1` | 456 | 108 | 1.083 / 1.296 | Audit: 1.083 validi; continuazione in G456-B |
| G456-B | `baseline_seed_456/test2` | 456 | 17 | 204 / 204 | Audit: 204 validi; da valutare |

**Conteggio corretto del benchmark:** 108 foto × 3 stili × 4 condizioni × 3 seed = **3.888 output attesi**. Le unioni dei record per seed contengono 1.296 output per 42, 1.281 per 123 e 1.287 per 456, per un totale di **3.864 combinazioni uniche registrate**. Mancano 24 record; inoltre un record punta a un PNG assente. L'audit certifica quindi **3.863 output validi e 25 combinazioni non utilizzabili** nella copia locale, dopo l'unione logica degli shard. Gli hash SHA-256 degli `input_original.png` presenti non mostrano conflitti per lo stesso `image_id` tra shard e seed.

Punti da risolvere prima di dichiarare il benchmark completo:

- Seed 42: i manifest dei due shard sono disgiunti e insieme coprono 108 foto. L'elemento aggiuntivo in `test_1/samples/` è il file di sistema `.DS_Store`, non una foto o un esperimento: non entra nei conteggi.
- Seed 123: `lsdir_0411.png` è completamente assente dai record; `lsdir_0405.png` manca di segmentation nei tre stili. Inoltre manca il PNG `0600x2.png / pixel_art / depth`, pur essendoci il record di successo.
- Seed 456: `test1` dichiara tutte le 108 foto ma registra output per 91; `test2` dichiara e registra le 17 successive. Le 17 foto sovrapposte fra i manifest non hanno record duplicati fra gli shard. Nell'unione restano 9 combinazioni mancanti: `lsdir_0149.png`, tutti e tre gli stili, condizioni canny/depth/segmentation (solo `none` presente).
- Il nome `rq1_none_vs_canny` nei manifest è storico: il contenuto dichiara realmente tutte e quattro le condizioni.

Riepilogo audit locale: G42-A **0** non validi, G42-B **0**, G123 **16**, G456-A **213**, G456-B **0**. Dei 213 mancanti di G456-A, 204 sono coperti dalla continuazione G456-B: non contarli due volte come lacune del benchmark. Blocchi completi foto/stile/seed nei singoli run: rispettivamente **180, 144, 317, 270, 51**. Restano **104 foto comuni complete su tutti gli stili e tutti i seed**; per stile le foto complete sui tre seed sono 105 per GTA V, 104 per Pixel Art e 105 per Watercolor. Questi sono conteggi pre-metriche: eventuali errori dei valutatori possono ridurli.

### Configurazione di riferimento già usata

| Fattore | Valore nei manifest |
|---|---|
| Backbone | `stable-diffusion-v1-5/stable-diffusion-v1-5`, nessuna LoRA |
| Stili | `gta5`, `pixel_art`, `watercolor` |
| Condizioni | `none`, `canny`, `depth`, `segmentation` |
| Seed | 42, 123, 456 in run distinti |
| Step / CFG / img2img strength | 50 / 7.5 / 0.5 |
| ControlNet conditioning scale | 0.8 |
| Preprocessing fotografico esplicito | Disabilitato (`preprocessing_enabled=false`) |
| Dimensioni configurate | 512 × 512; controllare quelle effettive nell'audit |
| Scheduler registrato | `PNDMScheduler` |
| Canny | Soglie 100 / 200 |
| Depth / segmentation | `Intel/dpt-hybrid-midas` / `openmmlab/upernet-convnext-base` |

Il manifest conserva gli ID dei modelli, ma non dimostra da solo l'identità delle revisioni scaricate o dell'ambiente di due sessioni Kaggle. Registrare quanto è recuperabile da notebook, log e cache; dichiarare eventuali informazioni mancanti. Non sostituire retroattivamente il manifest con il TOML corrente.

## 2. Quali confronti costituiscono lo studio

### Confronti principali: usare tutti gli output esistenti

| ID analisi | Confronto appaiato, separato per stile | Domanda |
|---|---|---|
| A01 | `canny - none` | I bordi aiutano il contenuto? Quanto stile si perde o guadagna? |
| A02 | `depth - none` | La struttura spaziale aiuta con un diverso compromesso stilistico? |
| A03 | `segmentation - none` | Il vincolo semantico preserva meglio oggetti e disposizione? |
| A04 | `canny - depth`, `canny - segmentation`, `depth - segmentation` | Quale ramo offre il compromesso più utile per ciascuno stile? |
| A05 | A01–A04 separatamente per seed e poi aggregati | Le conclusioni sono stabili rispetto alla casualità? |
| A06 | A01–A04 per tipo di scena e casi difficili | Dove funziona o fallisce ogni ramo? |

Per ciascuno dei sei contrasti tra condizioni ci sono tre stili e quattro metriche: **72 stime di differenza** nella matrice completa. Le nove coppie ControlNet-vs-none per stile sono il nucleo della RQ1; i confronti tra ControlNet completano la lettura. Non serve rigenerare immagini per A01–A06.

`none` significa **SD img2img con prompt stilistico, senza ControlNet**: non significa foto originale, assenza di prompt o text-to-image. L'originale è il riferimento delle metriche di contenuto e delle griglie.

Questa è un'ablation della presenza del conditioning e un confronto fra tre sistemi strutturali. Cambiando ramo cambiano **sia preprocessore sia checkpoint ControlNet**: non attribuire causalmente ogni differenza al solo tipo di mappa. A scala 0.8 uguale confrontiamo un punto operativo condiviso; non dimostriamo che tutti i metodi siano confrontati al loro optimum.

### Confronti aggiuntivi: da attivare dopo l'analisi principale

I valori sotto sono proposte di protocollo, **non esperimenti già eseguiti**. Congelare griglia e subset prima di lanciarli, assegnare un ID e cambiare un fattore alla volta.

| ID esperimento | Fattore variato | Livelli proposti | Cosa resta fisso / scopo |
|---|---|---|---|
| E01 | Scala ControlNet, per ciascun ramo | 0.4, 0.8, 1.2 | Strength 0.5, stessi input/stili/seed; verificare il vincolo troppo forte o debole |
| E02 | Img2img strength | 0.3, 0.5, 0.7 | Tutte le condizioni, scala 0.8; distinguere il limite di stilizzazione dalla scelta ControlNet |
| E03 | Interazione scala × strength | Solo combinazioni motivate da E01/E02 | Verificare se l'effetto della scala dipende dalla strength |
| E04 | Qualità del preprocessore | Un parametro per volta, livelli da definire | Solo se le mappe mostrano errori sistematici; es. soglie Canny |
| E05 | Robustezza al prompt di valutazione | Piccolo insieme di testi fissati prima della verifica | Stesse immagini, nessuna generazione; verificare dipendenza da CLIP-T senza scegliere il testo più favorevole |

Ordine consigliato: **A01–A06 → eventuale E01 o E02 → E03 solo se necessario**. E04/E05 rispondono a problemi specifici. Non occorre esplorare automaticamente CFG, scheduler, step, risoluzione e tutte le combinazioni: allargare indiscriminatamente la griglia rende più difficile interpretare la RQ1.

Per tuning successivo, fissare un subset di sviluppo stratificato e un insieme di conferma separato, condivisi da tutti i metodi. Le 108 foto già esaminate restano un benchmark esplorativo: una divisione scelta dopo aver visto gli esiti non crea un test indipendente. Per una conferma indipendente servono immagini non usate nelle decisioni. Salvare liste e hash degli split. Se si seleziona una scala per ramo, mostrare sia il confronto originale a scala comune sia quello ottimizzato con uguale budget di ricerca.

## 3. Passo 1 — Audit e preparazione prima della GPU

- [x] Inventariare manifest, seed, parametri condivisi e record locali.
- [x] Eseguire l'audit locale e riportarne esiti e chiavi mancanti in questo documento.
- [x] Controllare hash degli input presenti tra shard e seed, geometrie, scheduler e prompt tramite audit e confronto dei manifest.
- [x] Archiviare i CSV/JSON di audit Kaggle: `result_evaluation_rq1/audit_seed_*/` e directory di evaluation.
- [x] Usare le 104 foto comuni complete, senza recuperare o rigenerare gli output mancanti (decisione dell'utente).
- [x] Congelare la lista delle 104 foto e preparare lo strumento di selezione coerente per l'evaluation, dopo OK dell'utente.
- [x] Congelare protocollo, testi CLIP-T e revisioni dei valutatori: `result_evaluation_rq1/evaluation_frozen.toml` e `environment.json`.

Eseguire l'audit su **ogni directory che contiene il proprio `run_config.json`**. `baseline_seed_42/`, `baseline_seed_456/` e `result_rq1/` non sono singoli run valutabili.

Esempio dalla radice del repository su Kaggle, dopo aver verificato i percorsi reali:

```bash
python --version  # serve Python >= 3.11
python -m src.evaluation.evaluate \
  --run-dir result_rq1/baseline_seed_42/test_1 \
  --audit-only --output-dir /kaggle/working/rq1_eval/G42-A_audit
```

Ripetere per G42-B, G123, G456-A, G456-B. Per ogni comando scegliere una directory di output nuova. L'audit non carica modelli; produce `audit.csv`, `audit_summary.json` e `generation_manifest.json`. Exit code 2 richiede di leggere gli errori, non prova che serva rigenerare tutto.

Per Kaggle è utile un `--output-dir` sotto `/kaggle/working/`: se gli artefatti sono in un dataset montato in `/kaggle/input/`, il default tenterebbe di scrivere nella directory del run, che è in sola lettura. Nel caso locale usare `.venv/Scripts/python.exe`, evitando un eventuale `python` di sistema precedente alla 3.11.

### Due strade valide per gli shard

**Percorso scelto dopo l'OK dell'utente:** usare [`configs/rq1_common104.json`](../../configs/rq1_common104.json) e [`src/evaluation/prepare_subset.py`](../../src/evaluation/prepare_subset.py) per preparare tre copie consolidate, una per seed, limitate alle stesse 104 foto. La selezione contiene ID inclusi/esclusi, hash SHA-256 degli input, matrice attesa e directory sorgenti. È stata verificata localmente l'intera selezione: **3.744 output presenti**, metadati coerenti e input corrispondenti agli hash congelati; nessun modello caricato.

Lo strumento senza `--output-dir` esegue solo verifiche. Con `--output-dir` copia input, PNG generati e metadati in una directory nuova; non copia le mappe di conditioning, che restano negli originali e non servono alle quattro metriche. Conserva manifest originali in sorgente e aggiunge provenienza ai manifest derivati, `selection.json` e `provenance.json` con gli hash dei PNG copiati. Non implementa resume della copia: usare una directory nuova anche dopo un'interruzione. Verificato offline su un piccolo run sintetico il ciclo copia → audit, il rifiuto di sovrascritture, hash input diversi e output mancanti; nessuna inferenza.

Questi comandi sono il riferimento per il futuro notebook, **non sono stati eseguiti su Kaggle**:

```bash
# Solo verifica: nessuna copia e nessuna inferenza.
python -m src.evaluation.prepare_subset --source-root /percorso/result_rq1

# Dopo autorizzazione: crea copie portabili, servono spazio e tempo di I/O.
python -m src.evaluation.prepare_subset \
  --source-root /percorso/result_rq1 \
  --output-dir /kaggle/working/rq1_common104
```

Il risultato previsto contiene `seed_42/`, `seed_123/`, `seed_456/`, ciascuna con **104 foto e 1.248 output**. Prima delle metriche rifare `--audit-only` su ciascuna: devono risultare zero output non validi. Per queste copie complete **non serve `--allow-partial`**. Se l'audit fallisce, fermarsi e leggere gli errori. I comandi più avanti sui run originari illustrano l'evaluator, ma per il benchmark scelto si useranno queste tre copie anziché i cinque shard grezzi.

1. **Valutare separatamente i cinque run originali**, poi aggregare i CSV a livello di singolo output. Per G123 e G456-A occorre esplicitamente `--allow-partial`; l'audit resta incompleto e l'analisi va dichiarata parziale finché non si risolvono le lacune.
2. **Preparare copie consolidate per seed**, senza modificare gli originali. Prima verificare configurazioni, hash input, unicità delle chiavi e provenienza; copiare i sample e costruire il manifest con l'unione degli input. Per 456 il manifest di `test1` contiene già 108 foto: aggiungere quelle 17 di `test2` non significa arrivare a 125. Rifare l'audit sulle copie. Questa preparazione non è automatizzata dall'evaluator.

La prima strada consente di iniziare subito; la seconda semplifica la lettura per seed. In entrambe manca ancora l'aggregazione finale fra i tre seed. **Non spostare cartelle o cambiare il manifest per nascondere output mancanti.** Se si recuperano o rigenerano combinazioni, registrare origine, parametri e compatibilità con i run iniziali.

## 4. Passo 2 — Evaluation completa su Kaggle

**Notebook eseguito dall'utente su Kaggle, dopo OK:** [RQ1_EVALUATION_COMMON104_KAGGLE.ipynb](../../notebooks/RQ1_EVALUATION_COMMON104_KAGGLE.ipynb), con [bundle iniziale](../../notebooks/rq1_evaluation_code.zip) e [istruzioni](../../notebooks/README_RQ1.md). Nella sessione reale Kaggle ha estratto lo ZIP: è stata adattata la cella per copiare `src/` e `configs/` e ricreare il bundle effettivo, conservato nell'export. Il notebook ha prodotto tre copie selezionate, audit completi e quattro metriche per tutti i seed; revisioni e ambiente sono archiviati. La sintassi e i controlli erano stati verificati localmente senza modelli; l'inferenza è avvenuta solo sulla T4 Kaggle. Anche l'aggregazione finale fra seed è ora completata e documentata nell'analisi quantitativa.

L'evaluator attuale è adeguato per audit e quattro metriche su un singolo run. Per questo studio non serve sostituirlo; vanno completate le analisi fra run, i confronti tra ControlNet e l'ispezione visiva.

Prima del lancio lungo fare un pilot su una copia di poche foto con manifest coerente. Controllare caricamento dei tre valutatori, VRAM, tempo per output e leggibilità dei primi punteggi. Un manifest completo con file rimossi artificialmente non è un pilot completo.

```bash
# Run completo: tutte e quattro le metriche, secondo configs/evaluation.toml.
python -m src.evaluation.evaluate \
  --run-dir result_rq1/baseline_seed_42/test_1 \
  --device cuda --output-dir /kaggle/working/rq1_eval/G42-A_full

# Run incompleto: accetta solo gli output validi e rende esplicita la parzialità.
python -m src.evaluation.evaluate \
  --run-dir result_rq1/baseline_seed_123 \
  --device cuda --allow-partial \
  --output-dir /kaggle/working/rq1_eval/G123_partial
```

Usare lo stesso file `configs/evaluation.toml` congelato per tutti i run. Attualmente le revisioni sono `main`: fissare commit immutabili prima del lancio definitivo e conservare `evaluators.json`, senza cambiare modello fra shard. Non modificare i testi dopo aver visto quale metodo vince.

- [x] Verifica tecnica iniziale su Kaggle e avanzamento controllato; nessun pilot separato o stima di durata documentata.
- [x] Evaluation seed 42 completata sulla copia consolidata di 104 foto (G42-A/B).
- [x] Evaluation seed 123 completata sulla copia di 104 foto (G123).
- [x] Evaluation seed 456 completata sulla copia consolidata di 104 foto (G456-A/B).
- [x] Archiviati CSV, JSON, log, bundle del codice con hash e ambiente Kaggle. Il bundle estratto non conserva il commit Git: usare hash del bundle e degli artefatti come provenienza disponibile.

Lanciare un run alla volta evita di caricare più copie dei valutatori sulla stessa GPU. Il codice salva progressivamente `per_image_metrics.csv`, ma **non implementa resume**: un nuovo comando ricalcola le metriche richieste. Un exit code 0 con `--allow-partial` può indicare successo sugli output validi senza rendere completo il benchmark.

## 5. Passo 3 — Consolidare senza introdurre bias

Questo è un passaggio di analisi da implementare o eseguire in notebook: non esiste un comando di merge nell'evaluator corrente. Non usare `src/evaluation/lora_eval.py` per questo scopo: è vuoto e riguarda l'estensione RQ2.

1. Unire i cinque `per_image_metrics.csv`, aggiungendo ID del run e percorso dell'evaluation per mantenere la provenienza.
2. Verificare unicità di `(image_id, style, condition, seed)`. Se una chiave ricompare, confrontare hash e metadati prima di scegliere; non fare una media di duplicati.
3. Verificare che la stessa foto abbia lo stesso `input_sha256` tra run. Verificare configurazioni dei valutatori, testi, dimensioni e configurazioni di generazione.
4. Costruire una tabella di copertura per foto × stile × seed × condizione. Distinguere output assente, audit non valido ed errore di metrica.
5. Per l'analisi principale usare le stesse foto con tutti e tre i seed e tutte e quattro le condizioni nello stile considerato. Per confrontare gli stili direttamente usare anche l'intersezione comune ai tre stili. Dichiarare sempre le esclusioni.
6. Conservare gli altri output nell'analisi supplementare dei blocchi completi disponibili, specificando la diversa copertura. Non eliminare i fallimenti dal conteggio tecnico.

L'evaluator seleziona già blocchi completi foto/stile/seed **dentro il singolo run**, ma non impone tre seed per ogni foto. Non basta concatenare `summary_metrics.csv` o mediare le cinque medie: shard da 60 e 48 foto avrebbero pesi errati e non ricostruiremmo gli appaiamenti.

### Regola di calcolo da mantenere

Per metrica `m`, stile `s` e due condizioni `A`, `B`:

```text
delta(i, s, seed) = m(i, s, A, seed) - m(i, s, B, seed)
delta_per_foto(i, s) = media dei delta sui medesimi seed 42, 123, 456
delta_finale(s) = media dei delta_per_foto sulle foto ammissibili
```

Calcolare media, deviazione standard tra foto e IC bootstrap al 95%, ricampionando **foto intere** con tutti i seed e condizioni associati (2.000 repliche, seed 2026 per coerenza con il codice). Non trattare i tre seed della stessa foto come tre fotografie indipendenti. Per l'analisi della variabilità dei seed riportare anche i delta per singolo seed; tre seed danno una verifica iniziale, non una stima precisa di ogni possibile variazione casuale.

Gli IC di `paired_vs_none.csv` riguardano le **differenze appaiate**. Il confronto fra IC delle due medie separate non li sostituisce. A04 richiede di calcolare gli altri tre contrasti dai punteggi individuali. Non ricavare il loro IC sottraendo gli estremi degli IC già presenti.

Trattare le 72 stime come descrittive, riportando ampiezza e incertezza. Se si aggiungono test formali, fissare prima ipotesi primarie e famiglia dei confronti e correggere la molteplicità, per esempio con Holm; non dichiarare un vincitore solo perché uno fra molti intervalli esclude lo zero. Non introdurre a posteriori una soglia di rilevanza calibrata sui risultati favorevoli.

## 6. Passo 4 — Leggere le metriche insieme

| Metrica | Direzione nel confronto | Limite da ricordare |
|---|---|---|
| `clip_i` | Maggiore similarità globale input/output | Può favorire output poco trasformati; non certifica geometria o ogni oggetto |
| `dino_i` | Maggiore similarità visiva input/output | Sensibile anche al cambio di stile; non è una percentuale di contenuto |
| `semantic_miou` | Maggiore accordo semantico input/output | Accordo tra predizioni SegFormer, senza ground truth umana |
| `clip_t` | Maggiore allineamento al testo dello stile | Non è una probabilità né una misura completa della qualità estetica |

Niente somma arbitraria delle quattro metriche. CLIP-I e CLIP-T condividono il backbone e non sono evidenze indipendenti. SegFormer e il segmentatore del conditioning sono distinti ma condividono ADE20K; verificare visivamente un eventuale vantaggio di segmentation. Leggere CLIP-T all'interno dello stesso stile, senza ordinare gli stili usando i suoi valori assoluti.

### Tabelle e figure da produrre

- Una tabella per stile: quattro condizioni × quattro metriche, media e IC, numero di foto, seed e output utilizzati.
- Una tabella dei delta vs `none` e una dei tre contrasti fra ControlNet, con IC appaiati e conteggi.
- Scatter contenuto–stile: CLIP-T vs CLIP-I, CLIP-T vs DINO-I, CLIP-T vs mIoU. Mostrare anche dispersione e punti per foto, senza dedurre la superiorità dalle sole medie.
- Distribuzioni dei delta e quota di foto con delta positivo/negativo; annotare se il beneficio medio dipende da pochi outlier.
- Grafico dei delta per seed e, se motivate, analisi per categoria di scena.
- Griglie con originale e quattro condizioni per lo stesso stile e seed.

Il fronte di Pareto aiuta a descrivere il compromesso: una condizione domina un'altra sui valori osservati solo se non peggiora nessuna metrica considerata e ne migliora almeno una. Una dominanza sulle sole medie non elimina l'incertezza o i fallimenti sui singoli input.

**Categorie:** i file sono nella radice delle liste input; il codice assegna `unassigned`, non inferisce categorie dai prefissi. Se servono scene urbane, persone, natura, interni ecc., creare una tabella esplicita `image_id → categoria`, con regole fissate guardando gli originali e prima di confrontare i punteggi. I prefissi di provenienza non sono classi semantiche. Dichiarare numerosità e risultati esplorativi dei gruppi piccoli.

## 7. Passo 5 — Decidere su cosa concentrarsi

Le righe seguenti sono regole operative, non risultati già osservati. Prima di aggiungere generazioni, verificare che l'effetto non dipenda da dati mancanti, metriche errate o selezione degli esempi.

| Segnale osservato | Verifica immediata | Passo successivo motivato |
|---|---|---|
| Contenuto aumenta, CLIP-T diminuisce | Le griglie confermano minore stilizzazione? Succede su molte foto e seed? | E01 per ridurre il vincolo; E02 separato se anche `none` stilizza poco |
| Contenuto e CLIP-T aumentano | IC dei delta, coerenza sui seed, artefatti e fedeltà degli oggetti | Confermare il risultato; nessuna nuova griglia obbligatoria |
| Tutto peggiora rispetto a `none` | Allineamento e qualità delle mappe, checkpoint e parametri | Correggere eventuali errori; altrimenti E01 e documentare il limite |
| CLIP-I alto, DINO-I/mIoU o giudizio umano basso | Oggetti mancanti, cambi di layout, bias degli embedding | Analisi mirata dei casi discordanti; evitare conclusioni da CLIP-I solo |
| Segmentation vince solo in mIoU | Mappe predette, classi dominanti e segmentazione di output stilizzati | Verifica cieca del layout; discutere bias ADE20K |
| `none` e ControlNet hanno tutti stile debole | Prompt, strength, esempi per stile e giudizio umano | E02; il conditioning potrebbe non essere il collo di bottiglia |
| Il vincitore cambia per stile | Delta entro stile e stessa copertura delle foto | Riportare conclusioni per stile; evitare un vincitore globale forzato |
| Il vincitore cambia tra seed / IC ampi | Effetti per foto e seed, outlier, lacune | Conclusione incerta; ulteriori seed solo con motivazione e budget registrati |
| Problemi concentrati in scene specifiche | Categorie definite dagli originali e mappe di quei casi | A06 e, se necessario, E04 |
| Differenze piccole e immagini equivalenti | Precisione degli IC e rilevanza pratica | Documentare assenza di evidenza di un vantaggio utile; non proclamare equivalenza statistica |

## 8. Passo 6 — Valutazione visiva e umana

Prevedere due insiemi distinti: un campione casuale fissato per valutare la qualità complessiva e una selezione diagnostica di successi, fallimenti e disaccordi tra metriche. I casi estremi servono a spiegare i risultati; non stimano la frequenza dei difetti sull'intero benchmark.

Per il campione umano: fissare dimensione e lista prima di annotare, bilanciare stili e condizioni, nascondere i nomi dei metodi e randomizzare l'ordine. Mostrare originale e stile richiesto. Far giudicare separatamente **contenuto/layout**, **riconoscibilità dello stile** e **artefatti/qualità**, con rubrica comune, per esempio 1–5 con ancore esplicite. Preferire più valutatori e registrare numero di giudizi e accordo; una valutazione dell'autore rimane un'ispezione qualitativa con quel limite dichiarato.

Conservare `image_id`, stile, seed, ordine mostrato, valutatore anonimo, giudizi e note. Se si usano più seed della stessa foto, tener conto del raggruppamento per foto anche nell'analisi umana. Cercare in particolare oggetti aggiunti/rimossi, volti e mani alterati, prospettiva incoerente, texture fotografiche residue e caratteristiche stilistiche mancanti. Per Pixel Art osservare palette, pixel e coerenza dei blocchi; per Watercolor lavature e bordi; per GTA V resa da videogioco anziché semplice fotografia o illustrazione.

## 9. Registro centrale: cosa abbiamo fatto e cosa manca

Stati da usare: `PIANIFICATO`, `GENERATO_PARZIALE`, `GENERATO`, `VALUTATO_PARZIALE`, `VALUTATO`, `ANALIZZATO`, `CHIUSO`, `NON_NECESSARIO`. Aggiornare lo stato solo quando esiste l'artefatto corrispondente. Una voce può essere chiusa anche con risultato negativo o inconcludente, purché documentato.

| ID | Attività / esperimento | Stato al 12/11/2026 | Evidenza disponibile | Prossima azione |
|---|---|---|---|---|
| B01 | Generazione originaria su 108 foto | GENERATO_PARZIALE | G42-A/B, G123, G456-A/B; audit: 3.863 output validi unici | Conservare come provenienza; nessun recupero delle lacune |
| B02 | Selezione comune: 104 foto, 3 stili, 4 condizioni, 3 seed | VALUTATO | Selezione congelata; audit Kaggle: 3.744 validi, zero invalidi | Conservare copertura e provenienza |
| V01 | Metriche sui 3.744 output di B02 | VALUTATO | `result_evaluation_rq1/`; tre seed completi, export COMPLETE, hash e punteggi verificati | Analisi quantitativa completata; passare al controllo visivo dopo OK |
| N01 | Preparazione notebook Kaggle e bundle | CHIUSO | Eseguito dall'utente su T4; ZIP estratto da Kaggle gestito adattando la cella di copia | Conservare il bundle effettivo nell'export, distinto dal bundle locale iniziale |
| A01 | Canny vs none | ANALIZZATO | `analysis_common104_v1/paired_contrasts.csv`; delta positivi di contenuto, CLIP-T negativo medio nei tre stili | Verificare visivamente il compromesso |
| A02 | Depth vs none | ANALIZZATO | Stesso CSV, IC appaiati e distribuzioni disponibili | Controllare esempi e riconoscibilità stilistica |
| A03 | Segmentation vs none | ANALIZZATO | Stesso CSV; IC CLIP-T Watercolor comprende zero | Verificare layout e bias mIoU |
| A04 | Confronti tra ControlNet | ANALIZZATO | Tutti i tre contrasti aggiuntivi; differenze e incertezze per stile | Esaminare Canny–depth Watercolor e depth–segmentation negli altri stili |
| A05 | Robustezza sui seed | ANALIZZATO | `seed_stability.csv`; stessa direzione media in 71/72 contrasti; variazione sulle singole foto documentata | Conservare l'eccezione mIoU depth–segmentation Watercolor e analizzare casi discordanti |
| A06 | Scene e fallimenti | PIANIFICATO | Campione casuale e casi diagnostici fissati in VIS01; nessuna categoria o annotazione percettiva assegnata | Concordare annotazioni e definire categorie senza inferirle dai prefissi |
| VIS01 | Preparazione griglie qualitative | CHIUSO | 30 PNG, indice HTML, 120 schede vuote e manifest in `visual_review_common104_v1/`; tre test di selezione superati | Aprire la galleria; annotare solo dopo OK e rubrica concordata |
| VIS02 | Modulo HTML e collaborazione sulle 30 griglie | CHIUSO | Quattro domande testuali standard, nome revisore, autosave nel browser, parti A/B da 15, import/export JSON/CSV e ZIP portabile; test di persistenza e fusione superati | I revisori compilano e condividono i JSON; mantenere separate le versioni e la provenienza casuale/diagnostica |
| Q01 | Raccolta delle osservazioni qualitative per griglia | PIANIFICATO | Modulo pronto; nessun giudizio compilato dal codice | Raccogliere i JSON esportati e analizzare le risposte senza trattarle come valutazione cieca |
| H01 | Giudizi umani ciechi | PIANIFICATO | Nessun giudizio raccolto; le griglie VIS01 mostrano nomi dei metodi e punteggi | Fissare campione/rubrica cieca e valutatori separatamente |
| E01 | Sweep scala ControlNet | PIANIFICATO, condizionale | Nessuna nuova generazione richiesta | Attivare solo con ipotesi derivata da A01–A06 |
| E02 | Sweep strength | PIANIFICATO, condizionale | Nessuna nuova generazione richiesta | Attivare se stilizzazione insufficiente o trade-off da chiarire |
| E03 | Interazione scala × strength | PIANIFICATO, condizionale | — | Decidere dopo E01/E02 |
| E04 | Preprocessore | PIANIFICATO, condizionale | Mappe esistenti ispezionabili | Attivare per errori sistematici delle mappe |
| E05 | Robustezza CLIP-T ai testi | PIANIFICATO, condizionale | Protocollo testuale corrente disponibile | Attivare per dubbi sulla misura dello stile |

La prova storica su `olanda.png`, documentata nelle guide precedenti, resta un pilot tecnico: non è conteggiata come evidenza del benchmark corrente. RQ2 con LoRA Pixel Art ha un registro separato in [RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md); non mescolare quelle immagini con B01.

### Scheda da duplicare dopo ogni esperimento o analisi

```markdown
### [ID] — [Titolo] — [Data]
- Stato:
- Domanda / ipotesi fissata prima del lancio:
- Motivazione e analisi precedente che la supporta:
- Directory sorgenti / notebook / commit / log:
- Fattore variato e livelli; parametri mantenuti costanti:
- Foto / split / stili / condizioni / seed previsti:
- Output attesi / presenti / validi / valutati:
- Esclusioni e motivazioni; foto comuni effettivamente analizzate:
- Protocollo metriche e revisioni; percorso dei CSV e JSON:
- Contrasti e metodo di aggregazione / incertezza:
- Risultati per stile, delta e IC (link tabella):
- Robustezza per seed e scene; casi discordanti:
- Griglie / giudizi umani / limiti:
- Decisione: cosa concludiamo e con quale evidenza:
- Prossima azione oppure motivo per non fare altri esperimenti:
```

### Diario delle decisioni

| Data | ID collegati | Evidenza | Decisione | Attività successiva |
|---|---|---|---|---|
| 12/11/2026 | B01, V01 | Audit su cinque cartelle; 3.863 output validi, 25 lacune; nessun CSV di metriche locale | Analizzare gli output esistenti prima di ampliare la griglia | Recupero lacune o subset comune, evaluation Kaggle |
| 12/11/2026 | B01, B02, V01 | 104 foto complete per tutti gli stili/condizioni/seed | L'utente sceglie il subset comune, senza rigenerazioni; GPU locale esclusa; OK richiesto prima di ogni passo | Proporre congelamento lista e selezione; metriche successivamente su Kaggle |
| 12/11/2026 | B02 | OK ricevuto; lista congelata e strumento verificato su 3.744 output e su copia sintetica | Preparazione completata senza modelli o GPU; originali intatti | Chiedere OK per preparare il notebook Kaggle, senza avviarlo |
| 12/11/2026 | N01, V01 | OK ricevuto per preparare il notebook; file e bundle creati e verificati offline | Notebook pronto e non eseguito; inferenza bloccata di default | Caricare i materiali su Kaggle, configurare i percorsi e attendere OK prima delle metriche |
| 12/11/2026 | B02, V01 | OK al lancio ricevuto; tre audit e tre evaluation complete su T4; export COMPLETE in `result_evaluation_rq1/` | Quattro metriche disponibili su tutti i 3.744 output | Aggregazione locale autorizzata separatamente |
| 12/11/2026 | A01–A05 | OK all'analisi ricevuto; 48 riepiloghi e 72 contrasti con IC, tabelle per seed e due figure; quattro test statistici offline superati | Analisi quantitativa completata, senza GPU; Canny privilegia il contenuto misurato, none il CLIP-T medio | Proporre ispezione visiva delle immagini esistenti, senza nuove generazioni |
| 12/11/2026 | VIS01, A06, H01 | OK ricevuto per preparare griglie: sei foto casuali × tre stili + quattro casi diagnostici per stile; seed 42 fisso | Galleria e 30 PNG creati senza modelli/GPU, 120 schede non valutate; selezione e hash salvati | Concordare lettura sistematica e rubrica; H01 e analisi per scena restano aperti |
| 12/11/2026 | VIS02, Q01 | L'utente richiede annotazione di tutte le 30 griglie tramite HTML e divisione con un collega | Una scheda di quattro risposte testuali per griglia/revisore; autosave locale, backup/riunione JSON, conflitti conservati e parti da 15 | Compilazione umana nel browser e raccolta dei JSON; nessuna nuova inferenza |

## 10. Quando possiamo considerare conclusa la RQ1

- [x] Copertura finale dichiarata: 104 foto comuni, quattro esclusioni per incompletezza esplicite.
- [x] Nessun duplicato ambiguo; input e protocolli comparabili fra shard e seed nei manifest disponibili.
- [x] Quattro metriche calcolate e controllate, nessun errore sui 3.744 output e artefatti archiviati.
- [x] A01–A04 riportate per tutti e tre gli stili con differenze appaiate, IC e numerosità.
- [x] A05 completata, senza contare i seed come foto indipendenti.
- [ ] A06 e griglie qualitative completate; protocollo umano e limiti dichiarati.
- [ ] Ogni ablation aggiuntiva ha ipotesi, unico fattore variato o interazione esplicita, copertura e decisione.
- [ ] E01–E05 eseguite quando motivate oppure marcate `NON_NECESSARIO` con motivazione.
- [ ] Conclusioni coerenti con il compromesso contenuto–stile; nessun vincitore imposto se evidenza mista o incerta.
- [ ] Registro aggiornato con link ai risultati, limiti di generalizzazione e materiale per il report finale.

La priorità attuale è **compilare le quattro domande nella [galleria](../../result_evaluation_rq1/visual_review_common104_v1/index.html), eventualmente dividendo A/B → esportare e riunire i JSON dei revisori → analisi delle osservazioni per stile e fallimenti, dopo OK → eventuale valutazione umana cieca**. Le ablation aggiuntive restano proposte condizionali e richiedono un nuovo OK; non si rigenerano gli output mancanti. Nessun nuovo notebook Kaggle serve per questa raccolta.
