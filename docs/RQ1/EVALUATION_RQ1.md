# RQ1 — Come eseguire e interpretare l'evaluation

Questo documento descrive il protocollo implementato il **29 ottobre 2026**. Il run Kaggle riferito dall'utente è in corso, con tutti gli stili e senza il preprocessing fotografico esplicito. I risultati reali non sono ancora stati valutati localmente.

## 1. Obiettivo e unità del confronto

La RQ1 confronta `none`, `canny`, `depth` e `segmentation` nei tre stili GTA V, Pixel Art e Watercolor. Vogliamo capire come cambia il compromesso fra contenuto preservato e stile acquisito, mantenendo costanti fotografia, prompt dello stile, seed e altri parametri.

L'evaluation è un programma separato: **non carica Stable Diffusion, non genera immagini e non modifica gli output**. Può essere eseguita sul computer locale o su Kaggle quando la cartella del run è disponibile.

Ogni unità di confronto è una combinazione `image_id × style × seed`, con un output per ciascuna condizione. Con 108 fotografie, tre stili, quattro condizioni e un seed, sarebbero attesi **1.296 output**. È un conteggio condizionale: fa fede il manifest del run Kaggle, non il numero di file nella cartella locale attuale.

## 2. Che cosa misuriamo

### CLIP-I: contenuto globale

Si calcola l'embedding CLIP della fotografia effettivamente fornita al runner e quello dell'output. Il punteggio è la similarità coseno fra i due vettori:

```text
CLIP-I = dot(embedding_input, embedding_output) / (norm_input * norm_output)
```

Un valore maggiore indica maggiore somiglianza nello spazio CLIP. Non garantisce che tutti gli oggetti siano rimasti identici o nella stessa posizione. Anche un'immagine poco stilizzata può ottenere un punteggio alto: per questo va letta insieme a CLIP-T.

### CLIP-T: allineamento al testo dello stile

Si confronta l'embedding CLIP dell'output con quello di una descrizione fissa dello stile. Le descrizioni sono in `configs/evaluation.toml`, sezione `style_texts`, e sono uguali per tutte le condizioni dello stesso stile.

Il punteggio è un **coseno non scalato**, nell'intervallo teorico `[-1, 1]`: non è una probabilità, una percentuale o il logit CLIP moltiplicato per la temperatura. Non esiste una soglia universale che certifichi lo stile.

Le descrizioni di valutazione sono brevi e distinte dai lunghi prompt di generazione. Congelarle prima del confronto; modificarle per favorire gli output osservati renderebbe l'analisi dipendente dal test. I testi troppo lunghi vengono rifiutati invece di essere troncati silenziosamente.

Il backbone scelto è `openai/clip-vit-base-patch32`. Si usa il suo preprocessing standard: resize, crop e normalizzazione interni possono ridurre l'informazione sui bordi di fotografie panoramiche. L'assenza del nostro preprocessing fotografico in generazione non elimina quello necessario ai valutatori. [Documentazione CLIP](https://huggingface.co/docs/transformers/model_doc/clip).

### Consistenza semantica mIoU: layout delle classi

Si segmentano **sia l'input sia l'output** con `nvidia/segformer-b2-finetuned-ade-512-512`, diverso dall'UPerNet usato per il conditioning. Il checkpoint usa ADE20K con 150 classi. Non si riutilizzano le label salvate dal conditioning come riferimento di valutazione. [Model card SegFormer](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512).

Procedura esatta:

1. Definire una griglia comune con rapporto d'aspetto dell'input e lato lungo massimo 512 pixel, senza ingrandire input più piccoli.
2. Eseguire il preprocessore e il modello di valutazione su ciascuna immagine.
3. Interpolare i logits con bilinear sulla griglia comune, poi applicare argmax. Non interpolare colori di palette e non applicare offset agli ID predetti.
4. Per ogni classe, calcolare `IoU = intersezione / unione`.
5. Fare la media delle IoU delle classi presenti in almeno una delle due mappe.

Una classe assente in entrambe è esclusa; presente solo da un lato contribuisce zero. Non viene esclusa automaticamente la classe 0, che è una classe ADE20K valida nelle predizioni. Il risultato è in `[0, 1]`.

La metrica misura **accordo fra due predizioni**, non accuratezza rispetto a ground truth umana. Usare un modello diverso riduce il riuso diretto del valutatore del conditioning, ma non elimina i bias condivisi del dataset ADE20K o la difficoltà di segmentare immagini stilizzate.

### DINO-I: similarità visiva complementare

Dal 29 ottobre 2026 è inclusa anche `dino_i`: coseno non scalato fra il token CLS finale dell'input e quello dell'output, estratti con **DINOv2 ViT-B/14**, checkpoint `facebook/dinov2-base`. Non è DINOv3. Il modello è congelato e usa il processore del checkpoint per resize/crop/normalizzazione. Non usa testo, classi ADE20K o confronti fra patch. [Model card DINOv2](https://huggingface.co/facebook/dinov2-base).

Serve come secondo indicatore di preservazione del contenuto, complementare a CLIP-I. Un cambio di stile può influenzarlo anche quando il contenuto è ben preservato; il vettore globale non garantisce l'allineamento geometrico. Interpretare il coseno come similarità visiva, non come percentuale di contenuto conservato, e non confrontare direttamente i valori assoluti con CLIP-I.

La sezione `[dino]` in `configs/evaluation.toml` definisce checkpoint e revisione. I CSV e le differenze appaiate includono automaticamente la nuova metrica. Per calcolare solo DINO, senza caricare CLIP o SegFormer:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --run-dir "outputs/<RUN>" --metrics dino_i
```

## 3. Quali file servono da Kaggle

Conservare **l'intera directory del singolo run**, inclusi:

```text
<run>/
  run_config.json
  samples/<image_id>/
    input_original.png                  # run --no-preprocess
    run_config.json
    metadata.jsonl
    maps/...
    images/<style>/<condition>/<nome>_seed42.png
```

Per un run con preprocessing il riferimento è invece `input_preprocessed.png`. Il valutatore usa questo snapshot del run, non l'immagine eventualmente cambiata in `data/raw`.

I percorsi assoluti `/kaggle/...` nei metadati vengono sostituiti ricostruendo la struttura locale `samples/<image_id>/...`. Si può quindi trasferire la cartella senza modificare i JSON. Non appiattire o rinominare le sottocartelle.

Si valuta **un run alla volta**. Run separati per condizione non vengono uniti automaticamente: prima di unirli servirebbe verificare stesso dataset, stessi input e parametri e costruire un manifest comune. I run storici senza `input_images` e struttura `samples/` non sono supportati automaticamente.

## 4. Primo comando: solo controllo dei file

Dalla radice del repository, sostituire `<RUN>` con la cartella effettiva:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --run-dir "outputs/<RUN>" --audit-only
```

Questo comando non importa Torch/Transformers per l'inferenza e non scarica pesi. Richiede comunque Python 3.11+ e Pillow installati. Usa il manifest originale per controllare anche immagini completamente mancanti, non solo gli output che trova.

Controlla record di successo, file leggibili, input salvato, dimensioni, prompt e parametri rispetto al manifest, checkpoint e scale ControlNet, scheduler e duplicati. Le dimensioni degli output devono essere uguali tra le condizioni/stili/seed dello stesso input; è ammessa una variazione di rapporto d'aspetto input/output fino al 2% per piccoli arrotondamenti dimensionali. Questo controllo non dimostra che il crop interno di ogni pipeline sia identico: verificare anche visivamente le prime griglie.

Produce:

- `audit.csv`: una riga per output atteso, con stato e problema rilevato;
- `audit_summary.json`: conteggi e indicazione se sono presenti tutte e quattro le condizioni RQ1;
- `generation_manifest.json`: copia del manifest analizzato.

Exit code `0` indica audit completo; `2` indica output non validi/mancanti. Un audit completo di una sola condizione non equivale al completamento della RQ1: controllare `full_rq1_conditions`.

## 5. Secondo comando: calcolo delle metriche

Prima verificare `configs/evaluation.toml`. Di default seleziona tutte e quattro le metriche (`clip_i`, `clip_t`, `semantic_miou`, `dino_i`) e sceglie CUDA se disponibile, altrimenti CPU. Il primo caricamento richiede accesso ai checkpoint Hugging Face; il tempo e la memoria effettivi vanno misurati su poche immagini nell'ambiente di esecuzione.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --run-dir "outputs/<RUN>" --device cuda
```

Su Kaggle, dalla radice del repository:

```bash
python -m src.evaluation.evaluate --run-dir /kaggle/working/outputs/<RUN> --device cuda
```

È possibile eseguire prima soltanto CLIP e poi tutte le metriche, sapendo che ogni invocazione crea una nuova evaluation e ricalcola quanto richiesto:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --run-dir "outputs/<RUN>" --metrics clip_i clip_t
```

Il default richiede che tutti gli output attesi superino l'audit. Per analizzare esplicitamente un run incompleto:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --run-dir "outputs/<RUN>" --allow-partial
```

I punteggi individuali vengono salvati anche per output rimasti senza confronto completo. Nei riepiloghi entrano solo blocchi foto/stile/seed con tutte le condizioni del manifest valutate con successo. Se non resta alcun blocco completo, i riepiloghi sono vuoti: controllare `evaluation_summary.json`.

Ogni invocazione crea `<run>/evaluation/<timestamp_UTC>/`; `--output-dir` permette di indicare una directory diversa, che deve essere nuova. Non sovrascrive valutazioni precedenti. Le righe già completate sono salvate durante l'esecuzione, ma **non è implementata la ripresa automatica**: una nuova invocazione riparte dall'inizio. Errori di memoria interrompono il processo.

## 6. Come leggere i risultati

| File | Contenuto |
|---|---|
| `per_image_metrics.csv` | Punteggi, ID, stile, condizione, seed, categoria, dimensioni, hash input/output, stato/errori |
| `summary_metrics.csv` | Media, deviazione standard fra input e intervallo bootstrap per stile/condizione/metrica |
| `paired_vs_none.csv` | Differenze appaiate `condizione - none`, con la stessa aggregazione |
| `evaluation_summary.json` | Output riusciti e numero di blocchi completi usati |
| `evaluation_config.json` | Protocollo, metriche richieste, device, versioni, commit e regola di aggregazione |
| `evaluators.json` | Config dei modelli/processori e revisioni risolte, se disponibili |

I riepiloghi sono prodotti sia globalmente (`category=__all__`) sia per categoria, derivata dalla sottocartella dell'immagine. Le foto direttamente nella radice hanno `category=unassigned`: non si inventa una categoria dal nome del file.

Per ogni foto si mediano prima i seed disponibili nei blocchi completi, poi si mediano le foto, che hanno quindi uguale peso. Viene registrato il numero di output oltre al numero di input. Non si considerano i seed della stessa foto come osservazioni indipendenti.

Gli intervalli al 95% usano 2.000 ricampionamenti bootstrap delle medie per input, seed bootstrap 2026. Con un solo input deviazione standard e intervallo sono vuoti. Con un solo seed di generazione, gli intervalli descrivono la variabilità fra foto; non quantificano quella fra seed. In un run parziale anche il numero di seed utilizzabili per foto può variare: dichiararlo nel report e preferire la matrice completa per l'analisi finale.

Esempio interpretativo, **non risultato reale**: se Canny aumenta CLIP-I ma riduce CLIP-T rispetto a `none`, suggerisce un guadagno di contenuto accompagnato da minore allineamento allo stile. Un valore positivo in `paired_vs_none.csv` indica una metrica maggiore della baseline, non automaticamente una qualità complessiva migliore.

Non sommare le metriche in un punteggio arbitrario e non confrontare i valori assoluti CLIP-T di stili diversi come fossero una scala calibrata di difficoltà.

## 7. Riproducibilità e verifiche già eseguite

Le revisioni configurate sono inizialmente `main`. Per il protocollo definitivo, sostituirle con commit immutabili; `evaluators.json` registra le revisioni risolte del modello quando il backend le espone. La configurazione del processore viene salvata insieme a quella del modello.

L'ambiente locale ispezionato contiene Python 3.11.9, Torch 2.14.1+cu126, Transformers 5.19.0, Diffusers 0.41.0, NumPy 2.4.6 e Pillow 12.3.0. Queste versioni **non coincidono con tutti i vincoli storici di `requirements.txt`**. Nessuna dipendenza è stata installata o aggiornata in questa sessione. L'evaluation non importa Diffusers, ma il caricamento effettivo dei checkpoint scelti resta da verificare nell'ambiente Kaggle/local definitivo.

Verifiche offline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Esito locale: 16 test passati.** Coprono formule, classi assenti, aggregazione e confronti appaiati, output mancanti, duplicati, incoerenze di prompt/geometria, trasferimento dei percorsi e CLI di audit. Un test attraversa l'intero flusso di evaluation e produzione CSV con valutatori deterministici, ora incluso DINO; un altro verifica l'esecuzione di solo DINO senza caricare gli altri valutatori. Tre ulteriori test eseguono i forward CLIP, SegFormer e DINOv2 con piccoli modelli casuali locali per verificare le API; non misurano qualità semantica e non sostituiscono il test con i pesi preaddestrati.

## 8. Che cosa faremo dopo i primi CSV

1. Controllare distribuzioni, errori ed eventuali anomalie su poche immagini.
2. Costruire griglie comparative per ciascuno stile, senza scegliere soltanto esempi favorevoli.
3. Produrre scatter CLIP-T vs CLIP-I e CLIP-T vs mIoU, usando confronti appaiati e riportando la dispersione.
4. Preparare una piccola valutazione umana cieca su stile, contenuto e artefatti.
5. Scrivere conclusioni e limiti, poi valutare più seed o un'ablation della control scale se necessaria.

Grafici, griglie e valutazione umana sono passi successivi: questa implementazione produce audit, punteggi e tabelle, senza dichiarare la RQ1 completata.


## 9. Riutilizzo delle metriche nella RQ2 Pixel Art

Aggiornamento al 10 novembre 2026. Lo stile dei LoRA RQ2 e **Pixel Art**;
Watercolor citato in questa guida rimane uno stile del benchmark RQ1.
Il training e l'inferenza RQ2 sono gia implementati: protocollo e stato in
[RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md).

L'evaluator puo calcolare le quattro metriche su un singolo run img2img RQ2,
ma i riepiloghi esistenti aggregano per condizione all'interno di quel run.
Se il run usa un LoRA, anche la condizione `none` usa lo stesso adattatore:
`paired_vs_none.csv` confronta quindi ControlNet contro img2img con quel LoRA,
non LoRA contro assenza di adattatore. Per confrontare off/4/8/16/32 servono
run distinti e un'aggregazione appaiata fra foto, seed e condizione.

L'aggregatore dedicato non e ancora implementato (`src/evaluation/lora_eval.py`
e vuoto). Occorre verificare fra run firme dataset, revisione SD, parametri
training e inferenza, pesi e scala LoRA. Il pilota rank 8 sul vecchio dataset
non puo essere confrontato direttamente con il rank 4 sul dataset attuale.
La loss di validation del trainer misura denoising e non sostituisce le
metriche sulle immagini. CLIP-T non e una percentuale di successo stilistico.
