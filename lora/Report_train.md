# Report dei training LoRA Pixel Art — RQ2

**Data di verifica: 10 novembre 2026.** Report basato sugli artefatti locali
dei rank **4, 8, 16 e 32**: configurazioni, CSV metriche, riepiloghi finali e
file `pytorch_lora_weights.safetensors`. Nessun training o test di inferenza
è stato avviato per produrre questo documento.

## 1. Esito complessivo

**Tutti e quattro i training risultano completati a 1000 aggiornamenti.**
I run condividono dataset, split, caption, revisione del backbone, impostazioni
operative, GPU, precisione effettiva e versioni software. Nelle configurazioni
registrate cambiano rank, alpha e, per alcuni run, la directory di output.

Il rank 32 ha la validation loss finale più bassa, ma il vantaggio sul rank 4
è piccolo: **0.00054336 in valore assoluto, circa 0.51% relativo**. Questo
risultato riguarda il denoising sul validation set e non dimostra che il
rank 32 produca la migliore Pixel Art o preservi meglio i contenuti.

Il rank 8 qui analizzato usa il **dataset corrente**, come gli altri rank;
non è il pilota storico addestrato sulla precedente selezione di immagini.

## 2. Protocollo comune verificato

| Parametro | Valore comune ai quattro run |
|---|---|
| Modello | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Revisione SD risolta | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Dataset | 400 immagini Pixel Art: 360 train, 40 validation |
| Risoluzione / interpolazione | 512×512 / nearest-neighbor con crop centrale |
| Seed training | 42 |
| Budget | 1000 aggiornamenti dell'optimizer |
| Batch / accumulo | 1 / 4, batch effettivo 4 |
| Optimizer | AdamW, LR 0.0001, weight decay 0.01 |
| Scheduler LR | `constant_with_warmup`, warmup 50 |
| Gradient clipping | max norm 1.0 |
| Gradient checkpointing | Attivo |
| Target LoRA | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Scaling | alpha = rank, quindi alpha/rank = 1 |
| Validation e checkpoint | Ogni 250 aggiornamenti |
| GPU / device | Tesla T4 / CUDA |
| Mixed precision effettiva | FP16 |
| DataLoader workers | 0 |
| Ripresa da checkpoint | Nessuna nei quattro riepiloghi: `resumed_from = null` |

Il training aggiorna soltanto gli adattatori della U-Net. VAE, text encoder
CLIP e pesi originali della U-Net rimangono congelati; ControlNet non viene
addestrato. La validation è una MSE di denoising con rumore e timestep a seed
fissato, senza aggiornamento dei pesi e senza generazione di immagini.

La scelta alpha = rank mantiene costante il fattore esplicito di scala, ma
non rende uguali capacità o dinamiche di ottimizzazione dei diversi rank.

### Dataset e firme

Il dataset locale è descritto in
[Dataset_LoRA.md](../data/pixel_art/Dataset_LoRA.md). Le firme di tutti i run
sono state confrontate anche con quelle ricalcolate sul dataset attuale:
**nomi, caption e byte delle immagini corrispondono**.

- Train: `87246eb9966643cc01862bee1d17a19161b9fbaef500dc1d42aa0d98b6798ae4`.
- Validation: `86d2d3053fce97d13cd8c514096eb97aef5ed5263afaf8edd0c0e4e01716d5f7`.

Le firme dell'intero esperimento possono differire perché includono rank e
alpha: ciò non implica una differenza del dataset.

## 3. Risultati della loss

Valori più bassi indicano minore errore di denoising. La media train considera
le 1000 loss registrate per aggiornamento; la media degli ultimi 100 riduce
la dipendenza da un singolo batch. Non sono accuratezze o percentuali di stile.

| Rank | Alpha | Update | Train loss media, tutti gli update | Train loss media, ultimi 100 | Validation loss finale | Miglior step validation osservato |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 1000 | 0.11355016 | 0.10819628 | 0.10638103 | 1000 |
| 8 | 8 | 1000 | 0.11345582 | 0.10804456 | 0.10620624 | 1000 |
| 16 | 16 | 1000 | 0.11330333 | 0.10781743 | 0.10603175 | 1000 |
| 32 | 32 | 1000 | 0.11316535 | 0.10760296 | 0.10583767 | 1000 |

### Validation loss ai quattro punti di controllo

| Step | Rank 4 | Rank 8 | Rank 16 | Rank 32 |
|---:|---:|---:|---:|---:|
| 250 | 0.10720498 | 0.10717017 | 0.10709836 | 0.10699720 |
| 500 | 0.10684496 | 0.10676949 | 0.10661920 | 0.10648724 |
| 750 | 0.10659285 | 0.10643210 | 0.10626911 | 0.10612103 |
| 1000 | 0.10638103 | 0.10620624 | 0.10603175 | 0.10583767 |

Per ogni rank la validation loss diminuisce nei quattro punti registrati.
Il minimo osservato è a step 1000 in tutti i casi: questa osservazione non
stabilisce che un training più lungo migliori lo stile o sia necessario.

La training loss non scende in modo monotono. Le medie su finestre di 250
aggiornamenti mostrano un aumento nella seconda finestra e successive
oscillazioni, presenti in tutti i rank:

| Intervallo update | Rank 4 | Rank 8 | Rank 16 | Rank 32 |
|---|---:|---:|---:|---:|
| 1–250 | 0.11301585 | 0.11299782 | 0.11295371 | 0.11290048 |
| 251–500 | 0.11763007 | 0.11755490 | 0.11740289 | 0.11728107 |
| 501–750 | 0.11137858 | 0.11124349 | 0.11104901 | 0.11087736 |
| 751–1000 | 0.11217615 | 0.11202709 | 0.11180771 | 0.11160250 |

Non si usa l'ultima train loss isolata per scegliere un rank: dipende dal
batch, dai latenti, dal rumore e dai timestep campionati. Le medie e la
validation fissata sono descrizioni più utili dell'andamento del training.

## 4. Parametri, dimensioni e costo osservato

| Rank | Parametri LoRA addestrabili | Rispetto a rank 4 | File pesi, MiB | Durata sessione, min:s | Secondi per update, media sessione | Picco CUDA allocato, GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 797,184 | 1× | 3.077 | 36:09 | 2.169 | 2.548 |
| 8 | 1,594,368 | 2× | 6.118 | 32:58 | 1.978 | 2.560 |
| 16 | 3,188,736 | 4× | 12.200 | 35:21 | 2.121 | 2.584 |
| 32 | 6,377,472 | 8× | 24.365 | 35:25 | 2.125 | 2.631 |

MiB = 2²⁰ byte; GiB = 2³⁰ byte. I tempi sono quelli di
`elapsed_seconds_this_session` nei riepiloghi, arrotondati al secondo;
comprendono il ciclo di training con validation e salvataggi. Il timer del
runner parte dopo il caricamento iniziale: non è il tempo totale comprensivo
di download, preparazione e caricamento dei modelli.

- I parametri aumentano linearmente con il rank: rank 32 ne ha **8 volte**
  quelli di rank 4. Le dimensioni dei file seguono approssimativamente lo
  stesso andamento, con un overhead di metadati.
- Il picco CUDA allocato passa da 2.548 a 2.631 GiB, circa **+3.26%**.
  È una misura delle allocazioni CUDA PyTorch, non di tutta la VRAM usata,
  della memoria riservata o della RAM di sistema. Non rappresenta il consumo MPS.
- Il rank 8 ha la durata più breve osservata. L'andamento dei tempi non è
  monotono con il rank: sono singole sessioni, non benchmark ripetuti in
  condizioni di carico controllate. Non dimostrano che rank 8 sia sempre
  più veloce o che aumentare il rank riduca il costo.

## 5. Integrità e comparabilità degli artefatti

Controlli completati sui quattro run:

- `status = completed` e `global_step = 1000` nei riepiloghi.
- 1000 righe metriche, con step 1–1000 contigui e senza duplicati.
- Train loss e learning rate finiti; quattro validation loss finite a
  250/500/750/1000. Tempi nel CSV non decrescenti.
- 256 tensori per adapter, tutti della U-Net, finiti e non interamente nulli.
- Dimensioni delle matrici LoRA coerenti con il rank; numero di parametri
  letto dai pesi uguale a quello dichiarato nel manifest del training.
- Stesse firme train/validation, revisione SD e versioni software; gli
  iperparametri registrati differiscono solo per rank/alpha e output_dir.

Le stringhe `/kaggle/working/...` nei JSON indicano il percorso di origine.
I rank 4 e 8 registrano entrambi `/kaggle/working/lora/pixel_art`, mentre
localmente sono in directory distinte; contenuto, dimensioni e rank degli
adapter sono stati verificati separatamente. Il percorso storico non va
usato come identificatore unico dell'esperimento.

Le verifiche sono di integrità e coerenza dei file. Non includono una nuova
prova di caricamento in Stable Diffusion, inferenza o valutazione visiva.
Il campo `git_commit` è nullo in tutti i manifest: non è possibile certificare
il commit del codice eseguito dai soli artefatti, anche se impostazioni e
versioni registrate coincidono.

### Ambiente registrato

| Componente | Versione comune |
|---|---|
| Python | 3.13.16 |
| torch | 2.11.0+cu130 |
| diffusers | 0.35.2 |
| transformers | 4.57.6 |
| accelerate | 1.15.0 |
| peft | 0.17.1 |
| safetensors | 0.8.0 |
| Pillow | 13.0.0 |
| numpy | 2.1.3 |

### SHA256 dei pesi finali

Questi hash identificano i file locali controllati, non una certificazione
esterna della loro origine.

| Rank | SHA256 di pytorch_lora_weights.safetensors |
|---:|---|
| 4 | `a02aed056bc46f8a876f408c53085374f005fedfd3fa54aea5e23d6550341710` |
| 8 | `9ec86bd42cd9c60653281dc3b26a4bd605ef0834e247cf53e99bc66740a520a9` |
| 16 | `7fb8a5ea6151718b20f5d395868af629b39593ed1b774d0fec1cb02204755322` |
| 32 | `2d67103d0edda4ec6feb927cac60656764b5309d99bb9a38be6adefe52f6aa53` |

## 6. Interpretazione per RQ2

A parità delle impostazioni registrate, tutti i quattro punti di validation
mostrano loss inferiori all'aumentare del rank. A step 1000 il passaggio da
rank 4 a rank 32 riduce la MSE da **0.10638103 a 0.10583767**. La differenza
relativa dello 0.51% è riferita soltanto alla MSE; non equivale a un aumento
percentuale della qualità delle immagini.

Le evidenze indicano che gli adapter sono stati addestrati e salvati
correttamente e che il rank maggiore adatta leggermente meglio l'obiettivo
di denoising sul validation set in queste run. **Non permettono ancora di
scegliere il rank migliore per il trasferimento Pixel Art**: occorre misurare
stile, preservazione del contenuto e artefatti sulle fotografie di inferenza.

Ogni rank ha una sola run con seed 42; non sono disponibili repliche del
training o stime della variabilità delle differenze. Non si attribuisce
significatività statistica al piccolo margine osservato e non si deduce
assenza di overfitting dalla sola diminuzione delle quattro validation loss.

## 7. Passo successivo: inferenza e valutazione appaiate

1. Usare gli adapter finali a step 1000 e gli stessi input, prompt, negative
   prompt, seed, preprocessing, strength, CFG e scala LoRA.
2. Generare una baseline separata **senza LoRA** e i quattro run con adapter.
3. Confrontare i rank a condizione fissa; Canny, depth e segmentation sono
   rami separati. `none` dentro un run con LoRA mantiene quel LoRA attivo.
4. Calcolare CLIP-I, CLIP-T, mIoU e DINO-I, affiancando griglie e analisi di
   folle, piccoli gruppi e figure lontane. La MSE di training non sostituisce
   queste verifiche.
5. Aggregare i confronti fra rank con un protocollo appaiato. Il riepilogo
   `paired_vs_none.csv` esistente confronta condizioni nello stesso run,
   non rank diversi o presenza/assenza di LoRA.

## 8. File sorgenti del report

I risultati sono ricavati dai file seguenti, letti senza modificarli:

| Rank | Configurazione | Riepilogo | Metriche |
|---:|---|---|---|
| 4 | [run_config.json](pixel_art_rank_4/run_config.json) | [training_summary.json](pixel_art_rank_4/training_summary.json) | [metrics_from_000000_1794227288831777144.csv](pixel_art_rank_4/metrics_from_000000_1794227288831777144.csv) |
| 8 | [run_config.json](pixel_art_rank_8/run_config.json) | [training_summary.json](pixel_art_rank_8/training_summary.json) | [metrics_from_000000_1794306906579984453.csv](pixel_art_rank_8/metrics_from_000000_1794306906579984453.csv) |
| 16 | [run_config.json](pixel_art_rank_16/run_config.json) | [training_summary.json](pixel_art_rank_16/training_summary.json) | [metrics_from_000000_1794309318165085036.csv](pixel_art_rank_16/metrics_from_000000_1794309318165085036.csv) |
| 32 | [run_config.json](pixel_art_rank_32/run_config.json) | [training_summary.json](pixel_art_rank_32/training_summary.json) | [metrics_from_000000_1794311467041779008.csv](pixel_art_rank_32/metrics_from_000000_1794311467041779008.csv) |

I pesi verificati sono `pytorch_lora_weights.safetensors` nelle stesse quattro
directory. I checkpoint intermedi non sono presenti nelle cartelle locali
ispezionate: le loss a step 250/500/750 sono documentate dai CSV, ma il
report non attesta la disponibilità dei rispettivi pesi intermedi.

