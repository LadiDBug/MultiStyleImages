# RQ2 — LoRA Pixel Art: protocollo e stato attuale

Aggiornato al **10 novembre 2026** in base a codice, config, dataset e artefatti
locali. Questo documento sostituisce il precedente piano RQ2 Watercolor.
Watercolor rimane uno degli stili di RQ1; non e lo stile dei LoRA RQ2.

## 1. Domanda e confronto

**Come influisce il rank LoRA sul compromesso fra fedelta Pixel Art e
preservazione del contenuto nella generazione img2img con controllo strutturale?**

Il confronto previsto comprende **senza LoRA, rank 4, 8, 16 e 32** su Stable
Diffusion 1.5. I config attuali selezionano `pixel_art`, seed di inferenza
`[42]` e condizioni `none`, `canny`, `depth`, `segmentation`.

Si confrontano i rank **a parita di condizione**, fotografia, prompt e seed.
Le tre mappe sono rami distinti, non un Multi-ControlNet. Se si eseguono tutti
i 5 livelli di adattamento e tutte le 4 condizioni si ottengono 20 output per
foto e seed. La designazione di un'eventuale condizione primaria unica resta
una decisione da esplicitare prima dell'analisi; non e stato fissato Semantic
come scelta definitiva. Non selezionarla a posteriori in base al risultato.

`none` significa senza ControlNet. In un run con LoRA anche `none` usa
l'adattatore; il modello senza LoRA richiede un run separato. Non esiste
una LoRA di rank zero nel nostro protocollo.

## 2. Dataset congelato

Percorso attuale: **`data/pixel_art/`**, contenente soltanto gli elementi
necessari al trainer:

```text
data/pixel_art/
├── train/           # 360 immagini + metadata.jsonl
└── validation/      # 40 immagini + metadata.jsonl
```

| Sottoinsieme | Train | Validation | Totale |
|---|---:|---:|---:|
| PixelParti originali congelate | 270 | 30 | 300 |
| Scene aggiuntive con persone | 90 | 10 | 100 |
| Totale | 360 | 40 | 400 |

Le 100 scene aggiuntive sono 25 Multi-Style, 5 ulteriori PixelParti e
70 MKS. In totale le fonti sono **305 PixelParti, 25 Multi-Style, 70 MKS**.
Gli esempi con persone gia presenti nelle 300 originali non vengono
conteggiati di nuovo per raggiungere quota 100.

Nelle 100 scene: 10 folle, 42 piccoli gruppi, 48 con persona principale;
24 hanno figure lontane/inquadrature ampie, caratteristica trasversale.
Le etichette sono valutazioni visive qualitative. Le immagini hanno stili
e risoluzioni misti: PixelParti 128x128, MKS 1024x1024, Multi-Style in JPEG
quadrati o rettangolari. MKS include personaggi illustrativi e soggetti di
ispirazione coreana. Non e dimostrato che la selezione garantisca uno stile
uniforme o un miglioramento del modello.

Le caption descrivono contenuto e Pixel Art; **non usiamo `wtrclrstyle`** o
un nuovo token di stile. Il trainer non richiede coppie fotografia/target:
usa immagini gia stilizzate con caption. Le foto del benchmark di inferenza
restano separate e la validation non aggiorna i pesi.

Fonti, revisioni, licenze dichiarate, criteri e tabella completa dei 400 file
sono in [Dataset_LoRA.md](../../data/pixel_art/Dataset_LoRA.md). Il trainer
legge soltanto immagini e `metadata.jsonl` degli split. La documentazione
di supporto precedentemente in docs/datasets/ e stata consolidata e rimossa
dal percorso operativo, con copia storica nella cache locale.

Firme del loader della revisione `people100-plus-frozen300-2026-11-06`:

- train: `87246eb9966643cc01862bee1d17a19161b9fbaef500dc1d42aa0d98b6798ae4`;
- validation: `86d2d3053fce97d13cd8c514096eb97aef5ed5263afaf8edd0c0e4e01716d5f7`.

Le firme sono rimaste invariate durante la pulizia del dataset. I controlli
RGB/dHash non escludono ogni duplicato semantico o identita ripetuta, ne
sostituiscono un audit esaustivo di overlap con le foto RQ1.

## 3. Training implementato

Codice: `src/training_lora/dataset.py` e `train.py`; adattamento dei config
nel main tramite `src/utils/lora_config.py`.

| Aspetto | Implementazione / default attuale |
|---|---|
| Backbone | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Revisione del rank 4 disponibile | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| Parametri addestrati | Solo LoRA della U-Net |
| Componenti congelati | VAE, text encoder CLIP, pesi originali U-Net |
| Moduli LoRA | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Rank / alpha | 4/4, 8/8, 16/16, 32/32 |
| Preprocessing | EXIF/RGB, resize proporzionale nearest, crop centrale 512x512 |
| Batch | 1, accumulo 4, batch effettivo 4 |
| Budget | 1000 aggiornamenti optimizer |
| Optimizer | AdamW, LR 1e-4, weight decay 0.01 |
| Scheduler LR | `constant_with_warmup`, warmup 50 |
| Seed training | 42 nei config attuali |
| Checkpoint / validation | Ogni 250 update e a fine run |
| Loss | MSE diffusion, target epsilon o v_prediction secondo lo scheduler |
| Validation | Denoising con rumore/timestep fissati; nessuna generazione immagini |

Con LoRA standard, `alpha = rank` mantiene `alpha/r = 1` fra i rank.
Non garantisce aggiornamenti della stessa ampiezza o dinamiche equivalenti;
la convenzione fa parte del protocollo, non e una dimostrazione di ottimalita.

Accelerate seleziona automaticamente CUDA, MPS o CPU. Il codice usa FP32 su
MPS/CPU e mixed precision richiesta su CUDA. Confrontare tempi e memoria
soltanto annotando hardware, precisione effettiva e versioni; non attribuire
al rank differenze dovute anche al passaggio T4/MPS. Non e implementata una
cache dei latenti VAE o degli embedding testuali.

Il text encoder CLIP non riceve un adattatore. Il warning di caricamento che
segnala l'assenza di chiavi LoRA `text_encoder` e quindi previsto per questi
export; non significa che CLIP manchi dal modello. Aggiungere un LoRA al
text encoder sarebbe un esperimento separato, non parte del confronto attuale.
Il CLIP usato nelle metriche e un valutatore distinto.

## 4. Comandi attuali

Dalla radice del repository, con l'ambiente attivo:

```bash
# Config effettiva e controllo dei dati, senza training
python3 main.py --lora lora_rank4.toml train --print-config
python3 main.py --lora lora_rank4.toml train --check-data

# Training
python3 main.py --lora lora_rank4.toml train

# Ripresa solo se esiste un checkpoint compatibile nella directory del rank
python3 main.py --lora lora_rank4.toml train --resume-from-checkpoint latest

# Inferenza con il LoRA addestrato
python3 main.py --lora lora_rank4.toml inference
python3 main.py --lora lora_rank4.toml inference --canny --lora-scale 1.0

# Esperimento RQ1 configurato, senza adattatore
python3 main.py
```

Per gli altri rank usare `lora_rank8.toml`, `lora_rank16.toml` e
`lora_rank32.toml`. `--lora-path` resta disponibile per il caricamento diretto.
`--dataset-dir` permette un override del percorso senza modificare i config.

In training, il config LoRA usa `data.input_dir` come dataset e
`data.output_dir` come directory dei pesi (`lora/pixel_art_rank_4` per rank 4).
In inferenza seleziona pesi ed `experiment`; le foto provengono invece da
`configs/main.toml`, e i risultati vanno nell'output definito da quel file.
Prompt, generazione, checkpoint ControlNet e device di inferenza restano nei
config condivisi. La selezione automatica del trainer non cambia il device
di inferenza di `base.toml`.

Una directory di training non vuota non viene sovrascritta. Per un nuovo
esperimento usare un output distinto; `resume` non serve per iniziare da
zero e richiede checkpoint completi, non il solo file finale safetensors.

## 5. Evidenze disponibili al 10 novembre

### Quattro rank completati sul dataset corrente

Sono ora presenti e verificati i training rank **4, 8, 16 e 32**, tutti a
1000 update sulle stesse firme dati, revisione SD e impostazioni registrate.
Il rank 8 attuale e distinto dal vecchio pilota. Per risultati, tempi,
memoria, hash e verifiche vedere [Report_train.md](../../lora/Report_train.md)
e il [resoconto RQ2](../Resoconto_RQ2.md).

| Rank | Validation loss finale | Parametri LoRA |
|---:|---:|---:|
| 4 | 0.10638103 | 797184 |
| 8 | 0.10620624 | 1594368 |
| 16 | 0.10603175 | 3188736 |
| 32 | 0.10583767 | 6377472 |

Il vantaggio relativo di MSE del rank 32 sul rank 4 e circa 0.51%: non
certifica una migliore qualita delle immagini. Le run non sono repliche
statistiche e il commit del codice non e registrato nei manifest.

### Dettaglio della verifica rank 4

Artefatti in [lora/pixel_art_rank_4](../../lora/pixel_art_rank_4):
`run_config.json`, CSV metriche, `training_summary.json` e pesi finali.

- Rank/alpha 4/4, 1000 update, 360/40 immagini; firme dati uguali alla
  revisione corrente. Le vecchie stringhe `/kaggle/working/...` nei JSON
  descrivono l'ambiente di origine, non un errore del trasferimento locale.
- GPU Tesla T4, FP16; 797184 parametri addestrabili, 256 tensori esportati
  della U-Net. Tutti i tensori controllati sono finiti.
- CSV: 1000 step contigui, training loss finite. Validation loss a
  250/500/750/1000: 0.107205 / 0.106845 / 0.106593 / 0.106381.
- Stato finale `completed`; durata circa 36 min 9 s; picco CUDA allocato
  circa 2.55 GiB. Questi valori non sono misure di prestazione del Mac M1.

Questa verifica riguarda file, firma dati e integrita numerica; non dimostra
superiorita stilistica o preservazione del contenuto. Non sono state eseguite
nuove generazioni o aggiornamenti dei pesi per redigere questo documento.

### Rank 8 precedente: pilota storico, altro dataset

La cronologia documenta un training rank 8 a 1000 step e una coppia di
inferenza su una fotografia di paesaggio. Quel training usava la prima
selezione da 400 immagini, con 80 ritratti Multi-Style, prima delle revisioni.
**Stesso numero di immagini non significa stessi dati:** il pilota non puo
essere trattato come rank 8 del confronto sul dataset attuale. Il nuovo
rank 8 disponibile e stato verificato sulla selezione corrente. I vecchi artefatti e script citati
nella cronologia non sono tutti presenti nel checkout attuale.

Per i quattro rank correnti sono stati verificati artefatti completi,
non soltanto la presenza dei config.

## 6. Valutazione e lavoro restante

L'evaluator esistente calcola CLIP-I, CLIP-T, mIoU con **SegFormer** e DINO-I
su un singolo run. UPerNet e invece il preprocessore di conditioning semantico.
La validation loss del training non sostituisce queste metriche o l'ispezione
visiva. La fedelta stilistica non e una percentuale ricavabile direttamente
da CLIP-T.

I CSV `paired_vs_none.csv` confrontano condizioni dentro lo stesso run:
se e caricato un LoRA, anche `none` lo usa. **Non sono confronti appaiati
contro il modello senza LoRA o contro gli altri rank.**

`src/evaluation/lora_eval.py` e vuoto: l'aggregatore RQ2 fra run e rank,
con controlli espliciti di dati, training e adattatore, resta da implementare.
Il device automatico dell'evaluator e CUDA/CPU, non MPS.

Prossimi passi:

1. Conservare i quattro training completati e le firme del dataset; i
   controlli di parita sono riportati in Report_train.md.
2. Fissare il protocollo di inferenza senza usare le foto di test per
   scegliere checkpoint o iperparametri e senza sovrascrivere i pesi esistenti.
3. Generare run distinti senza LoRA e con ciascun rank, su stesse foto,
   condizioni, prompt e seed; fissare e registrare la scala LoRA.
4. Implementare i confronti appaiati fra rank; aggiungere griglie e analisi
   degli errori su folle, gruppi e persone lontane, senza selezionare solo
   casi favorevoli.
5. Redigere risultati e limiti; la valutazione umana cieca rimane da pianificare.

Riferimenti locali: [architettura](ARCHITETTURA.md),
[stato del progetto](../RQ1/STATO_PROGETTO_RQ1.md),
[roadmap](ROADMAP_MultiStyleImages_RQ1_RQ2.md),
[guida alle metriche](../RQ1/EVALUATION_RQ1.md).
