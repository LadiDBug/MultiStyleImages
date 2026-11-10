# MultiStyleImages — Roadmap operativa RQ1/RQ2

**Aggiornata al 10 novembre 2026.** RQ2 usa **Pixel Art**. Questa roadmap
sostituisce il piano Watercolor e le vecchie attivita di creazione del trainer.

Per parametri, evidenze e comandi completi vedere [RQ2 Pixel Art](RQ2_PIXEL_ART.md);
per lo stato verificato vedere [STATO_PROGETTO_RQ1.md](../RQ1/STATO_PROGETTO_RQ1.md).

## 1. Domande scientifiche

**RQ1 — Structural conditioning:** effetto di `none`, Canny, depth e
segmentation sul compromesso contenuto/stile, nei tre stili GTA V,
Pixel Art e Watercolor. Backbone e sistemi preaddestrati; nessun training.
L'utente riferisce run completi con seed 42/123/456. Verificare gli artefatti
prima di dichiarare conclusa l'analisi; non rigenerare il benchmark automaticamente.

**RQ2 — LoRA rank:** effetto dei rank 4/8/16/32, rispetto al modello senza
adattatore, sul trasferimento Pixel Art. Si addestrano solo LoRA della U-Net
SD 1.5, lasciando congelati VAE, text encoder e pesi base.

## 2. Percorso implementato

```text
Training:
Pixel Art + caption -> dataset.py -> VAE e CLIP congelati
-> latenti rumorosi -> U-Net con LoRA -> MSE -> optimizer dei soli LoRA
-> checkpoint, log e pytorch_lora_weights.safetensors

Inferenza:
Fotografia -> preprocessing -> none / Canny / depth / segmentation
-> SD 1.5 img2img con ControlNet opzionale e LoRA opzionale
-> immagini, mappe e metadati -> evaluator separato
```

Non si allena ControlNet e non si usano le fotografie del benchmark nel
training. La baseline riceve comunque la fotografia img2img. I rami usano
checkpoint ControlNet differenti; le conclusioni riguardano i sistemi di
conditioning, non solo il formato astratto della mappa.

## 3. Checklist con stato effettivo

### A — Dataset e struttura

- [x] Separati generazione, preprocessing, training, utilita config e metriche.
- [x] Main con `--lora CONFIG train` e `--lora CONFIG inference`.
- [x] Config `lora_rank4.toml`, `lora_rank8.toml`, `lora_rank16.toml`, `lora_rank32.toml`.
- [x] Dataset corrente: 400 immagini, 360 train e 40 validation.
- [x] 300 immagini originali congelate + 100 scene aggiuntive con persone.
- [x] Fonti: 305 PixelParti, 25 Multi-Style, 70 MKS.
- [x] Verificati caricamento, caption, integrita e split; documentati i limiti.
- [x] Dataset in `data/pixel_art/`: split train/validation e `Dataset_LoRA.md` con fonti e inventario completo.
- [ ] Audit esaustivo di identita/duplicati semantici e overlap col benchmark.

### B — Training e integrazione

- [x] Trainer funzionante, revisioni e versioni registrate, soli LoRA addestrabili.
- [x] Checkpoint riprendibili con optimizer, scheduler, RNG e firma dati.
- [x] Moduli target fissi: `to_q`, `to_k`, `to_v`, `to_out.0`.
- [x] Convenzione alpha = rank; batch 1 con accumulo 4, 512 nearest, seed 42.
- [x] Pilota rank 8 storico completato, su dataset precedente.
- [x] Rank 4 corrente completato a 1000 update e artefatti verificati.
- [x] Rank 8 riaddestrato sul dataset corrente e verificato.
- [x] Rank 16 completato e verificato.
- [x] Rank 32 completato e verificato.

I quattro training completati mantengono dati/caption/split, revisione SD,
budget 1000 update, LR 1e-4, batch effettivo 4, optimizer/scheduler e moduli
uguali. Risultati e verifiche in [Report_train.md](../../lora/Report_train.md). La convenzione alpha/r = 1 non garantisce
costo computazionale o dinamiche di apprendimento identici.
I default completi e i risultati numerici verificati sono in RQ2_PIXEL_ART.md.

Il rank 4 disponibile proviene da Tesla T4 FP16. Una run MPS usa FP32 nel
codice attuale: registrare questa differenza e non trattare i tempi come
misura isolata dell'effetto del rank. Non scegliere nuovi iperparametri sul test.

### C — Inferenza appaiata RQ2

- [x] Caricamento LoRA opzionale, compatibilita modello/rank e hash nei metadati.
- [x] Pipeline senza LoRA utilizzabile per RQ1 e baseline RQ2.
- [ ] Fissare protocollo finale di inferenza: foto, prompt, seed, scala LoRA,
  preprocessing, strength, CFG e scheduler.
- [ ] Generare senza LoRA e con i rank disponibili sulle stesse fotografie.
- [ ] Coprire Canny, depth e segmentation separatamente, come richiesto;
  i config attuali includono anche `none`.
- [ ] Esplicitare prima dell'analisi se il confronto principale riguarda
  ciascuna condizione o una condizione primaria unica. Nessuna scelta
  definitiva di Semantic e registrata come obbligo attuale.

Matrice prevista: 5 livelli di adattamento (off, 4, 8, 16, 32) x 4 condizioni
= 20 immagini per foto/seed. Confrontare i rank a condizione fissa. Il seed
attuale dei config RQ2 e 42; gli altri seed RQ1 non vengono ereditati
implicitamente. Non chiamare `none` di un run con LoRA baseline senza LoRA.

### D — Evaluation e analisi

- [x] Evaluator separato per singolo run con audit e metadati dei valutatori.
- [x] CLIP-I, CLIP-T, mIoU semantica con SegFormer e DINO-I.
- [x] Aggregazione a pari peso per foto, dopo media dei seed, e bootstrap per foto.
- [ ] Aggregatore appaiato RQ2 fra run e rank: `src/evaluation/lora_eval.py` e vuoto.
- [ ] Verifica automatica fra rank di firme dati, revisioni, impostazioni e foto.
- [ ] Tabelle e grafici del compromesso stile/contenuto, senza punteggio unico arbitrario.
- [ ] Griglie rappresentative di successo e fallimento, comprese folle e figure lontane.
- [ ] Eventuale valutazione umana cieca, con protocollo da fissare.
- [ ] Completare risultati, discussione, limiti e relazione RQ1/RQ2.

Il valutatore semantico e **SegFormer**, distinto da UPerNet usato nelle mappe.
`paired_vs_none.csv` confronta condizioni nello stesso run, non rank o LoRA
on/off. Le metriche e la loss sono proxy: nessuna garantisce da sola fedelta
stilistica o identita dei soggetti.

### E — Estensioni dopo il confronto principale

- [ ] Eventuale ablation della scala ControlNet o della scala LoRA su sviluppo.
- [ ] Eventuale confronto U-Net soltanto vs U-Net + LoRA text encoder, separato
  dall'ablation dei rank corrente. Non e necessario per eliminare il warning CLIP.
- [ ] Demo solo dopo stabilizzazione del protocollo e disponibilita dei pesi.

## 4. Comandi operativi

```bash
python3 main.py --lora lora_rank4.toml train --print-config
python3 main.py --lora lora_rank4.toml train --check-data
python3 main.py --lora lora_rank8.toml train
python3 main.py --lora lora_rank4.toml inference
python3 -m src.evaluation.evaluate --run-dir outputs/NOME_RUN
```

I comandi sono esempi; i quattro training disponibili non vanno sovrascritti. Il resume
richiede un checkpoint completo compatibile, non soltanto l'export finale.
Le foto e i risultati di inferenza vengono da `configs/main.toml`; dataset e
pesi di training dal config del rank. Nessun TOML e modificato da questa roadmap.

## 5. Priorita della prossima sessione

1. Verificare prima di ogni training che dataset e configurazione siano
   quelli confrontabili con rank 4. Conservare log e pesi gia disponibili.
2. Fissare le impostazioni e preparare i run di inferenza per tutti i rank.
3. Eseguire la matrice di inferenza fissata e registrare tutti gli adattatori.
4. Implementare confronto RQ2 e analizzare i risultati; nessun automatismo
   deve trasformare la presenza di un config in un esperimento completato.
