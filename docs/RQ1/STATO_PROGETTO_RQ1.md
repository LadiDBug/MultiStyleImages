# MultiStyleImages — Stato attuale RQ1 e RQ2

**Aggiornato al 10 novembre 2026.** Questo riepilogo prevale sui vecchi piani.
Il nome del file e mantenuto per compatibilita con i riferimenti esistenti.
Per una sintesi per domanda leggere [Resoconto RQ1](../Resoconto_RQ1.md)
e [Resoconto RQ2](../Resoconto_RQ2.md).
Le vecchie istruzioni RQ2 Watercolor e i percorsi dei dataset precedenti
non sono piu indicazioni operative.

## RQ1

- Stable Diffusion 1.5 **img2img**, senza LoRA, con condizioni `none`,
  `canny`, `depth`, `segmentation` e stili GTA V, Pixel Art e Watercolor.
- L'utente ha riferito l'esecuzione della pipeline completa con seed
  **42, 123 e 456**. Non rigenerare automaticamente il benchmark.
- La cronologia registra una prima prova end-to-end su `olanda.png`,
  con 12 combinazioni e quattro metriche. Questo pilot non dimostra
  conclusioni generali sul benchmark.
- La completezza degli artefatti di tutti i seed, l'analisi aggregata e la
  relazione RQ1 non sono state nuovamente certificate in questo aggiornamento.
- Watercolor resta uno stile RQ1. Il passaggio a Pixel Art per il LoRA
  riguarda **RQ2**, non una modifica retroattiva dei risultati RQ1.

## RQ2: LoRA Pixel Art

**Protocollo e dettagli correnti:** [RQ2_PIXEL_ART.md](../RQ2/RQ2_PIXEL_ART.md).

| Elemento | Stato |
|---|---|
| Stile LoRA | Pixel Art, non Watercolor |
| Confronto | Senza LoRA e rank 4/8/16/32, con alpha = rank |
| Dataset | 400 immagini: 300 originali congelate + 100 scene con persone |
| Split | 360 train, 40 validation |
| Percorso attivo | `data/pixel_art/train/` e `data/pixel_art/validation/` |
| Descrizione e selezione | `data/pixel_art/Dataset_LoRA.md`, con fonti e tabella dei 400 file |
| Parametri addestrati | Solo adattatori U-Net; VAE, CLIP e base U-Net congelati |
| Training | Implementato in `src/training_lora/`, accessibile dal main |
| Inferenza | Caricamento opzionale nella pipeline img2img condivisa |
| Rank 4 corrente | Training a 1000 step completato, artefatti verificati |
| Rank 8 storico | Pilota su dataset precedente, distinto dal rank 8 corrente |
| Rank 8 corrente / 16 / 32 | Tutti completati a 1000 update e verificati sul dataset corrente |
| Metriche | Evaluator per singolo run disponibile |
| Aggregatore fra rank | Ancora da implementare: `lora_eval.py` e vuoto |

### Dataset attuale

Revisionato dopo il pilota rank 8 per aumentare le persone con contesto:
305 PixelParti, 25 Multi-Style e 70 immagini MKS. Nel sottoinsieme aggiuntivo
sono presenti 10 folle, 42 piccoli gruppi e 48 scene con persona principale;
24 hanno figure lontane, come caratteristica trasversale. Le 300 immagini
originali non sono state modificate e non sono conteggiate per quota 100.

Il 10 novembre il dataset e stato ripulito e la descrizione consolidata in
[Dataset_LoRA.md](../../data/pixel_art/Dataset_LoRA.md), che conserva fonti,
revisioni, criteri, firme e tabella completa dei file. La cartella docs/datasets/
non e richiesta dal trainer ed e stata rimossa su richiesta dell'utente,
con copia storica ZIP nella cache locale. Il trainer continua a usare solo
immagini e metadata.jsonl; immagini, caption, split e firme non sono cambiati.

### Evidenza rank 4 verificata il 10 novembre

In `lora/pixel_art_rank_4/` sono presenti pesi finali, configurazione, CSV e
riepilogo `completed` a step 1000. Le firme train e validation corrispondono
al dataset corrente. Rank/alpha 4/4, seed 42, 512 nearest, batch effettivo 4,
LR 1e-4, Tesla T4 FP16; 797184 parametri LoRA e 256 tensori finiti.
CSV con step 1–1000 contigui e training loss finite; validation loss finale
0.106381. Durata circa 36 min 9 s e picco CUDA allocato circa 2.55 GiB.

Questi artefatti attestano un training concluso, non la riuscita stilistica.
Non sono stati lanciati training o nuove generazioni in questo aggiornamento.
Il rank 8 del 4 novembre aveva una selezione diversa, nonostante lo stesso
totale di 400 immagini: non usarlo come confronto diretto del rank.

## Comandi e codice

```bash
python3 main.py --lora lora_rank4.toml train --print-config
python3 main.py --lora lora_rank4.toml train --check-data
python3 main.py --lora lora_rank4.toml train
python3 main.py --lora lora_rank4.toml inference
python3 main.py
```

L'ultimo comando avvia l'esperimento senza adattatore. Per rank 8/16/32
cambiare soltanto il nome del config corrispondente, a protocollo fissato.
In inferenza il config LoRA determina pesi ed esperimento; foto e risultati
rimangono quelli di `configs/main.toml`. Il training usa selezione automatica
del device, con FP32 su MPS/CPU; il device di inferenza e quello di `base.toml`.

La verifica precedente dell'integrazione del main ha superato 24 test offline
e l'audit 360/40. Non equivale a un benchmark dei modelli preaddestrati.
Gli script del vecchio prototipo e alcuni builder citati nella cronologia
sono stati rimossi dal checkout; non usarli come entry point corrente.

## Prossimo lavoro

1. Conservare i quattro training verificati e il relativo
   [report](../../lora/Report_train.md); proseguire con il
   [resoconto RQ2 aggiornato](../Resoconto_RQ2.md).
2. Generare confronti appaiati senza adattatore e con i rank, a parita di
   foto, prompt, seed e condizione. Non e stato scelto definitivamente un
   ControlNet primario unico; i config supportano tutti e quattro i rami.
3. Implementare aggregazione RQ2 e controlli di comparabilita fra run.
4. Analizzare contenuto e stile con metriche e ispezione visiva; non dedurre
   qualita dalla sola loss. Completare tabelle, limiti e relazione RQ1/RQ2.

## Riferimenti

- [Architettura del codice](../RQ2/ARCHITETTURA.md)
- [Roadmap operativa](../RQ2/ROADMAP_MultiStyleImages_RQ1_RQ2.md)
- [Valutazione delle immagini](EVALUATION_RQ1.md)
- [README principale](../../README.md)
