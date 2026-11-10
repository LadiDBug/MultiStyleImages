# Resoconto RQ1 — Condizionamento strutturale

**Aggiornato al 10 novembre 2026.** Questo documento riassume implementazione,
prove note e lavoro restante. Non certifica nuovi risultati del benchmark.

## Obiettivo

Studiare come cambia il compromesso tra preservazione del contenuto e
fedelta allo stile usando quattro condizioni: **none, Canny, depth,
segmentation**, nei tre stili **GTA V, Pixel Art e Watercolor**.
Il backbone e Stable Diffusion 1.5 img2img, senza adattatori LoRA per RQ1.

## Pipeline implementata

1. `main.py` legge i TOML condivisi e trova ricorsivamente le fotografie.
2. Prepara l'immagine in RGB con resize/crop, oppure usa `--no-preprocess`.
3. Produce la mappa Canny, la profondita MiDaS/DPT o la segmentazione
   UPerNet ADE20K, secondo la condizione. `none` non usa mappe o ControlNet.
4. Genera con SD 1.5 img2img e il checkpoint ControlNet specifico del ramo,
   iterando su foto, stili e seed. Le mappe non sono combinate fra loro.
5. Salva input, mappe, output, configurazione e metadati per immagine.
6. Un comando separato esegue audit e metriche, senza rigenerare le immagini.

Ogni ramo usa un sistema di conditioning diverso, comprensivo di preprocessore
e checkpoint: i risultati non isolano causalmente il solo formato della mappa.

## Configurazione locale corrente

| Aspetto | Valore |
|---|---|
| Config esperimento | `configs/main.toml` |
| Stili | gta5, pixel_art, watercolor |
| Condizioni | none, canny, depth, segmentation |
| Seed selezionato nel TOML attuale | 123 |
| Foto / risultati | `data/test_images` / `outputs/` |
| Backbone | stable-diffusion-v1-5/stable-diffusion-v1-5 |
| Risoluzione / step | 512x512 / 50 |
| CFG / strength | 7.5 / 0.5 |
| Scala ControlNet | 0.8 |
| Device / precisione di inferenza configurati | cuda / float16 |

Il TOML rappresenta una selezione di avvio, non l'inventario di tutti i run.
Con tre stili e quattro condizioni si hanno 12 output per foto e seed.

## Valutazione disponibile

| Metrica | Cosa confronta |
|---|---|
| CLIP-I | Similarita globale fra input e output |
| CLIP-T | Output e testo di valutazione dello stile |
| mIoU semantica | Mappe input/output prodotte da SegFormer |
| DINO-I | Similarita visiva input/output con DINOv2 |

SegFormer e distinto dall'UPerNet usato nel conditioning. Le medie sono
calcolate su blocchi completi foto/stile/seed; prima si mediano i seed per
foto, poi le foto, con intervalli bootstrap per foto. Il confronto appaiato
`paired_vs_none.csv` riguarda le condizioni dentro un singolo run.

## Prove e risultati noti

- L'utente ha riferito l'esecuzione della pipeline completa con seed
  **42, 123 e 456**. Non e stata richiesta o eseguita una nuova generazione.
- La cronologia documenta una prima prova end-to-end su `olanda.png`,
  con 12 combinazioni e quattro metriche, utile a verificare il flusso tecnico.
- La disponibilita e completezza locale degli artefatti di tutti i seed,
  l'aggregazione finale e le conclusioni sul benchmark non sono certificate
  da questo riordino della documentazione.
- Il pilot su una sola foto non basta a stabilire la superiorita di una
  condizione; CLIP-T non e una percentuale di riuscita dello stile.

## Comandi

```bash
python3 main.py
python3 main.py --canny
python3 main.py --prepare-canny-only
python3 -m src.evaluation.evaluate --run-dir outputs/NOME_RUN
```

Il device di generazione resta quello di `base.toml`; l'evaluator sceglie
CUDA o CPU. La selezione automatica CUDA/MPS/CPU del trainer RQ2 e separata.

## Lavoro restante

Verificare gli inventari dei run esistenti, consolidare tabelle e confronti
appaiati, analizzare successi/fallimenti e fedelta stilistica, quindi completare
risultati, discussione e limiti. Non rilanciare automaticamente il benchmark
per colmare un'assenza nella documentazione.

## Documenti e collegamenti

- [Guida alle metriche e all'audit](RQ1/EVALUATION_RQ1.md)
- [Stato dettagliato e contesto](RQ1/STATO_PROGETTO_RQ1.md)
- [Riferimento storico RQ1](RQ1/RQ1_PROJECT_REFERENCE.md)
- [README storico](RQ1/README_STORICO.md)
- [Architettura condivisa con RQ2](RQ2/ARCHITETTURA.md)
- [Roadmap condivisa](RQ2/ROADMAP_MultiStyleImages_RQ1_RQ2.md)
- [Resoconto RQ2](Resoconto_RQ2.md)
- [Struttura e installazione](../README.md)


## Verifica del riordino e dipendenze

Documenti riorganizzati in docs/RQ1 e docs/RQ2; i materiali condivisi di
architettura e roadmap sono in RQ2 e collegati da entrambi i resoconti.
Requirements aggiornati per l'intera pipeline. Superati i 16 test offline;
verificati gli import e i vincoli diretti nell'ambiente esistente.
Il controllo globale pip rileva un conflitto preesistente fra datasets 5.1.0
e huggingface-hub: dettagli nel README principale. Non sono stati installati
o rimossi pacchetti, ne avviati nuovi training o benchmark.
