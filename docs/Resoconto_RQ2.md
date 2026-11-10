# Resoconto RQ2 — LoRA Pixel Art

**Aggiornato al 10 novembre 2026.** Sono disponibili e verificati i quattro
training **rank 4, 8, 16 e 32** sul dataset corrente. La valutazione del
trasferimento di stile sulle fotografie e distinta dai risultati del training.

## Obiettivo e metodo

Confrontare l'effetto del rank sul compromesso fra fedelta Pixel Art e
preservazione del contenuto: **senza LoRA, rank 4, 8, 16, 32**.
Si addestrano adattatori della U-Net SD 1.5, lasciando congelati VAE, text
encoder CLIP e pesi base U-Net; ControlNet entra soltanto in inferenza.

Alpha e uguale al rank in ogni run, per mantenere alpha/r = 1. La scala
esplicita e costante, ma capacita e dinamiche di ottimizzazione cambiano.
Non sono stati addestrati adattatori CLIP; l'assenza delle relative chiavi
nei pesi e prevista. Un eventuale training del text encoder sarebbe
un'ablation separata.

## Dataset

400 immagini con caption in `data/pixel_art/`: 360 train e 40 validation.
La selezione e composta da **300 originali congelate + 100 scene aggiuntive
con persone**, senza conteggiare due volte gli esempi con persone gia nelle
300. Le fonti sono 305 PixelParti, 25 Multi-Style e 70 MKS.

Le 100 aggiunte comprendono 10 folle, 42 piccoli gruppi e 48 scene con persona
principale; 24 hanno figure lontane, caratteristica trasversale. Descrizione,
fonti, revisioni e lista dei 400 file sono in
[Dataset_LoRA.md](../data/pixel_art/Dataset_LoRA.md). Il trainer legge immagini
e metadata.jsonl degli split, non la documentazione.

## Training completati

I quattro run condividono firme dati, revisione SD, iperparametri, versioni
software e GPU Tesla T4 FP16. Impostazioni comuni: seed 42, 512 nearest,
batch 1 con accumulo 4, AdamW LR 1e-4, 1000 update, validation ogni 250.
I moduli LoRA sono to_q, to_k, to_v e to_out.0.

| Rank | Alpha | Update | Parametri LoRA | Validation loss finale | Stato |
|---:|---:|---:|---:|---:|---|
| 4 | 4 | 1000 | 797184 | 0.10638103 | completed |
| 8 | 8 | 1000 | 1594368 | 0.10620624 | completed |
| 16 | 16 | 1000 | 3188736 | 0.10603175 | completed |
| 32 | 32 | 1000 | 6377472 | 0.10583767 | completed |

Il rank 32 ha la validation loss piu bassa, con una riduzione relativa di
circa **0.51%** rispetto al rank 4. Il dato riguarda denoising MSE, non
qualita Pixel Art o preservazione dei soggetti. Una run per rank non consente
di stimare la variabilita del training o la significativita del piccolo margine.

Sono stati verificati 1000 step contigui, loss finite e 256 tensori finiti
per adapter. Il rank 8 in queste cartelle usa il dataset corrente: e distinto
dal pilota storico su una selezione precedente. Il report completo contiene
tempi, memoria, andamento della loss, hash, versioni e limiti:
[Report_train.md](../lora/Report_train.md).

## Main e inferenza

```bash
# Verifica, senza addestramento
python3 main.py --lora lora_rank4.toml train --print-config
python3 main.py --lora lora_rank4.toml train --check-data

# Inferenza con un adattatore disponibile
python3 main.py --lora lora_rank4.toml inference
python3 main.py --lora lora_rank32.toml inference --canny
```

Le directory dei quattro rank contengono gia i pesi finali: non vanno
sovrascritte con nuovi training. Il comando train resta disponibile per nuovi
esperimenti con output distinti; il resume richiede checkpoint completi.
In training, dataset e pesi vengono dal config del rank. In inferenza quel
config seleziona pesi ed esperimento, mentre foto e risultati vengono da
configs/main.toml. La pipeline RQ1 senza adattatore resta disponibile.

## Confronto ancora da completare

I config RQ2 selezionano Pixel Art, seed 42 e quattro condizioni: none,
Canny, depth e segmentation, eseguite separatamente. La matrice completa
comprende 5 livelli di adattamento x 4 condizioni = 20 output per foto/seed.
Confrontare i rank con fotografie, prompt, seed e condizione identici.

`none` con un LoRA significa senza ControlNet ma con adattatore attivo.
La baseline senza LoRA richiede un run separato e parametri appaiati: il
main.toml RQ1 corrente ha tre stili e seed 123, quindi il suo avvio standard
non costituisce automaticamente una baseline RQ2 appaiata.

L'evaluator esistente calcola CLIP-I, CLIP-T, mIoU SegFormer e DINO-I per
singolo run. Il suo paired_vs_none.csv confronta condizioni nello stesso
run, non rank o presenza/assenza di LoRA. L'aggregatore RQ2 dedicato e ancora
da implementare: src/evaluation/lora_eval.py e vuoto.

Prossimi passi: fissare il protocollo di inferenza, completare i confronti
appaiati sulle fotografie, misurare contenuto/stile e analizzare gli errori
su folle, gruppi e figure lontane. Non si conclude quale rank sia migliore
dalla sola loss di training.

## Riferimenti

- [Protocollo RQ2 dettagliato](RQ2/RQ2_PIXEL_ART.md)
- [Architettura RQ1/RQ2](RQ2/ARCHITETTURA.md)
- [Roadmap operativa](RQ2/ROADMAP_MultiStyleImages_RQ1_RQ2.md)
- [Resoconto RQ1](Resoconto_RQ1.md)
- [Guida alla valutazione](RQ1/EVALUATION_RQ1.md)
- [Installazione e CLI](../README.md)


## Verifica del riordino e dipendenze

Documenti riorganizzati in docs/RQ1 e docs/RQ2; i materiali condivisi di
architettura e roadmap sono in RQ2 e collegati da entrambi i resoconti.
Requirements aggiornati per l'intera pipeline. Superati i 16 test offline;
verificati gli import e i vincoli diretti nell'ambiente esistente.
Il controllo globale pip rileva un conflitto preesistente fra datasets 5.1.0
e huggingface-hub: dettagli nel README principale. Non sono stati installati
o rimossi pacchetti, ne avviati nuovi training o benchmark.
