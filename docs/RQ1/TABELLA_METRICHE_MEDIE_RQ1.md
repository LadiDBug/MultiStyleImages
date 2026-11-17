# RQ1 — Tabella delle metriche medie

Medie sulle **104 foto comuni e sui seed 42, 123 e 456**. Si mediano prima i tre seed per foto, poi le 104 foto con uguale peso. Ogni combinazione stile/condizione rappresenta **312 output valutati**. Valori arrotondati a tre decimali; grassetto = media maggiore fra le quattro condizioni dello stesso stile e della stessa metrica.

## Modelli che calcolano le metriche

| Modello valutatore | Checkpoint | Metriche calcolate |
|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | CLIP-I: similarità originale/output; CLIP-T: allineamento output/testo dello stile |
| DINOv2 ViT-B/14 | `facebook/dinov2-base` | DINO-I: similarità visiva originale/output |
| SegFormer B2, ADE20K | `nvidia/segformer-b2-finetuned-ade-512-512` | mIoU: accordo fra segmentazioni predette di originale e output |

Revisioni esatte congelate nel [protocollo esportato da Kaggle](../../result_evaluation_rq1/evaluation_frozen.toml).

## Medie per modello valutato e stile

Tutte le condizioni usano **Stable Diffusion 1.5 img2img, senza LoRA**. `none` è la baseline senza ControlNet; gli altri tre rami aggiungono il rispettivo ControlNet con scala 0.8.

In ogni cella i valori seguono l'ordine **CLIP-I / CLIP-T / DINO-I / mIoU**.

| Condizione | GTA V | Pixel Art | Watercolor |
|---|---|---|---|
| none | 0.656 / **0.286** / 0.374 / 0.181 | 0.729 / **0.231** / 0.431 / 0.216 | 0.758 / **0.247** / 0.521 / 0.238 |
| Canny | **0.812** / 0.262 / **0.728** / **0.337** | **0.886** / 0.204 / **0.797** / **0.348** | **0.838** / 0.242 / **0.771** / **0.343** |
| Depth | 0.742 / 0.278 / 0.539 / 0.253 | 0.835 / 0.213 / 0.621 / 0.288 | 0.809 / 0.241 / 0.639 / 0.297 |
| Segmentation | 0.736 / 0.276 / 0.524 / 0.277 | 0.806 / 0.218 / 0.574 / 0.283 | 0.796 / 0.244 / 0.617 / 0.300 |

CLIP-I, DINO-I e mIoU descrivono aspetti della preservazione del contenuto; CLIP-T descrive l'allineamento al testo stilistico fissato. Le metriche non sono percentuali di qualità, e la media più alta non determina automaticamente il metodo migliore nel complesso. Il grassetto non implica un test di significatività.

I confronti vanno letti entro ciascuno stile. In particolare, CLIP-T non è una scala calibrata per confrontare la qualità assoluta di stili diversi.

**Risultato descrittivo:** Canny ha le medie maggiori nelle tre metriche di contenuto in tutti gli stili; `none` ha il CLIP-T medio maggiore. La lettura qualitativa serve a verificare come questo compromesso appare nelle immagini.

Fonte: [summary_metrics.csv](../../result_evaluation_rq1/analysis_common104_v1/summary_metrics.csv), che include anche deviazione standard, IC bootstrap al 95% e numerosità. Per le differenze appaiate e la discussione completa: [analisi RQ1](ANALISI_RISULTATI_RQ1_COMMON104.md).
