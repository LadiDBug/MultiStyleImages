# RQ1 — Analisi aggregata delle 104 foto comuni

**Analisi del 12 novembre 2026. Stato: analisi quantitativa completata; ispezione visiva e valutazione umana ancora da fare.**

Canny ha le medie più alte in tutte e tre le metriche di contenuto, in ciascuno dei tre stili. La baseline `none` ha il CLIP-T medio più alto. I dati descrivono quindi un **compromesso fra similarità al contenuto e allineamento al testo stilistico**, non un vincitore assoluto di qualità. Queste sono misure automatiche: non abbiamo ancora verificato se le differenze corrispondano a immagini stilisticamente convincenti e prive di artefatti.

## 1. Che cosa abbiamo misurato e verificato

- Stessi **104 input** per ogni stile, condizione e seed; **3.744 output**, tutti valutati con successo su Kaggle.
- Tre stili: GTA V, Pixel Art e Watercolor; quattro condizioni: `none`, Canny, depth e segmentation; tre seed: 42, 123, 456.
- Nessun duplicato, combinazione assente o valore non finito nelle quattro metriche. Punteggi negli intervalli teorici. Hash input corrispondenti alla selezione congelata e hash output corrispondenti alla provenienza esportata.
- Stesso protocollo, processori e revisioni risolte dei valutatori tra seed; stessa configurazione di generazione dichiarata e geometrie coerenti per input. Originali ed export Kaggle non modificati.
- Escluse uniformemente `0600x2.png`, `lsdir_0149.png`, `lsdir_0405.png`, `lsdir_0411.png` per incompletezza degli output originali, prima di leggere le metriche. Nessuna rigenerazione dei mancanti.

I file sorgenti sono in [`result_evaluation_rq1/`](../../result_evaluation_rq1/). Tabelle, figure e manifest riproducibile sono in [`analysis_common104_v1/`](../../result_evaluation_rq1/analysis_common104_v1/). Per i dettagli di selezione e cronologia leggere il [piano dello studio](PIANO_ABLATION_STUDY_RQ1.md).

**Configurazione effettiva:** SD 1.5 img2img senza LoRA, scheduler registrato PNDM, 50 step, CFG 7.5, strength 0.5 e scala ControlNet 0.8. Il preprocessing fotografico esplicito era disabilitato. Le dimensioni effettive non sono fisse a 512 × 512: gli output hanno larghezze 784–1248 e altezze 424–1016 pixel, con piccoli arrotondamenti rispetto agli originali. Non descrivere questo benchmark come generato interamente a 512 × 512. SegFormer usa la propria griglia di evaluation con lato lungo massimo 512.

## 2. Come sono calcolate le tabelle

Per ogni foto/stile/condizione mediamo i tre seed. Poi mediamo le **104 foto con uguale peso**. Per i contrasti sottraiamo prima le due condizioni sulla stessa foto e seed e poi mediamo. Ogni metrica per stile/condizione deriva da 312 output, ma l'unità indipendente del bootstrap è la foto, non il singolo output.

Gli IC al 95% sono percentile bootstrap con 2.000 repliche e seed 2026. A ogni replica si ricampionano 104 foto con rimpiazzo, mantenendo insieme tutti i seed e le condizioni. Gli stessi indici sono riusati per metriche e contrasti. Le tabelle riportano anche deviazione standard, mediana, quantili 5% e 95% delle medie per foto e frazioni di delta positivi/negativi.

**Analisi descrittiva:** 72 stime di differenza complessive (6 contrasti × 3 stili × 4 metriche). Non sono stati eseguiti test di ipotesi né correzioni per confronti multipli; gli IC sono marginali, non simultanei. Non chiamare automaticamente "significativo" ogni intervallo che esclude lo zero. Non usiamo una somma arbitraria delle metriche, soglie di successo a posteriori o percentuali di "contenuto conservato". La variabilità stimata è tra le foto del benchmark, con tre seed fissati; non include ogni possibile seed o revisione del modello.

## 3. Medie aggregate: dove si collocano i metodi

Ogni riga usa le stesse 104 foto e gli stessi tre seed. Valori maggiori indicano più similarità/allineamento nella rispettiva metrica, non necessariamente qualità complessiva maggiore. Gli IC delle medie sono nel CSV completo.

| Stile | Condizione | CLIP-I | DINO-I | mIoU | CLIP-T |
|---|---|---:|---:|---:|---:|
| gta5 | none | 0.6559 | 0.3741 | 0.1808 | 0.2865 |
| gta5 | canny | 0.8116 | 0.7283 | 0.3366 | 0.2623 |
| gta5 | depth | 0.7416 | 0.5389 | 0.2532 | 0.2776 |
| gta5 | segmentation | 0.7364 | 0.5244 | 0.2767 | 0.2762 |
| pixel_art | none | 0.7290 | 0.4310 | 0.2155 | 0.2311 |
| pixel_art | canny | 0.8859 | 0.7969 | 0.3479 | 0.2043 |
| pixel_art | depth | 0.8347 | 0.6207 | 0.2881 | 0.2134 |
| pixel_art | segmentation | 0.8063 | 0.5740 | 0.2829 | 0.2178 |
| watercolor | none | 0.7582 | 0.5209 | 0.2380 | 0.2468 |
| watercolor | canny | 0.8379 | 0.7705 | 0.3429 | 0.2416 |
| watercolor | depth | 0.8090 | 0.6394 | 0.2966 | 0.2409 |
| watercolor | segmentation | 0.7959 | 0.6166 | 0.2999 | 0.2441 |

[Medie, dispersione e IC completi](../../result_evaluation_rq1/analysis_common104_v1/summary_metrics.csv).

## 4. Differenze appaiate: quanto cambia rispetto all'alternativa

Ogni cella è **media della differenza [IC descrittivo al 95%]**. `A_minus_B` significa A − B. Per esempio, `+0.3542` in DINO-I è una differenza di coseno, non un aumento del 35,42% del contenuto. Gli arrotondamenti qui sono a quattro decimali; i CSV mantengono la precisione originale.

### gta5

| Contrasto | Delta CLIP-I [IC 95%] | Delta DINO-I [IC 95%] | Delta mIoU [IC 95%] | Delta CLIP-T [IC 95%] |
|---|---:|---:|---:|---:|
| canny_minus_none | +0.1558 [+0.1401; +0.1725] | +0.3542 [+0.3200; +0.3916] | +0.1558 [+0.1333; +0.1807] | -0.0242 [-0.0281; -0.0204] |
| depth_minus_none | +0.0858 [+0.0740; +0.0984] | +0.1648 [+0.1436; +0.1901] | +0.0724 [+0.0608; +0.0843] | -0.0088 [-0.0120; -0.0060] |
| segmentation_minus_none | +0.0805 [+0.0707; +0.0911] | +0.1503 [+0.1301; +0.1706] | +0.0959 [+0.0845; +0.1078] | -0.0103 [-0.0129; -0.0078] |
| canny_minus_depth | +0.0700 [+0.0578; +0.0825] | +0.1893 [+0.1629; +0.2156] | +0.0835 [+0.0661; +0.1056] | -0.0154 [-0.0191; -0.0116] |
| canny_minus_segmentation | +0.0752 [+0.0619; +0.0886] | +0.2039 [+0.1751; +0.2336] | +0.0599 [+0.0411; +0.0824] | -0.0139 [-0.0177; -0.0105] |
| depth_minus_segmentation | +0.0052 [-0.0027; +0.0134] | +0.0145 [-0.0049; +0.0354] | -0.0235 [-0.0338; -0.0128] | +0.0014 [-0.0004; +0.0032] |

### pixel_art

| Contrasto | Delta CLIP-I [IC 95%] | Delta DINO-I [IC 95%] | Delta mIoU [IC 95%] | Delta CLIP-T [IC 95%] |
|---|---:|---:|---:|---:|
| canny_minus_none | +0.1569 [+0.1389; +0.1766] | +0.3659 [+0.3283; +0.4052] | +0.1324 [+0.1134; +0.1548] | -0.0268 [-0.0316; -0.0224] |
| depth_minus_none | +0.1057 [+0.0885; +0.1233] | +0.1897 [+0.1619; +0.2207] | +0.0725 [+0.0563; +0.0937] | -0.0177 [-0.0218; -0.0138] |
| segmentation_minus_none | +0.0772 [+0.0641; +0.0915] | +0.1430 [+0.1185; +0.1687] | +0.0673 [+0.0573; +0.0789] | -0.0134 [-0.0169; -0.0098] |
| canny_minus_depth | +0.0512 [+0.0388; +0.0637] | +0.1762 [+0.1506; +0.2023] | +0.0598 [+0.0461; +0.0739] | -0.0091 [-0.0127; -0.0055] |
| canny_minus_segmentation | +0.0796 [+0.0660; +0.0940] | +0.2229 [+0.1949; +0.2516] | +0.0650 [+0.0471; +0.0859] | -0.0135 [-0.0173; -0.0098] |
| depth_minus_segmentation | +0.0285 [+0.0158; +0.0421] | +0.0467 [+0.0281; +0.0682] | +0.0052 [-0.0094; +0.0243] | -0.0044 [-0.0073; -0.0017] |

### watercolor

| Contrasto | Delta CLIP-I [IC 95%] | Delta DINO-I [IC 95%] | Delta mIoU [IC 95%] | Delta CLIP-T [IC 95%] |
|---|---:|---:|---:|---:|
| canny_minus_none | +0.0797 [+0.0666; +0.0946] | +0.2496 [+0.2180; +0.2829] | +0.1049 [+0.0874; +0.1241] | -0.0052 [-0.0100; -0.0006] |
| depth_minus_none | +0.0508 [+0.0400; +0.0626] | +0.1185 [+0.0937; +0.1471] | +0.0586 [+0.0471; +0.0711] | -0.0059 [-0.0093; -0.0026] |
| segmentation_minus_none | +0.0377 [+0.0286; +0.0483] | +0.0957 [+0.0754; +0.1186] | +0.0620 [+0.0505; +0.0748] | -0.0027 [-0.0055; +0.0002] |
| canny_minus_depth | +0.0289 [+0.0176; +0.0402] | +0.1311 [+0.1029; +0.1598] | +0.0463 [+0.0319; +0.0614] | +0.0007 [-0.0038; +0.0052] |
| canny_minus_segmentation | +0.0420 [+0.0290; +0.0551] | +0.1539 [+0.1263; +0.1819] | +0.0429 [+0.0264; +0.0602] | -0.0025 [-0.0076; +0.0024] |
| depth_minus_segmentation | +0.0131 [+0.0059; +0.0207] | +0.0228 [+0.0062; +0.0399] | -0.0034 [-0.0145; +0.0080] | -0.0032 [-0.0060; -0.0003] |

[CSV dei 72 contrasti](../../result_evaluation_rq1/analysis_common104_v1/paired_contrasts.csv).

## 5. Che cosa emerge per stile

### GTA V

Canny è il ramo più orientato alla preservazione misurata: rispetto a `none`, CLIP-I aumenta di 0.1558, DINO-I di 0.3542 e mIoU di 0.1558. CLIP-T scende di 0.0242. Il guadagno di similarità non è limitato a una singola metrica, ma resta da verificare se corrisponde anche a una trasformazione stilistica riconoscibile.

Depth e segmentation hanno un costo CLIP-T inferiore a Canny e guadagni di contenuto più contenuti. Nel confronto diretto depth − segmentation, gli IC di CLIP-I, DINO-I e CLIP-T comprendono lo zero: non c'è una separazione netta sulle tre misure. Segmentation ha invece mIoU maggiore di depth: la differenza depth − segmentation è −0.0235 [−0.0338; −0.0128]. È un risultato da controllare visivamente, perché la mIoU misura accordo tra predizioni e i modelli di segmentazione condividono ADE20K.

**Focus proposto:** guardare `none` vs Canny per capire la natura del compromesso; aggiungere depth vs segmentation per verificare se il guadagno mIoU corrisponde a oggetti/layout effettivamente meglio preservati.

### Pixel Art

Canny ha le medie più alte per CLIP-I (0.8859), DINO-I (0.7969) e mIoU (0.3479), ma il CLIP-T medio più basso (0.2043). Rispetto a `none`, il delta DINO-I è +0.3659 e il delta CLIP-T −0.0268. Questo è compatibile con output più vicini alla fotografia, ma da solo non dimostra che manchino i pixel o che la stilizzazione sia peggiore per un osservatore.

Depth supera segmentation in CLIP-I (+0.0285) e DINO-I (+0.0467), mentre segmentation ha CLIP-T maggiore: depth − segmentation è −0.0044 [−0.0073; −0.0017]. Il loro delta mIoU ha IC che comprende lo zero. Quindi anche tra questi due rami rimane un compromesso, non un ordinamento unico.

**Focus proposto:** controllare palette, blocchi e pixel visibili nelle stesse foto per `none`, Canny, depth e segmentation. Prima di suggerire nuove strength o scale, capire se i metodi differiscono davvero nella riconoscibilità Pixel Art o se parte dello scarto è un limite di CLIP-T.

### Watercolor

Canny supera `none` nelle tre metriche di contenuto (DINO-I +0.2496) e ha un delta CLIP-T di −0.0052 [−0.0100; −0.0006]. Il calo è vicino allo zero rispetto alla scala osservata, ma non abbiamo una soglia predefinita che lo renda trascurabile in pratica.

Segmentation − `none` ha delta CLIP-T −0.0027 [−0.0055; +0.0002]: i dati sono compatibili con una differenza piccola, negativa o prossima a zero. **Non è una prova di equivalenza.**

Canny supera depth nelle tre metriche di contenuto; il delta CLIP-T è +0.0007 [−0.0038; +0.0052], quindi non c'è una separazione chiara nell'allineamento al testo. Sulle sole medie Canny domina depth nelle quattro metriche, ma non possiamo trasformare questo ordinamento osservato in una garanzia fuori dal campione. Rispetto a segmentation, Canny guadagna contenuto con un delta CLIP-T di −0.0025 e IC che comprende lo zero.

**Focus proposto:** Watercolor è il primo stile in cui verificare visivamente se Canny conserva strutture utili lasciando comunque visibili lavature, pigmento e bordi pittorici. Non serve ancora una nuova generazione per rispondere a questa domanda.

## 6. È un effetto di poche foto o di un solo seed?

La tabella indica quante foto hanno **delta medio sui tre seed strettamente positivo per Canny − none**. Non è la percentuale di output "riusciti". Il complemento può includere delta negativi o esattamente nulli; le quote separate sono nel CSV.

| Stile | CLIP-I maggiore | DINO-I maggiore | mIoU maggiore | CLIP-T maggiore |
|---|---:|---:|---:|---:|
| gta5 | 103/104 (99.0%) | 104/104 (100.0%) | 101/104 (97.1%) | 6/104 (5.8%) |
| pixel_art | 102/104 (98.1%) | 103/104 (99.0%) | 98/104 (94.2%) | 10/104 (9.6%) |
| watercolor | 95/104 (91.3%) | 102/104 (98.1%) | 92/104 (88.5%) | 35/104 (33.7%) |

Il vantaggio di contenuto è diffuso. Per esempio, DINO-I aumenta in 104/104 foto per GTA V, 103/104 per Pixel Art e 102/104 per Watercolor. Il CLIP-T medio per foto diminuisce nella maggioranza dei casi.

**Stabilità delle medie fra seed:** in 71 dei 72 contrasti/metrica/stile, la differenza media ha lo stesso segno nei seed 42, 123 e 456. L'eccezione è mIoU per depth − segmentation in Watercolor: −0.00880, +0.00052, −0.00179; il suo IC aggregato comprende lo zero. I 72 contrasti sono correlati e questo conteggio non rappresenta 72 repliche indipendenti.

**La stabilità media non implica stabilità per ogni foto.** Per Canny − none, il vantaggio DINO-I è positivo in tutti e tre i seed nel 97.1% delle foto GTA V, 98.1% Pixel Art e 94.2% Watercolor. Il calo CLIP-T si ripete in tutti e tre i seed solo nel 65.4%, 64.4% e 35.6% delle foto, rispettivamente. La dipendenza da foto e seed è quindi più visibile nella misura stilistica; vanno conservati e ispezionati anche i casi discordanti.

[Stabilità e differenze per seed](../../result_evaluation_rq1/analysis_common104_v1/seed_stability.csv). Non generalizzare questi tre seed a tutta la distribuzione casuale di generazione.

## 7. Figure e come leggerle

![Differenze appaiate rispetto a none](../../result_evaluation_rq1/analysis_common104_v1/paired_vs_none.png)

A destra dello zero la condizione ha metrica maggiore di `none`; a sinistra minore. Ogni pannello ha una scala propria: confrontare il valore numerico, non la lunghezza visiva fra metriche. Per decidere il segno della differenza leggere gli IC dei delta, non gli IC di due medie separate.

![Compromesso contenuto e allineamento stilistico](../../result_evaluation_rq1/analysis_common104_v1/content_style_tradeoff.png)

Sull'asse X c'è CLIP-T, sull'asse Y una misura di contenuto. I punti trasparenti sono le 104 medie per foto di ciascun metodo; i punti grandi sono le medie e le barre sono IC marginali al 95%. Più in alto e a destra significa valori maggiori delle due misure. Le barre non sono ellissi di confidenza congiunte. Non confrontare direttamente i valori assoluti CLIP-T di stili diversi come una scala calibrata di qualità.

Disponibili anche i PDF: [delta vs none](../../result_evaluation_rq1/analysis_common104_v1/paired_vs_none.pdf), [compromesso contenuto–stile](../../result_evaluation_rq1/analysis_common104_v1/content_style_tradeoff.pdf).

## 8. Artefatti prodotti e riproduzione

| File nella directory di analisi | Che cosa contiene |
|---|---|
| `per_output_metrics.csv` | 3.744 righe originali consolidate, con ID del run di provenienza |
| `per_input_metrics.csv` | 1.248 righe: 104 foto × 3 stili × 4 condizioni, dopo media dei tre seed |
| `summary_metrics.csv` | 48 riepiloghi: medie, IC, dispersione e quantili tra foto |
| `paired_contrasts.csv` | 72 riepiloghi: i sei contrasti per stile e metrica, IC e frazioni di foto positive/negative |
| `per_input_contrasts.csv` | 1.872 righe: delta medi sui seed per foto/stile/contrasto, con quattro metriche |
| `summary_by_seed.csv` | 144 riepiloghi di condizione/metrica/stile/seed |
| `paired_contrasts_by_seed.csv` | 216 riepiloghi appaiati separati per seed |
| `seed_stability.csv` | 72 righe: delta medi per seed e coerenza del segno sulle singole foto |
| `analysis_manifest.json` | Hash dei sorgenti e degli artefatti, selezione, formula statistica, versioni e seed bootstrap |
| `paired_vs_none.*`, `content_style_tradeoff.*` | Figure statiche in PNG e PDF |

Script: [`src/evaluation/aggregate_rq1.py`](../../src/evaluation/aggregate_rq1.py). Dipendenze CPU per l'analisi e i grafici: [`requirements-analysis.txt`](../../requirements-analysis.txt). Per i grafici è stato installato Matplotlib 3.11.2 nella `.venv` di questo progetto, senza modificare le dipendenze di Kaggle o usare la GPU. Nessun import Torch nello script di aggregazione.

```powershell
# Dalla radice del progetto; indicare una directory di output NUOVA.
.\.venv\Scripts\python.exe -m src.evaluation.aggregate_rq1 `
  --results-dir result_evaluation_rq1 `
  --output-dir result_evaluation_rq1/analysis_common104_replica `
  --figures
```

Verifiche numeriche offline su casi sintetici: media e varianza calcolate per foto, IC appaiati che conservano le condizioni insieme, rilevamento di inversioni tra seed e rifiuto di matrici incomplete o duplicate. Non sostituiscono l'ispezione delle immagini.

## 9. Limiti e cosa non possiamo ancora concludere

- La selezione per completezza può introdurre un bias se i quattro input esclusi sono sistematicamente diversi; i risultati descrivono le 104 foto ammissibili, non un campione casuale dimostrato di ogni scena possibile.
- CLIP-I, DINO-I e mIoU sono proxy di contenuto, senza annotazioni ground truth. CLIP-I e CLIP-T condividono il backbone; non sono giudizi indipendenti.
- SegFormer è diverso dall'UPerNet del conditioning, ma entrambi sono legati ad ADE20K. La mIoU misura accordo tra predizioni anche su immagini stilizzate.
- CLIP-T dipende dai testi fissati. Un valore alto non certifica stile, qualità estetica o assenza di artefatti; non c'è una soglia universale di "stile riuscito".
- Le categorie automatiche dei CSV sono `unassigned`: non abbiamo completato A06 per tipologia di scena. Le distribuzioni e i delta individuali sono disponibili, ma le scene e i fallimenti visivi non sono ancora annotati.
- I rami cambiano preprocessore e checkpoint insieme. Questo confronto non isola causalmente il solo formato della mappa. Una scala comune 0.8 non dimostra l'ottimo di ciascun ramo.
- L'export fissa revisioni e ambiente dei valutatori; i manifest di generazione storici non provano revisioni e ambienti identici di ogni sessione iniziale.
- Gli IC non sono corretti per confronti multipli. Non abbiamo definito prima del test una soglia di equivalenza o di rilevanza pratica, né eseguito una valutazione umana cieca.

## 10. Decisione operativa e prossimo passo

**A01–A05 hanno ora una risposta quantitativa documentata.** Non lanciamo altre generazioni e non passiamo automaticamente a sweep di scala, strength o LoRA. Dopo il successivo OK sono state preparate le [griglie delle immagini esistenti](GRIGLIE_ISPEZIONE_RQ1_COMMON104.md): 18 casuali e 12 diagnostiche, con seed 42 fissato. La [galleria locale](../../result_evaluation_rq1/visual_review_common104_v1/index.html) è pronta; la lettura sistematica e le annotazioni percettive restano da autorizzare e svolgere. Il percorso è:

1. Fissare un campione casuale riproducibile di foto per le griglie generali, con un seed fissato prima di guardare gli output, mostrando originale e tutte e quattro le condizioni per ciascuno stile.
2. Aggiungere griglie diagnostiche separate per forte guadagno di contenuto, forte calo CLIP-T e disaccordi fra metriche. Questi casi scelti in base ai punteggi servono a spiegare fenomeni, non a stimarne la frequenza.
3. Annotare struttura, riconoscibilità stilistica e artefatti; mantenere separati giudizi soggettivi e numeri automatici. Per affermazioni di qualità percettiva predisporre poi H01 con campione e rubrica cieca.
4. Solo dopo questa verifica decidere se E01 (scala) o E02 (strength) risponde a un problema osservato. Un semplice guadagno di CLIP-I non basta a richiedere tuning.

L'analisi quantitativa non ha prodotto giudizi umani. Nel passaggio successivo VIS01 sono state aperte immagini soltanto per comporre le griglie e controllarne la leggibilità; le 120 schede percettive sono ancora vuote. Nessuna inferenza, training o GPU locale è stata usata.
