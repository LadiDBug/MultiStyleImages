# Dataset di immagini di test

Aggiornato all'**11 novembre 2026**, sulla base dei file presenti in
[`test_images/`](test_images/) e della loro ispezione visiva.

## Scopo e collocazione nel progetto

`data/test_images/` contiene **108 immagini sorgente** selezionate per i
test di trasferimento di stile. È la cartella di input indicata nella
configurazione corrente [`configs/main.toml`](../configs/main.toml).
Le immagini consentono di confrontare il comportamento della pipeline
su scene differenti, negli stili **GTA5, Pixel Art e Watercolor**, con
baseline img2img e condizionamenti **Canny, depth e segmentation**.
Possono essere usate anche per confrontare l'inferenza con e senza LoRA.
La presenza di un file nella cartella non certifica che tutte le combinazioni
sperimentali siano già state eseguite su quel file.

Questo insieme è distinto da [`pixel_art/`](pixel_art/), che contiene le
immagini stilizzate e le caption per il training del LoRA. La composizione
di quest'ultimo è descritta in [`Dataset_LoRA.md`](pixel_art/Dataset_LoRA.md).
Le 108 immagini qui documentate sono input da trasformare, non target
stilistici né un dataset già etichettato per allenare il futuro classificatore.

## Composizione e fonti

La selezione proviene da **quattro dataset**. I link seguenti sono le fonti
indicate per il progetto; il conteggio è stato verificato sui file locali.

| Dataset | Immagini | Quota sul totale | Formato locale | Nomi dei file locali | Fonte |
| --- | ---: | ---: | --- | --- | --- |
| DIV2K | 60 | 55,6% | PNG | `0002x2.png`, `0003x2.png`, …, `0731x2.png` | [DIV2K su Kaggle](https://www.kaggle.com/datasets/joe1995/div2k-dataset) |
| Flickr2K | 18 | 16,7% | PNG | `flickr2k_*.png` e `flickr2k_part2_*.png` | [Flickr2K su Kaggle](https://www.kaggle.com/datasets/hliang001/flickr2k) |
| 4KLSDB | 11 | 10,2% | JPG | `4klsdb_*.jpg` | [Sito del dataset 4KLSDB](https://4klsdb.github.io/) |
| LSDIR | 19 | 17,6% | PNG | `lsdir_*.png` | [Sito del dataset LSDIR](https://ofsoundof.github.io/lsdir-data/) |
| **Totale** | **108** | **100%** | **97 PNG + 11 JPG** | | |

Le percentuali delle singole fonti sono arrotondate. Le due famiglie di
nomi Flickr2K appartengono alla stessa fonte, non a due dataset distinti.
L'attribuzione locale segue le fonti indicate e i nomi dei file; non è stata
eseguita una verifica tramite hash contro gli archivi originali.

DIV2K e Flickr2K sono fonti utilizzate nella ricerca sulla super-resolution;
LSDIR è dedicato al restauro delle immagini; 4KLSDB al restauro e alla
generazione ad alta risoluzione. Nel nostro lavoro vengono utilizzati per
la varietà dei contenuti delle immagini sorgente, senza adottare il loro
compito originale o i loro split ufficiali come protocollo sperimentale.

## Criterio di selezione

La scelta delle immagini è stata guidata dalla **diversità dei contesti**,
con particolare attenzione ai panorami. L'obiettivo è osservare se il
trasferimento di stile mantiene la struttura della scena in ambienti
naturali, urbani e misti, anziché valutarlo soltanto su una tipologia di
fotografia o su soggetti isolati.

L'ispezione visiva conferma un insieme ricco di vedute ampie: montagne,
laghi, coste, boschi, città, strade, edifici e monumenti. A questi si
affiancano interni, scene con persone, mezzi di trasporto e alcuni animali,
che ampliano i casi osservabili. Non tutte le immagini sono quindi panorami.

La selezione è qualitativa: non è un campionamento casuale dei quattro
dataset e non prevede lo stesso numero di immagini per ogni categoria.
La tabella seguente descrive i contesti osservati e la loro utilità per
gli esperimenti; le categorie possono sovrapporsi e non costituiscono
annotazioni ufficiali dei file.

| Contesto osservato | Esempi locali | Motivo della presenza nei test |
| --- | --- | --- |
| Montagne, valli e ambienti innevati | `0002x2.png`, `0030x2.png`, `0072x2.png`, `0209x2.png`, `0229x2.png` | Ampie distanze, profili montuosi e piani sovrapposti permettono di osservare la conservazione di geometria e profondità. |
| Laghi, fiumi, coste e mare | `0004x2.png`, `0039x2.png`, `0051x2.png`, `0205x2.png`, `lsdir_0359.png` | Superfici d'acqua, riflessi e confini tra acqua, terreno e cielo introducono strutture diverse da quelle delle scene urbane. |
| Boschi, vegetazione e paesaggi rurali | `0211x2.png`, `0218x2.png`, `4klsdb_0382.jpg`, `4klsdb_0429.jpg`, `lsdir_0356.png` | Fogliame, tronchi, coltivazioni e sentieri offrono texture dense e irregolari su cui verificare l'effetto della stilizzazione. |
| Rocce, canyon, cascate e fenomeni naturali | `0203x2.png`, `0236x2.png`, `0266x2.png`, `0396x2.png`, `0432x2.png` | Forme organiche, acqua in movimento, fumo e contrasti luminosi ampliano la varietà dei paesaggi naturali. |
| Skyline, quartieri e vedute urbane | `0045x2.png`, `0057x2.png`, `0684x2.png`, `0712x2.png`, `4klsdb_0243.jpg`, `flickr2k_0467.png` | Edifici ripetuti, prospettive e dettagli su più scale permettono di valutare quanto rimanga leggibile l'organizzazione della città. |
| Strade, piazze e infrastrutture | `0022x2.png`, `0680x2.png`, `4klsdb_0083.jpg`, `flickr2k_part2_0081.png`, `lsdir_0411.png` | Ponti, carreggiate e facciate presentano linee e punti di fuga utili al confronto con il condizionamento Canny. |
| Monumenti e architettura storica | `0053x2.png`, `0127x2.png`, `0600x2.png`, `0691x2.png`, `flickr2k_part2_0383.png`, `lsdir_0234.png` | Sagome riconoscibili, simmetrie e decorazioni consentono di verificare la conservazione dei caratteri distintivi degli edifici. |
| Ambienti misti naturali e costruiti | `0146x2.png`, `0433x2.png`, `0498x2.png`, `flickr2k_0475.png`, `flickr2k_part2_0095.png` | La compresenza di vegetazione, acqua e costruzioni permette di osservare come vengono trattate regioni semanticamente differenti nella stessa immagine. |
| Interni e spazi arredati | `0314x2.png`, `0483x2.png`, `0653x2.png`, `4klsdb_0302.jpg`, `4klsdb_0303.jpg` | Arredi, pareti, libri e decorazioni introducono occlusioni, geometrie ravvicinate e illuminazione interna. Sono presenti anche terrazze e cortili, ad esempio `flickr2k_part2_0348.png` e `flickr2k_part2_0380.png`. |
| Persone inserite nel contesto e folle | `0173x2.png`, `0432x2.png`, `0674x2.png`, `0731x2.png`, `4klsdb_0120.jpg`, `flickr2k_0226.png` | Sono presenti figure piccole nel paesaggio, persone nelle strade, mercati e pubblico numeroso: consentono di osservare sia la scena globale sia i soggetti al suo interno. |
| Trasporti e veicoli | `0609x2.png`, `0617x2.png`, `0631x2.png`, `0667x2.png`, `lsdir_0190.png`, `lsdir_0334.png`, `lsdir_0405.png` | Automobili, moto, treni, imbarcazioni e aerei introducono sagome riconoscibili e dettagli meccanici in contesti differenti. |
| Animali | `0235x2.png`, `0629x2.png`, `lsdir_0356.png` | Questi casi ampliano i soggetti oltre paesaggi e architettura, includendo animali in primo piano e nell'ambiente. |

## Varietà visiva rilevante per la pipeline

Oltre al contenuto, cambiano le condizioni di ripresa: luce diurna,
tramonti e scene notturne; stagioni e presenza di neve; superfici lisce
e riflettenti accanto a vegetazione e architetture ricche di dettagli;
scenari aperti accanto a interni e strade affollate. È presente anche
un'immagine in bianco e nero, `0617x2.png`.

Questa varietà consente di porre domande differenti ai condizionamenti:

- **Canny:** quanto vengono mantenuti contorni, strutture architettoniche,
  infrastrutture e sagome dei soggetti?
- **Depth:** quanto resta coerente l'organizzazione spaziale tra primo
  piano, piani intermedi e sfondo nelle scene con distanze differenti?
- **Segmentation:** quanto vengono rispettate la disposizione e
  l'estensione delle regioni della scena, come cielo, vegetazione, edifici,
  persone e veicoli, nei limiti delle classi riconosciute dal modello?

Queste sono motivazioni sperimentali della selezione, non risultati:
l'efficacia dei condizionamenti deve essere verificata sulle immagini
generate e sulle metriche della pipeline.

## Caratteristiche dei file e limiti dell'insieme

- **108 immagini:** 97 PNG e 11 JPG; i file di servizio come `.DS_Store`
  sono esclusi dal conteggio.
- **Orientamento:** 106 immagini orizzontali, una verticale e una quadrata.
  L'insieme privilegia quindi le vedute ampie.
- **Dimensioni locali:** larghezze tra 786 e 1344 pixel e altezze tra 426 e
  1020 pixel. Sono intervalli delle singole dimensioni, non due risoluzioni
  comuni a tutti i file.
- I file locali non vanno descritti automaticamente come originali alla
  massima risoluzione delle fonti. In particolare, la provenienza da
  4KLSDB non implica che le copie in questa cartella siano ancora in 4K.
  Il suffisso `x2` delle immagini DIV2K è coerente con una variante
  sottocampionata per super-resolution; non indica un ingrandimento
  applicato dalla nostra pipeline.
- I panorami e l'architettura sono molto presenti; persone, animali e
  interni ampliano la copertura ma non rendono il dataset bilanciato per
  soggetto. Le categorie descritte sopra non hanno conteggi formalmente
  annotati e non sono mutuamente esclusive.
- Il set di test del progetto è una selezione locale: non coincide
  necessariamente con gli split denominati `test` nei dataset sorgente.
  Nomi e prefissi locali non documentano da soli lo split originale,
  l'indice originale dopo eventuali rinominazioni o tutte le trasformazioni
  effettuate prima dell'importazione.

Per i confronti sperimentali occorre mantenere la stessa selezione di
immagini tra le condizioni, così che il contenuto di partenza resti
comparabile. Se si riutilizzerà questo insieme nel futuro progetto di
classificazione, serviranno un protocollo di split dedicato e controlli
di sovrapposizione con i dati di training; questa documentazione non
certifica già l'indipendenza tra tutti gli insiemi.
