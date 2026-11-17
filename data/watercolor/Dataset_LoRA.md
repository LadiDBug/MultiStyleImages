# Dataset per il LoRA Watercolor

Aggiornato al **17 novembre 2026**. Selezione **`watercolor400-v1-2026-11-17`**.
Il dataset contiene **400 immagini con caption in inglese**, suddivise in
**360 train** e **40 validation**, per il training LoRA di Stable Diffusion 1.5.
Le immagini sono già stilizzate ad acquerello: non sono fotografie da trasformare
e non costituiscono coppie fotografia/target. Il training non è stato eseguito.

## Criteri ripresi dal dataset Pixel Art

Riferimento: [`../pixel_art/Dataset_LoRA.md`](../pixel_art/Dataset_LoRA.md).
Sono stati mantenuti dimensione, rapporto train/validation, varietà dei soggetti,
caption del contenuto visibile, attenzione alle persone nel loro ambiente,
conservazione dei byte nativi e controlli sui duplicati.

La selezione è organizzata in **300 immagini di base + 100 scene aggiuntive con persone**.
La base include paesaggi, architettura, interni, animali, trasporti, oggetti,
cibo e piante; può contenere anche persone. Le scene aggiuntive sono file
distinti dai 300 e comprendono:

- **10 scene con folla**, fra mercati, strade, festival e luoghi collettivi;
- **42 con piccoli gruppi**, fra famiglie, amici, scuola, lavoro e attività all'aperto;
- **48 con una persona principale** e un ambiente riconoscibile;
- **24 delle 100** con figure lontane o inquadrature ampie, trasversali ai tre gruppi.

Le etichette sono qualitative, assegnate attraverso ispezione visiva, senza una
soglia rigida di numero di persone. Una scena con persona principale può
contenere figure secondarie. Le 24 viste ampie non vanno sommate al totale.
Sono inclusi anche mezzi busti con contesto visibile.

La parte storica «300 originali congelati» del Pixel Art non è trasferibile:
per watercolor non esisteva una precedente selezione locale. Questi 300 sono
una **nuova base**, ora fissata insieme alle altre 100 nel manifest.
Sono stati esclusi ritratti isolati dalla quota delle 100 scene, schede di
personaggi, collage di studi, immagini con scritte decorative dominanti e
alcuni esempi prevalentemente fantascientifici. Rimangono oggetti su fondo
chiaro, piccole insegne, eventuali firme delle opere e alcune scene fantastiche:
la selezione privilegia contenuti rappresentativi senza pretendere che tutti
siano direttamente fotografici.

## Fonti, licenze dichiarate e revisioni fissate

### Watercolor FLUX samples — 255 immagini

[DarkMoonDragon/watercolor-flux-samples](https://huggingface.co/datasets/DarkMoonDragon/watercolor-flux-samples/tree/59ba3fb6b2cc426b9b06feff0d382f5941d135af). Licenza dichiarata: **MIT**. Revisione: `59ba3fb6b2cc426b9b06feff0d382f5941d135af`.

Campioni sintetici generati con FLUX-dev e il modello
`SebastianBodza/Flux_Aquarell_Watercolor_v2`, secondo la dataset card.
Sono state scelte 239 immagini per la base e 16 per le scene aggiuntive.
La fonte non fornisce caption: quelle locali sono state scritte dal contenuto
visibile. I nomi `dark_NNNNN.png` corrispondono a `img_N.png` nella fonte.

Dataset card conservata: [`provenance/dark_README.md`](provenance/dark_README.md).

### GPT Watercolor Anime — 99 immagini

[neonforestmist/GPT_Watercolor_Anime_Style_Images](https://huggingface.co/datasets/neonforestmist/GPT_Watercolor_Anime_Style_Images/tree/4c7392a697adc3aac54faedcc99aab08073c8639). Licenza dichiarata: **Apache-2.0**. Revisione: `4c7392a697adc3aac54faedcc99aab08073c8639`.

Immagini sintetiche GPT con lavature ad acquerello e disegno anime,
secondo la dataset card. Selezione: 35 immagini di base e 64 scene aggiuntive.
Le caption sorgenti sono state accorciate, rimuovendo il suffisso stilistico
ripetuto, e corrette quando necessario dopo il controllo visivo. Per esempio,
la caption della stanza 47 ora cita le due persone effettivamente visibili.
I nomi `neon_NNNNN.png` rimandano a `images/gpt_watercolor_anime_NNNN.png`.
La componente anime resta visibile e può influenzare il LoRA.

Dataset card conservata: [`provenance/neon_README.md`](provenance/neon_README.md).

### watercolour2k — 28 immagini

[jtorregrosa/watercolour2k](https://huggingface.co/datasets/jtorregrosa/watercolour2k/tree/5ac3aaf9545ff81bdc205b157362324619952999). Licenza dichiarata: **MIT**. Revisione: `5ac3aaf9545ff81bdc205b157362324619952999`.

Selezione di 8 immagini di base e 20 scene aggiuntive da un campione
riproducibile di 720 candidati, estratto con `random.Random(42).sample`
dall'elenco ordinato dei 19.814 JPEG alla revisione indicata. Il file sorgente
`metadata.jsonl` contiene nomi, senza caption: quelle locali sono state redatte
visivamente. La card non documenta la provenienza delle singole opere né
specifica quali siano sintetiche. Non si inferiscono autori dai soggetti.
`w2k_NNNNN.jpg` mantiene l'indice numerico del JPEG sorgente; il percorso
completo è nel manifest. Il nome della fonte non implica una risoluzione 2K.

Dataset card conservata: [`provenance/w2k_README.md`](provenance/w2k_README.md).

### Marie-Claire Lefèbure — 18 immagini

[reversely/marie-claire-lefebure-watercolours](https://huggingface.co/datasets/reversely/marie-claire-lefebure-watercolours/tree/ba30230f0a07f29dcdbbc6f0f21e9fbc6c1fc387). Licenza dichiarata: **CC-BY-4.0**. Revisione: `ba30230f0a07f29dcdbbc6f0f21e9fbc6c1fc387`.

Acquerelli attribuiti a **Marie-Claire Lefèbure**, ripubblicati da
Wikimedia Commons secondo la dataset card. Attribuzione da conservare:
**By Marie-Claire — Own work, CC BY 4.0**.
[Categoria e attribuzione su Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Marie-Claire_Lef%C3%A9bure).
Tutte le 18 immagini sono nella base; i byte delle opere non sono stati modificati.
La card parla di 18 opere, ma il repository contiene 19 JPEG: il file con titolo
`Ancienne ferme à Ambly` è stato escluso perché mostra una veduta molto simile
alla chiesa di Hastière già presente. Le caption sono state riscritte dal
contenuto visibile: il JPEG intitolato `Le Renard`, per esempio, mostra un vaso
di fiori. I titoli sorgenti non sono usati come descrizioni garantite.
I nomi locali `marie_NNNNN.jpg` sono associati ai percorsi originali nel manifest.

Dataset card conservata: [`provenance/marie_README.md`](provenance/marie_README.md).

Le licenze sono quelle dichiarate dai repository: non rappresentano una
verifica indipendente dei diritti di ogni immagine. In particolare, le card
minime non documentano i diritti opera per opera; la dichiarazione MIT dei
campioni FLUX non risolve da sola eventuali termini del modello generatore.
Il dataset locale ha **provenienza e licenze miste**, non una nuova licenza unica.

## Composizione e split

| Fonte | Train | Validation | Totale |
|---|---:|---:|---:|
| Watercolor FLUX samples | 229 | 26 | 255 |
| GPT Watercolor Anime | 89 | 10 | 99 |
| watercolour2k | 26 | 2 | 28 |
| Marie-Claire Lefèbure | 16 | 2 | 18 |
| **Totale** | **360** | **40** | **400** |

La base ha split **270/30**; le scene aggiuntive **90/10**. Nel validation
delle scene aggiuntive sono presenti **1 folla, 4 piccoli gruppi e 5 scene
con persona principale**. Lo split è fissato nel manifest, con seed 42,
e include tutte e quattro le fonti nella validation.
Il campionamento è qualitativo; non è una stima rappresentativa di tutti
gli acquerelli e non bilancia in modo uniforme le categorie della base.

## File e utilizzo

```text
data/watercolor/
├── Dataset_LoRA.md
├── train/                 # 360 immagini native + metadata.jsonl
├── validation/            # 40 immagini native + metadata.jsonl
├── manifest.jsonl         # selezione, caption, split, SHA256 e annotazioni
├── sources.json           # repository, revisioni e licenze dichiarate
├── provenance/            # dataset card alle revisioni fissate
├── audit.json             # esito dei controlli locali
├── prepare_dataset.py     # ripristino e audit della selezione fissata
└── index.html             # galleria locale, ricerca e anteprima crop
```

Il trainer legge i soli `metadata.jsonl` degli split, con `file_name` e `text`.
Ogni caption inizia con `Watercolor painting.` e descrive il contenuto visibile.
Le caption sono state riviste su tavole di contatto; sono stati controllati
anche i crop centrali delle fonti non quadrate. La revisione non equivale a
una certificazione dei dettagli più piccoli di ogni immagine.
Il tokenizer CLIP locale non tronca alcuna caption: il massimo verificato è
**22 token**, su un limite di 77. Il controllo `--check-data` del trainer è
passato per tutti i 400 esempi; gli esiti sono in
[`validation_checks.json`](validation_checks.json).

I file conservano i byte scaricati, senza ricompressione. Il loader esistente,
benché si chiami `PixelArtDataset`, supporta anche questo dataset. Nei quattro
nuovi config `configs/lora_watercolor_rank{4,8,16,32}.toml` usa EXIF/RGB,
resize proporzionale **Lanczos**, crop centrale **512×512** e normalizzazione
in [-1, 1]. Lanczos preserva meglio le sfumature dell'acquerello; è una scelta
diversa dal nearest-neighbor usato per Pixel Art. Non sono stati modificati
loader, config o immagini del Pixel Art.

Dalla radice del repository:

```bash
# Verifica dei file, delle quote e dei duplicati, senza rete o training:
python data/watercolor/prepare_dataset.py --audit

# Ripristino di eventuali file mancanti dalle revisioni fissate:
python data/watercolor/prepare_dataset.py --download

# Verifica con il trainer esistente:
python main.py --lora lora_watercolor_rank4.toml train --check-data

# Training, da eseguire separatamente quando desiderato:
python main.py --lora lora_watercolor_rank4.toml train
```

Lo script di ripristino richiede `requests`, Pillow e NumPy; usa soltanto i
400 percorsi nel manifest e verifica SHA256 prima di salvare un download.
Non ricampiona né riassegna gli split. Il controllo del trainer richiede le
dipendenze del progetto. Usare gli stessi file, caption e split per tutti i
rank 4/8/16/32; la validation non aggiorna i pesi.

Le immagini e i metadata degli split sono esclusi da Git, come per Pixel Art.
Manifest, provenienza e script permettono di ripristinarli finché le fonti
rimangono accessibili. Le fotografie in `data/test_images/` restano separate.

## Integrità e limiti

Dimensione delle sole immagini: **496.063.491 byte** (circa 473.1 MiB).

Verificati tutti i 400 SHA256, la decodifica RGB, i file effettivi rispetto
ai metadata, i conteggi e le quote. Nessun duplicato RGB esatto nel dataset.
Nessuna coppia tra split con **dHash ≤4**, sia sull'immagine originale sia sul
crop centrale Lanczos usato per il training. Il minimo dHash fra split è
9. Per fissare lo split sono state previste componenti di immagini simili;
la selezione finale non contiene coppie con dHash ≤4 neppure nello stesso split.
Questi controlli non escludono identità ripetute, soggetti simili, varianti
semantiche o overlap semantico con i benchmark RQ1. Non è stato certificato
l'overlap opera per opera con dataset esterni.

Firme compatibili con l'audit del loader (nomi, caption e byte):

- train: `7963171c6fcebcb549b03ef87ab0566eb9a70916d5aa496665ab9dfc91c0d172`;
- validation: `4cd401e589215ae6464db09b9caab291506e11b373b066e93fe6a67a3a84b3d4`;

Lo stile non è uniforme: la fonte FLUX domina numericamente, una parte è anime,
le opere umane hanno composizioni e firme proprie e watercolour2k include
illustrazioni con tratti diversi. Ci sono soggetti isolati nella base, paesaggi
simili e poche scene fantastiche. La qualità del trasferimento va valutata
sulle immagini generate, non dedotta dalla sola loss o dai conteggi.

## Inventario completo

Le caption, i percorsi sorgente, gli SHA256 e le etichette sono nel
[`manifest.jsonl`](manifest.jsonl); le dimensioni sono anche in [`audit.json`](audit.json).
La [galleria locale](index.html) permette di vedere originali, anteprima crop,
caption, fonte e split. `core300` identifica la nuova base e `people100` le
scene aggiuntive, senza doppio conteggio.

| File locale | Split | Gruppo | Fonte |
|---|---|---|---|
| dark_00001.png | train | core300 | Watercolor FLUX samples |
| dark_00002.png | validation | core300 | Watercolor FLUX samples |
| dark_00003.png | train | core300 | Watercolor FLUX samples |
| dark_00004.png | train | core300 | Watercolor FLUX samples |
| dark_00005.png | train | core300 | Watercolor FLUX samples |
| dark_00006.png | train | core300 | Watercolor FLUX samples |
| dark_00007.png | validation | core300 | Watercolor FLUX samples |
| dark_00010.png | validation | core300 | Watercolor FLUX samples |
| dark_00011.png | validation | core300 | Watercolor FLUX samples |
| dark_00012.png | train | core300 | Watercolor FLUX samples |
| dark_00013.png | train | core300 | Watercolor FLUX samples |
| dark_00014.png | train | core300 | Watercolor FLUX samples |
| dark_00015.png | train | core300 | Watercolor FLUX samples |
| dark_00016.png | train | core300 | Watercolor FLUX samples |
| dark_00017.png | train | core300 | Watercolor FLUX samples |
| dark_00018.png | train | core300 | Watercolor FLUX samples |
| dark_00019.png | train | core300 | Watercolor FLUX samples |
| dark_00020.png | train | core300 | Watercolor FLUX samples |
| dark_00021.png | train | core300 | Watercolor FLUX samples |
| dark_00022.png | validation | people100 | Watercolor FLUX samples |
| dark_00024.png | train | people100 | Watercolor FLUX samples |
| dark_00025.png | train | core300 | Watercolor FLUX samples |
| dark_00026.png | train | core300 | Watercolor FLUX samples |
| dark_00028.png | train | core300 | Watercolor FLUX samples |
| dark_00029.png | train | core300 | Watercolor FLUX samples |
| dark_00030.png | validation | core300 | Watercolor FLUX samples |
| dark_00031.png | train | core300 | Watercolor FLUX samples |
| dark_00032.png | train | core300 | Watercolor FLUX samples |
| dark_00033.png | train | people100 | Watercolor FLUX samples |
| dark_00034.png | train | core300 | Watercolor FLUX samples |
| dark_00035.png | train | core300 | Watercolor FLUX samples |
| dark_00036.png | train | core300 | Watercolor FLUX samples |
| dark_00037.png | train | core300 | Watercolor FLUX samples |
| dark_00038.png | train | core300 | Watercolor FLUX samples |
| dark_00039.png | train | people100 | Watercolor FLUX samples |
| dark_00040.png | train | core300 | Watercolor FLUX samples |
| dark_00041.png | train | core300 | Watercolor FLUX samples |
| dark_00042.png | train | core300 | Watercolor FLUX samples |
| dark_00044.png | train | core300 | Watercolor FLUX samples |
| dark_00045.png | train | core300 | Watercolor FLUX samples |
| dark_00046.png | train | core300 | Watercolor FLUX samples |
| dark_00047.png | train | core300 | Watercolor FLUX samples |
| dark_00048.png | train | core300 | Watercolor FLUX samples |
| dark_00050.png | train | core300 | Watercolor FLUX samples |
| dark_00051.png | train | core300 | Watercolor FLUX samples |
| dark_00052.png | train | core300 | Watercolor FLUX samples |
| dark_00053.png | train | core300 | Watercolor FLUX samples |
| dark_00055.png | train | core300 | Watercolor FLUX samples |
| dark_00056.png | train | core300 | Watercolor FLUX samples |
| dark_00057.png | train | core300 | Watercolor FLUX samples |
| dark_00058.png | train | core300 | Watercolor FLUX samples |
| dark_00059.png | train | people100 | Watercolor FLUX samples |
| dark_00060.png | train | core300 | Watercolor FLUX samples |
| dark_00061.png | train | core300 | Watercolor FLUX samples |
| dark_00062.png | train | core300 | Watercolor FLUX samples |
| dark_00063.png | validation | core300 | Watercolor FLUX samples |
| dark_00064.png | train | core300 | Watercolor FLUX samples |
| dark_00065.png | train | core300 | Watercolor FLUX samples |
| dark_00066.png | train | core300 | Watercolor FLUX samples |
| dark_00067.png | train | core300 | Watercolor FLUX samples |
| dark_00068.png | validation | core300 | Watercolor FLUX samples |
| dark_00070.png | validation | core300 | Watercolor FLUX samples |
| dark_00072.png | train | core300 | Watercolor FLUX samples |
| dark_00073.png | train | core300 | Watercolor FLUX samples |
| dark_00074.png | validation | core300 | Watercolor FLUX samples |
| dark_00077.png | train | core300 | Watercolor FLUX samples |
| dark_00079.png | train | core300 | Watercolor FLUX samples |
| dark_00080.png | train | core300 | Watercolor FLUX samples |
| dark_00081.png | train | core300 | Watercolor FLUX samples |
| dark_00082.png | train | core300 | Watercolor FLUX samples |
| dark_00083.png | train | core300 | Watercolor FLUX samples |
| dark_00085.png | train | core300 | Watercolor FLUX samples |
| dark_00086.png | train | core300 | Watercolor FLUX samples |
| dark_00087.png | train | core300 | Watercolor FLUX samples |
| dark_00088.png | train | core300 | Watercolor FLUX samples |
| dark_00089.png | train | core300 | Watercolor FLUX samples |
| dark_00090.png | validation | core300 | Watercolor FLUX samples |
| dark_00091.png | train | core300 | Watercolor FLUX samples |
| dark_00092.png | train | core300 | Watercolor FLUX samples |
| dark_00093.png | train | people100 | Watercolor FLUX samples |
| dark_00094.png | train | core300 | Watercolor FLUX samples |
| dark_00097.png | train | core300 | Watercolor FLUX samples |
| dark_00098.png | train | core300 | Watercolor FLUX samples |
| dark_00100.png | train | core300 | Watercolor FLUX samples |
| dark_00101.png | train | core300 | Watercolor FLUX samples |
| dark_00103.png | train | core300 | Watercolor FLUX samples |
| dark_00104.png | train | core300 | Watercolor FLUX samples |
| dark_00105.png | train | core300 | Watercolor FLUX samples |
| dark_00106.png | train | core300 | Watercolor FLUX samples |
| dark_00107.png | train | core300 | Watercolor FLUX samples |
| dark_00108.png | train | core300 | Watercolor FLUX samples |
| dark_00109.png | train | core300 | Watercolor FLUX samples |
| dark_00110.png | train | core300 | Watercolor FLUX samples |
| dark_00111.png | train | core300 | Watercolor FLUX samples |
| dark_00114.png | train | core300 | Watercolor FLUX samples |
| dark_00115.png | train | core300 | Watercolor FLUX samples |
| dark_00117.png | train | core300 | Watercolor FLUX samples |
| dark_00120.png | train | core300 | Watercolor FLUX samples |
| dark_00121.png | train | core300 | Watercolor FLUX samples |
| dark_00122.png | train | core300 | Watercolor FLUX samples |
| dark_00124.png | train | core300 | Watercolor FLUX samples |
| dark_00125.png | train | core300 | Watercolor FLUX samples |
| dark_00126.png | train | core300 | Watercolor FLUX samples |
| dark_00127.png | train | core300 | Watercolor FLUX samples |
| dark_00128.png | train | core300 | Watercolor FLUX samples |
| dark_00129.png | train | core300 | Watercolor FLUX samples |
| dark_00130.png | train | core300 | Watercolor FLUX samples |
| dark_00132.png | train | core300 | Watercolor FLUX samples |
| dark_00134.png | train | core300 | Watercolor FLUX samples |
| dark_00135.png | train | core300 | Watercolor FLUX samples |
| dark_00136.png | train | core300 | Watercolor FLUX samples |
| dark_00137.png | train | core300 | Watercolor FLUX samples |
| dark_00139.png | train | core300 | Watercolor FLUX samples |
| dark_00140.png | validation | core300 | Watercolor FLUX samples |
| dark_00141.png | train | people100 | Watercolor FLUX samples |
| dark_00142.png | validation | core300 | Watercolor FLUX samples |
| dark_00144.png | train | core300 | Watercolor FLUX samples |
| dark_00145.png | train | core300 | Watercolor FLUX samples |
| dark_00146.png | train | core300 | Watercolor FLUX samples |
| dark_00147.png | train | core300 | Watercolor FLUX samples |
| dark_00148.png | train | core300 | Watercolor FLUX samples |
| dark_00149.png | train | people100 | Watercolor FLUX samples |
| dark_00150.png | validation | core300 | Watercolor FLUX samples |
| dark_00151.png | train | core300 | Watercolor FLUX samples |
| dark_00153.png | train | core300 | Watercolor FLUX samples |
| dark_00155.png | train | core300 | Watercolor FLUX samples |
| dark_00156.png | train | people100 | Watercolor FLUX samples |
| dark_00157.png | train | core300 | Watercolor FLUX samples |
| dark_00158.png | train | core300 | Watercolor FLUX samples |
| dark_00159.png | train | core300 | Watercolor FLUX samples |
| dark_00161.png | train | core300 | Watercolor FLUX samples |
| dark_00162.png | train | core300 | Watercolor FLUX samples |
| dark_00163.png | train | core300 | Watercolor FLUX samples |
| dark_00164.png | train | core300 | Watercolor FLUX samples |
| dark_00165.png | train | core300 | Watercolor FLUX samples |
| dark_00167.png | train | core300 | Watercolor FLUX samples |
| dark_00168.png | train | people100 | Watercolor FLUX samples |
| dark_00170.png | train | core300 | Watercolor FLUX samples |
| dark_00171.png | train | core300 | Watercolor FLUX samples |
| dark_00172.png | validation | core300 | Watercolor FLUX samples |
| dark_00173.png | train | core300 | Watercolor FLUX samples |
| dark_00174.png | train | core300 | Watercolor FLUX samples |
| dark_00176.png | train | core300 | Watercolor FLUX samples |
| dark_00177.png | train | core300 | Watercolor FLUX samples |
| dark_00178.png | train | core300 | Watercolor FLUX samples |
| dark_00179.png | train | core300 | Watercolor FLUX samples |
| dark_00180.png | train | core300 | Watercolor FLUX samples |
| dark_00183.png | train | core300 | Watercolor FLUX samples |
| dark_00184.png | train | core300 | Watercolor FLUX samples |
| dark_00186.png | validation | core300 | Watercolor FLUX samples |
| dark_00187.png | train | core300 | Watercolor FLUX samples |
| dark_00188.png | train | core300 | Watercolor FLUX samples |
| dark_00190.png | train | core300 | Watercolor FLUX samples |
| dark_00191.png | validation | core300 | Watercolor FLUX samples |
| dark_00192.png | train | core300 | Watercolor FLUX samples |
| dark_00193.png | train | core300 | Watercolor FLUX samples |
| dark_00194.png | train | core300 | Watercolor FLUX samples |
| dark_00195.png | train | core300 | Watercolor FLUX samples |
| dark_00196.png | train | people100 | Watercolor FLUX samples |
| dark_00197.png | train | core300 | Watercolor FLUX samples |
| dark_00198.png | train | core300 | Watercolor FLUX samples |
| dark_00199.png | validation | core300 | Watercolor FLUX samples |
| dark_00200.png | validation | core300 | Watercolor FLUX samples |
| dark_00201.png | train | core300 | Watercolor FLUX samples |
| dark_00202.png | train | core300 | Watercolor FLUX samples |
| dark_00203.png | validation | core300 | Watercolor FLUX samples |
| dark_00204.png | train | core300 | Watercolor FLUX samples |
| dark_00205.png | train | core300 | Watercolor FLUX samples |
| dark_00206.png | train | core300 | Watercolor FLUX samples |
| dark_00207.png | train | core300 | Watercolor FLUX samples |
| dark_00208.png | train | core300 | Watercolor FLUX samples |
| dark_00209.png | train | core300 | Watercolor FLUX samples |
| dark_00211.png | train | core300 | Watercolor FLUX samples |
| dark_00212.png | train | core300 | Watercolor FLUX samples |
| dark_00213.png | train | core300 | Watercolor FLUX samples |
| dark_00214.png | train | core300 | Watercolor FLUX samples |
| dark_00217.png | train | core300 | Watercolor FLUX samples |
| dark_00218.png | validation | core300 | Watercolor FLUX samples |
| dark_00219.png | train | core300 | Watercolor FLUX samples |
| dark_00222.png | train | core300 | Watercolor FLUX samples |
| dark_00223.png | train | core300 | Watercolor FLUX samples |
| dark_00224.png | train | core300 | Watercolor FLUX samples |
| dark_00225.png | train | core300 | Watercolor FLUX samples |
| dark_00226.png | train | core300 | Watercolor FLUX samples |
| dark_00227.png | train | core300 | Watercolor FLUX samples |
| dark_00228.png | train | core300 | Watercolor FLUX samples |
| dark_00229.png | train | core300 | Watercolor FLUX samples |
| dark_00230.png | train | core300 | Watercolor FLUX samples |
| dark_00231.png | train | core300 | Watercolor FLUX samples |
| dark_00232.png | train | core300 | Watercolor FLUX samples |
| dark_00233.png | validation | core300 | Watercolor FLUX samples |
| dark_00234.png | train | core300 | Watercolor FLUX samples |
| dark_00236.png | train | core300 | Watercolor FLUX samples |
| dark_00237.png | train | core300 | Watercolor FLUX samples |
| dark_00238.png | validation | core300 | Watercolor FLUX samples |
| dark_00239.png | train | core300 | Watercolor FLUX samples |
| dark_00240.png | train | core300 | Watercolor FLUX samples |
| dark_00241.png | train | core300 | Watercolor FLUX samples |
| dark_00244.png | train | core300 | Watercolor FLUX samples |
| dark_00245.png | train | core300 | Watercolor FLUX samples |
| dark_00246.png | train | core300 | Watercolor FLUX samples |
| dark_00247.png | train | core300 | Watercolor FLUX samples |
| dark_00248.png | train | core300 | Watercolor FLUX samples |
| dark_00249.png | train | core300 | Watercolor FLUX samples |
| dark_00250.png | train | core300 | Watercolor FLUX samples |
| dark_00251.png | train | people100 | Watercolor FLUX samples |
| dark_00252.png | validation | core300 | Watercolor FLUX samples |
| dark_00253.png | train | core300 | Watercolor FLUX samples |
| dark_00254.png | train | core300 | Watercolor FLUX samples |
| dark_00255.png | train | core300 | Watercolor FLUX samples |
| dark_00256.png | train | core300 | Watercolor FLUX samples |
| dark_00257.png | train | core300 | Watercolor FLUX samples |
| dark_00258.png | train | core300 | Watercolor FLUX samples |
| dark_00259.png | train | core300 | Watercolor FLUX samples |
| dark_00260.png | train | core300 | Watercolor FLUX samples |
| dark_00262.png | train | core300 | Watercolor FLUX samples |
| dark_00264.png | train | people100 | Watercolor FLUX samples |
| dark_00265.png | train | core300 | Watercolor FLUX samples |
| dark_00266.png | train | core300 | Watercolor FLUX samples |
| dark_00268.png | train | core300 | Watercolor FLUX samples |
| dark_00269.png | validation | core300 | Watercolor FLUX samples |
| dark_00270.png | train | core300 | Watercolor FLUX samples |
| dark_00271.png | train | people100 | Watercolor FLUX samples |
| dark_00272.png | train | core300 | Watercolor FLUX samples |
| dark_00273.png | train | core300 | Watercolor FLUX samples |
| dark_00274.png | train | core300 | Watercolor FLUX samples |
| dark_00275.png | train | core300 | Watercolor FLUX samples |
| dark_00276.png | train | core300 | Watercolor FLUX samples |
| dark_00277.png | train | core300 | Watercolor FLUX samples |
| dark_00278.png | train | core300 | Watercolor FLUX samples |
| dark_00279.png | train | core300 | Watercolor FLUX samples |
| dark_00280.png | train | core300 | Watercolor FLUX samples |
| dark_00281.png | train | core300 | Watercolor FLUX samples |
| dark_00283.png | train | core300 | Watercolor FLUX samples |
| dark_00284.png | train | core300 | Watercolor FLUX samples |
| dark_00285.png | train | core300 | Watercolor FLUX samples |
| dark_00286.png | train | core300 | Watercolor FLUX samples |
| dark_00287.png | train | core300 | Watercolor FLUX samples |
| dark_00288.png | train | people100 | Watercolor FLUX samples |
| dark_00289.png | train | core300 | Watercolor FLUX samples |
| dark_00290.png | train | core300 | Watercolor FLUX samples |
| dark_00291.png | train | core300 | Watercolor FLUX samples |
| dark_00292.png | train | core300 | Watercolor FLUX samples |
| dark_00294.png | train | core300 | Watercolor FLUX samples |
| dark_00295.png | train | core300 | Watercolor FLUX samples |
| dark_00296.png | train | core300 | Watercolor FLUX samples |
| dark_00297.png | train | core300 | Watercolor FLUX samples |
| dark_00298.png | train | core300 | Watercolor FLUX samples |
| dark_00299.png | train | core300 | Watercolor FLUX samples |
| dark_00300.png | train | core300 | Watercolor FLUX samples |
| dark_00301.png | train | core300 | Watercolor FLUX samples |
| dark_00302.png | train | core300 | Watercolor FLUX samples |
| dark_00303.png | train | core300 | Watercolor FLUX samples |
| dark_00304.png | train | core300 | Watercolor FLUX samples |
| dark_00306.png | validation | people100 | Watercolor FLUX samples |
| neon_00001.png | train | core300 | GPT Watercolor Anime |
| neon_00003.png | train | people100 | GPT Watercolor Anime |
| neon_00004.png | train | people100 | GPT Watercolor Anime |
| neon_00005.png | validation | people100 | GPT Watercolor Anime |
| neon_00006.png | train | people100 | GPT Watercolor Anime |
| neon_00008.png | train | people100 | GPT Watercolor Anime |
| neon_00009.png | train | core300 | GPT Watercolor Anime |
| neon_00010.png | train | people100 | GPT Watercolor Anime |
| neon_00012.png | train | core300 | GPT Watercolor Anime |
| neon_00013.png | train | core300 | GPT Watercolor Anime |
| neon_00015.png | validation | people100 | GPT Watercolor Anime |
| neon_00016.png | validation | people100 | GPT Watercolor Anime |
| neon_00017.png | train | people100 | GPT Watercolor Anime |
| neon_00018.png | train | core300 | GPT Watercolor Anime |
| neon_00019.png | train | people100 | GPT Watercolor Anime |
| neon_00020.png | train | core300 | GPT Watercolor Anime |
| neon_00022.png | train | people100 | GPT Watercolor Anime |
| neon_00024.png | validation | people100 | GPT Watercolor Anime |
| neon_00025.png | train | people100 | GPT Watercolor Anime |
| neon_00028.png | train | people100 | GPT Watercolor Anime |
| neon_00029.png | train | people100 | GPT Watercolor Anime |
| neon_00031.png | train | people100 | GPT Watercolor Anime |
| neon_00032.png | train | people100 | GPT Watercolor Anime |
| neon_00033.png | train | people100 | GPT Watercolor Anime |
| neon_00035.png | train | core300 | GPT Watercolor Anime |
| neon_00037.png | train | core300 | GPT Watercolor Anime |
| neon_00038.png | train | people100 | GPT Watercolor Anime |
| neon_00039.png | train | people100 | GPT Watercolor Anime |
| neon_00043.png | train | people100 | GPT Watercolor Anime |
| neon_00044.png | train | core300 | GPT Watercolor Anime |
| neon_00047.png | train | people100 | GPT Watercolor Anime |
| neon_00048.png | train | people100 | GPT Watercolor Anime |
| neon_00049.png | train | core300 | GPT Watercolor Anime |
| neon_00050.png | train | people100 | GPT Watercolor Anime |
| neon_00052.png | train | people100 | GPT Watercolor Anime |
| neon_00053.png | validation | core300 | GPT Watercolor Anime |
| neon_00054.png | train | people100 | GPT Watercolor Anime |
| neon_00056.png | train | people100 | GPT Watercolor Anime |
| neon_00057.png | train | people100 | GPT Watercolor Anime |
| neon_00058.png | train | core300 | GPT Watercolor Anime |
| neon_00059.png | train | people100 | GPT Watercolor Anime |
| neon_00060.png | train | people100 | GPT Watercolor Anime |
| neon_00062.png | train | people100 | GPT Watercolor Anime |
| neon_00064.png | validation | people100 | GPT Watercolor Anime |
| neon_00066.png | train | core300 | GPT Watercolor Anime |
| neon_00067.png | train | people100 | GPT Watercolor Anime |
| neon_00068.png | train | core300 | GPT Watercolor Anime |
| neon_00069.png | train | core300 | GPT Watercolor Anime |
| neon_00071.png | train | people100 | GPT Watercolor Anime |
| neon_00072.png | train | people100 | GPT Watercolor Anime |
| neon_00073.png | train | people100 | GPT Watercolor Anime |
| neon_00075.png | train | core300 | GPT Watercolor Anime |
| neon_00076.png | train | people100 | GPT Watercolor Anime |
| neon_00077.png | train | core300 | GPT Watercolor Anime |
| neon_00079.png | train | people100 | GPT Watercolor Anime |
| neon_00080.png | train | core300 | GPT Watercolor Anime |
| neon_00082.png | train | people100 | GPT Watercolor Anime |
| neon_00087.png | validation | people100 | GPT Watercolor Anime |
| neon_00089.png | train | core300 | GPT Watercolor Anime |
| neon_00092.png | train | people100 | GPT Watercolor Anime |
| neon_00093.png | train | people100 | GPT Watercolor Anime |
| neon_00095.png | train | people100 | GPT Watercolor Anime |
| neon_00096.png | train | core300 | GPT Watercolor Anime |
| neon_00097.png | train | people100 | GPT Watercolor Anime |
| neon_00098.png | train | people100 | GPT Watercolor Anime |
| neon_00099.png | train | core300 | GPT Watercolor Anime |
| neon_00102.png | validation | core300 | GPT Watercolor Anime |
| neon_00103.png | train | core300 | GPT Watercolor Anime |
| neon_00104.png | train | core300 | GPT Watercolor Anime |
| neon_00105.png | validation | people100 | GPT Watercolor Anime |
| neon_00106.png | train | people100 | GPT Watercolor Anime |
| neon_00107.png | train | people100 | GPT Watercolor Anime |
| neon_00115.png | train | core300 | GPT Watercolor Anime |
| neon_00117.png | train | people100 | GPT Watercolor Anime |
| neon_00118.png | train | people100 | GPT Watercolor Anime |
| neon_00119.png | train | core300 | GPT Watercolor Anime |
| neon_00123.png | train | people100 | GPT Watercolor Anime |
| neon_00127.png | train | people100 | GPT Watercolor Anime |
| neon_00128.png | train | core300 | GPT Watercolor Anime |
| neon_00129.png | train | people100 | GPT Watercolor Anime |
| neon_00130.png | train | people100 | GPT Watercolor Anime |
| neon_00131.png | validation | core300 | GPT Watercolor Anime |
| neon_00132.png | train | people100 | GPT Watercolor Anime |
| neon_00133.png | train | people100 | GPT Watercolor Anime |
| neon_00134.png | train | core300 | GPT Watercolor Anime |
| neon_00135.png | train | core300 | GPT Watercolor Anime |
| neon_00136.png | train | people100 | GPT Watercolor Anime |
| neon_00138.png | train | people100 | GPT Watercolor Anime |
| neon_00139.png | train | core300 | GPT Watercolor Anime |
| neon_00141.png | train | core300 | GPT Watercolor Anime |
| neon_00142.png | train | people100 | GPT Watercolor Anime |
| neon_00143.png | train | people100 | GPT Watercolor Anime |
| neon_00144.png | train | core300 | GPT Watercolor Anime |
| neon_00145.png | train | people100 | GPT Watercolor Anime |
| neon_00146.png | train | people100 | GPT Watercolor Anime |
| neon_00147.png | train | core300 | GPT Watercolor Anime |
| neon_00148.png | train | people100 | GPT Watercolor Anime |
| neon_00149.png | train | core300 | GPT Watercolor Anime |
| neon_00150.png | train | people100 | GPT Watercolor Anime |
| w2k_17512.jpg | train | people100 | watercolour2k |
| w2k_16123.jpg | train | people100 | watercolour2k |
| w2k_01369.jpg | train | core300 | watercolour2k |
| w2k_02536.jpg | train | core300 | watercolour2k |
| w2k_10051.jpg | train | people100 | watercolour2k |
| w2k_17109.jpg | train | people100 | watercolour2k |
| w2k_09712.jpg | train | core300 | watercolour2k |
| w2k_12753.jpg | train | core300 | watercolour2k |
| w2k_12665.jpg | validation | core300 | watercolour2k |
| w2k_08777.jpg | train | people100 | watercolour2k |
| w2k_07641.jpg | train | core300 | watercolour2k |
| w2k_07476.jpg | train | people100 | watercolour2k |
| w2k_03760.jpg | train | people100 | watercolour2k |
| w2k_09463.jpg | train | core300 | watercolour2k |
| w2k_07033.jpg | train | core300 | watercolour2k |
| w2k_04024.jpg | validation | people100 | watercolour2k |
| w2k_19618.jpg | train | people100 | watercolour2k |
| w2k_10657.jpg | train | people100 | watercolour2k |
| w2k_11377.jpg | train | people100 | watercolour2k |
| w2k_15957.jpg | train | people100 | watercolour2k |
| w2k_17629.jpg | train | people100 | watercolour2k |
| w2k_09734.jpg | train | people100 | watercolour2k |
| w2k_08500.jpg | train | people100 | watercolour2k |
| w2k_11538.jpg | train | people100 | watercolour2k |
| w2k_06184.jpg | train | people100 | watercolour2k |
| w2k_14366.jpg | train | people100 | watercolour2k |
| w2k_10345.jpg | train | people100 | watercolour2k |
| w2k_14771.jpg | train | people100 | watercolour2k |
| marie_00001.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00002.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00003.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00004.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00005.jpg | validation | core300 | Marie-Claire Lefèbure |
| marie_00006.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00007.jpg | validation | core300 | Marie-Claire Lefèbure |
| marie_00008.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00009.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00010.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00011.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00012.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00013.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00014.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00015.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00016.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00017.jpg | train | core300 | Marie-Claire Lefèbure |
| marie_00018.jpg | train | core300 | Marie-Claire Lefèbure |
