# Dataset per il LoRA Pixel Art

Aggiornato al **10 novembre 2026**. Il dataset serve ad addestrare adattatori
LoRA della U-Net di Stable Diffusion 1.5 per il trasferimento di stile
**Pixel Art** nella RQ2. Contiene **400 immagini con caption**, suddivise
in **360 immagini train** e **40 validation**.

Le immagini di training sono gia stilizzate: non sono tutte fotografie
reali. Il dataset non viene usato come insieme di fotografie da trasformare
nei test di inferenza e non richiede coppie fotografia/target per il trainer.

## Fonti

### PixelParti128 — 305 immagini

[Dataset originale: nerijs/pixelparti-128-v0.1](https://huggingface.co/datasets/nerijs/pixelparti-128-v0.1).

Immagini sintetiche PNG a **128x128 pixel**, generate con SDXL Turbo e Pixel
Art XL e successivamente ridotte e quantizzate, secondo la descrizione degli
autori. La dataset card dichiara **Apache-2.0**.

Sono state mantenute le **300 immagini originali** della prima selezione,
con soggetti vari: paesaggi, architettura, monumenti, interni, animali,
veicoli, oggetti, cibo, piante e persone. Immagini, caption e split di queste
300 non sono stati modificati durante l'ampliamento. E stata scelta una
sola variante per indice di prompt.

Sono state aggiunte altre **5 scene con persone**, distinte dalle 300:
`parti_0653_v3.png`, `parti_0687_v0.png`, `parti_1090_v0.png`,
`parti_1092_v1.png`, `parti_1099_v0.png`. Mostrano pubblico ai fuochi,
trasporti affollati, bambini su un divano, una parata e persone in auto.

Revisione fissata:
[`9cbd3463039ea84aa3a28e3f090432d82750fc30`](https://huggingface.co/datasets/nerijs/pixelparti-128-v0.1/tree/9cbd3463039ea84aa3a28e3f090432d82750fc30).

### Multi-Style Stylized Image Pairs — 25 immagini

[Dataset originale: yufan/image_style_transfer_GPTImage2](https://huggingface.co/datasets/yufan/image_style_transfer_GPTImage2).

La fonte contiene fotografie `raw` e varianti sintetiche in diversi stili.
Per il training e stata estratta **solo la colonna `Pixel_Art`**, senza
includere le fotografie raw o le altre stilizzazioni. La dataset card
dichiara **Apache-2.0**.

Sono state selezionate **25 immagini di persone inserite in un ambiente**:
consegne, sport, panchine, bar, imbarcazioni, giardini, serre, paesaggi,
abitazioni e piccoli gruppi. La selezione precedente, dominata da primi
piani, e stata sostituita: gli 80 ritratti della prima selezione Multi-Style
non sono nel dataset corrente. Sono rimasti anche mezzi busti con contesto
visibile, non soltanto persone a figura intera.

I file sono JPEG nativi: **1024x1024, 1024x682 o 682x1024**. Le caption sono
state scritte e verificate visivamente. Sono stati controllati anche i crop
centrali del training per evitare la perdita del contesto scelto.

Revisione fissata:
[`c469c2930ed7dc939ee8e43ef7fa890b0c0491e0`](https://huggingface.co/datasets/yufan/image_style_transfer_GPTImage2/tree/c469c2930ed7dc939ee8e43ef7fa890b0c0491e0).

### Pixel Art Image Dataset — 70 immagini

[Dataset originale: mks0813/pixel_image_dataset](https://huggingface.co/datasets/mks0813/pixel_image_dataset).

Scene sintetiche PNG a **1024x1024**. La card indica generazione con
Tongyi-MAI/Z-Image-Turbo, soggetti di ispirazione coreana e licenza **MIT**.

Sono state selezionate **70 scene con persone e contesto**, privilegiando
la varieta: feste e mercati, vie frequentate, famiglie e amici, musicisti,
attivita sportive, lavori quotidiani, gite, paesaggi ampi e figure lontane.
Le caption usate nel training descrivono il contenuto visibile; non copiano
il suffisso tecnico `pxlstl` dei prompt generativi sorgenti.

Revisione fissata:
[`544aced8db45619909c531588dd7cce50639d4c1`](https://huggingface.co/datasets/mks0813/pixel_image_dataset/tree/544aced8db45619909c531588dd7cce50639d4c1).

Le licenze riportate sono quelle dichiarate dagli autori delle fonti; non
costituiscono una verifica indipendente dei diritti di ogni immagine.

## Composizione e criteri di selezione

Il dataset e costituito da **300 immagini originali congelate + 100 scene
aggiuntive con persone**. Gli esempi con persone gia presenti nelle prime
300 non sono conteggiati di nuovo per raggiungere quota 100.

Le 100 scene aggiuntive provengono da 25 Multi-Style, 5 PixelParti e 70 MKS:

- **10 scene con folla**: pubblico, feste, mercati, trasporti e attivita collettive;
- **42 con piccoli gruppi**: famiglie, amici e persone impegnate insieme;
- **48 con una persona principale** e ambiente riconoscibile;
- **24 delle 100** includono figure lontane/inquadrature ampie. Questa
  caratteristica attraversa i tre gruppi precedenti e non va sommata al totale.

Le etichette di composizione e distanza sono qualitative, basate su revisione
visiva. Alcune scene con persona principale contengono figure secondarie.
Sono stati preferiti contenuti utili al trasferimento di fotografie, senza
limitare la selezione a facce isolate, sprite o icone. Rimangono comunque
esempi illustrativi, viste dall'alto e differenze di stile fra le fonti.

| Fonte | Train | Validation | Totale |
|---|---:|---:|---:|
| PixelParti128 | 275 | 30 | 305 |
| Multi-Style Stylized Image Pairs | 22 | 3 | 25 |
| Pixel Art Image Dataset (mks0813) | 63 | 7 | 70 |
| **Totale** | **360** | **40** | **400** |

I 300 originali hanno split 270/30; le 100 scene aggiuntive hanno split 90/10.
Nel validation di queste ultime sono presenti 1 folla, 4 piccoli gruppi e
5 scene con persona principale.

## Struttura e utilizzo nel training

```text
data/pixel_art/
├── Dataset_LoRA.md
├── train/           # 360 immagini + metadata.jsonl
└── validation/      # 40 immagini + metadata.jsonl
```

Il trainer legge soltanto le immagini e i `metadata.jsonl` degli split.
Ogni riga contiene almeno `file_name` e `text`; i nomi dei file sono relativi
al rispettivo split. Questo documento non viene letto dal training.
Le caption effettive e l'assegnazione train/validation restano quindi nei
metadata; la tabella finale serve come inventario delle fonti.

Le immagini conservano i byte nativi, senza ricompressione. Durante il
training il loader applica orientamento EXIF e RGB, resize proporzionale
nearest-neighbor, crop centrale a 512x512 e normalizzazione in [-1, 1].
La validation non aggiorna i pesi. Usare gli stessi file, caption e split
per il confronto dei rank 4/8/16/32 e tenere separate le foto di inferenza.

```bash
python3 main.py --lora lora_rank4.toml train --check-data
```

## Integrita e tracciabilita

Revisione della selezione: **`people100-plus-frozen300-2026-11-06`**.
Controllati i 400 file contro gli hash del manifest prima del suo ritiro.
La verifica precedente non ha trovato duplicati RGB esatti ne coppie con
distanza dHash <=4 fra split. Le scene note Multi-Style 656/657 e i tre
festival MKS 10/102/400 sono nello stesso split. Questi filtri non escludono
ogni identita ripetuta, duplicato semantico o overlap con il benchmark RQ1.

Firme del loader, che includono nomi, caption e byte delle immagini:

- train: `87246eb9966643cc01862bee1d17a19161b9fbaef500dc1d42aa0d98b6798ae4`;
- validation: `86d2d3053fce97d13cd8c514096eb97aef5ed5263afaf8edd0c0e4e01716d5f7`.

La corrispondenza fra nomi locali e fonti e recuperabile dalle revisioni fissate:

- `multi_NNNN.jpg`: valore `id` NNNN nella colonna `Pixel_Art` di Multi-Style;
- `parti_NNNN_vV.png`: file sorgente `image_N_V.png` di PixelParti;
- `mks_NNNN.png`: indice di riga NNNN, a partire da zero, nel Parquet MKS.

Le caption PixelParti sono state ricostruite dagli indici di
[PartiPrompts](https://github.com/google-research/parti/blob/main/PartiPrompts.tsv),
con controlli e correzioni visive. La corrispondenza e inferita dai nomi,
non garantita da una colonna prompt nella fonte PixelParti. Le altre caption
sono state redatte tramite ispezione visiva. Il prompt sorgente non garantisce
che tutti gli oggetti descritti siano presenti nell'immagine generata.

Hash SHA256 dei Parquet usati per la selezione:

```text
multistyle-0.parquet: a8c32f3af726b4c379e3738a7b5749302464785b7f9e7b3423aeadc4f13516d8
multistyle-1.parquet: 861c3403c021686fe772a1b3767c304b73fe9b782fc15108906bf7d6c5397757
pixelparti.parquet: abc7f28ba246b2a51c96e832d05eb2799f23bff01e07382d9e983d069c7149b7
mks0813-544aced8.parquet: ac784434d286f6203485f672119a1f797eecd0454016e7e10438b9018237364a
```

Gli stili e le risoluzioni delle fonti sono misti; il dataset non garantisce
una griglia Pixel Art uniforme. La qualita del LoRA va valutata sulle immagini
generate e non dedotta dalla sola training o validation loss.

## Elenco completo delle immagini selezionate

Le righe sono raggruppate per dataset e ordinate per nome. La colonna `type`
indica l'estensione del file locale; la posizione nello split e nei metadata.

| name_file | type | from dataset |
|---|---|---|
| parti_0042_v0.png | png | PixelParti128 |
| parti_0046_v0.png | png | PixelParti128 |
| parti_0054_v0.png | png | PixelParti128 |
| parti_0059_v0.png | png | PixelParti128 |
| parti_0062_v0.png | png | PixelParti128 |
| parti_0063_v0.png | png | PixelParti128 |
| parti_0064_v3.png | png | PixelParti128 |
| parti_0065_v0.png | png | PixelParti128 |
| parti_0066_v0.png | png | PixelParti128 |
| parti_0067_v0.png | png | PixelParti128 |
| parti_0068_v2.png | png | PixelParti128 |
| parti_0070_v0.png | png | PixelParti128 |
| parti_0071_v0.png | png | PixelParti128 |
| parti_0072_v0.png | png | PixelParti128 |
| parti_0073_v0.png | png | PixelParti128 |
| parti_0074_v0.png | png | PixelParti128 |
| parti_0077_v0.png | png | PixelParti128 |
| parti_0078_v0.png | png | PixelParti128 |
| parti_0079_v0.png | png | PixelParti128 |
| parti_0080_v0.png | png | PixelParti128 |
| parti_0086_v0.png | png | PixelParti128 |
| parti_0089_v0.png | png | PixelParti128 |
| parti_0090_v0.png | png | PixelParti128 |
| parti_0091_v0.png | png | PixelParti128 |
| parti_0093_v0.png | png | PixelParti128 |
| parti_0094_v2.png | png | PixelParti128 |
| parti_0097_v0.png | png | PixelParti128 |
| parti_0098_v0.png | png | PixelParti128 |
| parti_0099_v0.png | png | PixelParti128 |
| parti_0100_v0.png | png | PixelParti128 |
| parti_0101_v0.png | png | PixelParti128 |
| parti_0103_v0.png | png | PixelParti128 |
| parti_0105_v0.png | png | PixelParti128 |
| parti_0108_v0.png | png | PixelParti128 |
| parti_0109_v0.png | png | PixelParti128 |
| parti_0110_v0.png | png | PixelParti128 |
| parti_0112_v0.png | png | PixelParti128 |
| parti_0115_v0.png | png | PixelParti128 |
| parti_0116_v0.png | png | PixelParti128 |
| parti_0117_v0.png | png | PixelParti128 |
| parti_0118_v0.png | png | PixelParti128 |
| parti_0128_v0.png | png | PixelParti128 |
| parti_0129_v1.png | png | PixelParti128 |
| parti_0134_v0.png | png | PixelParti128 |
| parti_0135_v0.png | png | PixelParti128 |
| parti_0136_v0.png | png | PixelParti128 |
| parti_0137_v0.png | png | PixelParti128 |
| parti_0140_v0.png | png | PixelParti128 |
| parti_0144_v0.png | png | PixelParti128 |
| parti_0146_v0.png | png | PixelParti128 |
| parti_0149_v0.png | png | PixelParti128 |
| parti_0150_v0.png | png | PixelParti128 |
| parti_0152_v3.png | png | PixelParti128 |
| parti_0155_v0.png | png | PixelParti128 |
| parti_0156_v0.png | png | PixelParti128 |
| parti_0159_v1.png | png | PixelParti128 |
| parti_0160_v0.png | png | PixelParti128 |
| parti_0165_v0.png | png | PixelParti128 |
| parti_0168_v0.png | png | PixelParti128 |
| parti_0169_v0.png | png | PixelParti128 |
| parti_0174_v0.png | png | PixelParti128 |
| parti_0176_v0.png | png | PixelParti128 |
| parti_0178_v1.png | png | PixelParti128 |
| parti_0179_v1.png | png | PixelParti128 |
| parti_0181_v0.png | png | PixelParti128 |
| parti_0191_v0.png | png | PixelParti128 |
| parti_0192_v0.png | png | PixelParti128 |
| parti_0193_v0.png | png | PixelParti128 |
| parti_0194_v0.png | png | PixelParti128 |
| parti_0196_v0.png | png | PixelParti128 |
| parti_0197_v0.png | png | PixelParti128 |
| parti_0198_v0.png | png | PixelParti128 |
| parti_0199_v0.png | png | PixelParti128 |
| parti_0201_v1.png | png | PixelParti128 |
| parti_0202_v0.png | png | PixelParti128 |
| parti_0203_v0.png | png | PixelParti128 |
| parti_0205_v0.png | png | PixelParti128 |
| parti_0207_v0.png | png | PixelParti128 |
| parti_0219_v0.png | png | PixelParti128 |
| parti_0220_v0.png | png | PixelParti128 |
| parti_0221_v0.png | png | PixelParti128 |
| parti_0222_v0.png | png | PixelParti128 |
| parti_0225_v0.png | png | PixelParti128 |
| parti_0227_v0.png | png | PixelParti128 |
| parti_0228_v0.png | png | PixelParti128 |
| parti_0230_v0.png | png | PixelParti128 |
| parti_0232_v0.png | png | PixelParti128 |
| parti_0234_v0.png | png | PixelParti128 |
| parti_0235_v0.png | png | PixelParti128 |
| parti_0236_v0.png | png | PixelParti128 |
| parti_0238_v0.png | png | PixelParti128 |
| parti_0241_v0.png | png | PixelParti128 |
| parti_0242_v0.png | png | PixelParti128 |
| parti_0244_v0.png | png | PixelParti128 |
| parti_0246_v0.png | png | PixelParti128 |
| parti_0249_v3.png | png | PixelParti128 |
| parti_0250_v0.png | png | PixelParti128 |
| parti_0251_v0.png | png | PixelParti128 |
| parti_0252_v0.png | png | PixelParti128 |
| parti_0253_v0.png | png | PixelParti128 |
| parti_0255_v0.png | png | PixelParti128 |
| parti_0257_v0.png | png | PixelParti128 |
| parti_0258_v0.png | png | PixelParti128 |
| parti_0260_v0.png | png | PixelParti128 |
| parti_0261_v0.png | png | PixelParti128 |
| parti_0262_v0.png | png | PixelParti128 |
| parti_0264_v0.png | png | PixelParti128 |
| parti_0271_v0.png | png | PixelParti128 |
| parti_0362_v0.png | png | PixelParti128 |
| parti_0364_v0.png | png | PixelParti128 |
| parti_0365_v0.png | png | PixelParti128 |
| parti_0366_v0.png | png | PixelParti128 |
| parti_0373_v0.png | png | PixelParti128 |
| parti_0375_v0.png | png | PixelParti128 |
| parti_0378_v0.png | png | PixelParti128 |
| parti_0379_v0.png | png | PixelParti128 |
| parti_0399_v0.png | png | PixelParti128 |
| parti_0400_v0.png | png | PixelParti128 |
| parti_0401_v0.png | png | PixelParti128 |
| parti_0402_v0.png | png | PixelParti128 |
| parti_0413_v0.png | png | PixelParti128 |
| parti_0423_v0.png | png | PixelParti128 |
| parti_0433_v0.png | png | PixelParti128 |
| parti_0440_v0.png | png | PixelParti128 |
| parti_0442_v0.png | png | PixelParti128 |
| parti_0443_v0.png | png | PixelParti128 |
| parti_0446_v0.png | png | PixelParti128 |
| parti_0447_v0.png | png | PixelParti128 |
| parti_0450_v0.png | png | PixelParti128 |
| parti_0454_v0.png | png | PixelParti128 |
| parti_0456_v0.png | png | PixelParti128 |
| parti_0459_v0.png | png | PixelParti128 |
| parti_0460_v0.png | png | PixelParti128 |
| parti_0461_v0.png | png | PixelParti128 |
| parti_0464_v0.png | png | PixelParti128 |
| parti_0465_v0.png | png | PixelParti128 |
| parti_0468_v0.png | png | PixelParti128 |
| parti_0469_v3.png | png | PixelParti128 |
| parti_0470_v0.png | png | PixelParti128 |
| parti_0471_v0.png | png | PixelParti128 |
| parti_0474_v0.png | png | PixelParti128 |
| parti_0478_v0.png | png | PixelParti128 |
| parti_0479_v0.png | png | PixelParti128 |
| parti_0480_v0.png | png | PixelParti128 |
| parti_0482_v0.png | png | PixelParti128 |
| parti_0483_v0.png | png | PixelParti128 |
| parti_0484_v0.png | png | PixelParti128 |
| parti_0485_v0.png | png | PixelParti128 |
| parti_0487_v0.png | png | PixelParti128 |
| parti_0489_v0.png | png | PixelParti128 |
| parti_0490_v0.png | png | PixelParti128 |
| parti_0493_v0.png | png | PixelParti128 |
| parti_0511_v0.png | png | PixelParti128 |
| parti_0512_v0.png | png | PixelParti128 |
| parti_0519_v0.png | png | PixelParti128 |
| parti_0520_v0.png | png | PixelParti128 |
| parti_0525_v0.png | png | PixelParti128 |
| parti_0535_v0.png | png | PixelParti128 |
| parti_0536_v0.png | png | PixelParti128 |
| parti_0538_v0.png | png | PixelParti128 |
| parti_0539_v0.png | png | PixelParti128 |
| parti_0545_v0.png | png | PixelParti128 |
| parti_0554_v0.png | png | PixelParti128 |
| parti_0555_v0.png | png | PixelParti128 |
| parti_0556_v0.png | png | PixelParti128 |
| parti_0557_v0.png | png | PixelParti128 |
| parti_0564_v0.png | png | PixelParti128 |
| parti_0565_v2.png | png | PixelParti128 |
| parti_0566_v0.png | png | PixelParti128 |
| parti_0567_v0.png | png | PixelParti128 |
| parti_0568_v0.png | png | PixelParti128 |
| parti_0569_v0.png | png | PixelParti128 |
| parti_0571_v0.png | png | PixelParti128 |
| parti_0572_v0.png | png | PixelParti128 |
| parti_0573_v0.png | png | PixelParti128 |
| parti_0574_v0.png | png | PixelParti128 |
| parti_0575_v0.png | png | PixelParti128 |
| parti_0578_v0.png | png | PixelParti128 |
| parti_0580_v0.png | png | PixelParti128 |
| parti_0581_v0.png | png | PixelParti128 |
| parti_0586_v0.png | png | PixelParti128 |
| parti_0587_v0.png | png | PixelParti128 |
| parti_0588_v0.png | png | PixelParti128 |
| parti_0591_v0.png | png | PixelParti128 |
| parti_0593_v0.png | png | PixelParti128 |
| parti_0594_v0.png | png | PixelParti128 |
| parti_0597_v0.png | png | PixelParti128 |
| parti_0600_v0.png | png | PixelParti128 |
| parti_0602_v0.png | png | PixelParti128 |
| parti_0603_v0.png | png | PixelParti128 |
| parti_0609_v0.png | png | PixelParti128 |
| parti_0622_v0.png | png | PixelParti128 |
| parti_0623_v0.png | png | PixelParti128 |
| parti_0625_v0.png | png | PixelParti128 |
| parti_0626_v0.png | png | PixelParti128 |
| parti_0631_v0.png | png | PixelParti128 |
| parti_0633_v0.png | png | PixelParti128 |
| parti_0634_v0.png | png | PixelParti128 |
| parti_0638_v0.png | png | PixelParti128 |
| parti_0645_v0.png | png | PixelParti128 |
| parti_0646_v0.png | png | PixelParti128 |
| parti_0652_v0.png | png | PixelParti128 |
| parti_0653_v3.png | png | PixelParti128 |
| parti_0654_v0.png | png | PixelParti128 |
| parti_0655_v0.png | png | PixelParti128 |
| parti_0657_v0.png | png | PixelParti128 |
| parti_0658_v0.png | png | PixelParti128 |
| parti_0661_v0.png | png | PixelParti128 |
| parti_0671_v0.png | png | PixelParti128 |
| parti_0672_v0.png | png | PixelParti128 |
| parti_0673_v0.png | png | PixelParti128 |
| parti_0676_v0.png | png | PixelParti128 |
| parti_0677_v0.png | png | PixelParti128 |
| parti_0679_v0.png | png | PixelParti128 |
| parti_0682_v0.png | png | PixelParti128 |
| parti_0685_v0.png | png | PixelParti128 |
| parti_0687_v0.png | png | PixelParti128 |
| parti_0691_v0.png | png | PixelParti128 |
| parti_0693_v0.png | png | PixelParti128 |
| parti_0695_v0.png | png | PixelParti128 |
| parti_0890_v0.png | png | PixelParti128 |
| parti_0894_v0.png | png | PixelParti128 |
| parti_0896_v0.png | png | PixelParti128 |
| parti_0900_v0.png | png | PixelParti128 |
| parti_0907_v0.png | png | PixelParti128 |
| parti_0911_v0.png | png | PixelParti128 |
| parti_0912_v0.png | png | PixelParti128 |
| parti_0941_v0.png | png | PixelParti128 |
| parti_0943_v0.png | png | PixelParti128 |
| parti_0945_v0.png | png | PixelParti128 |
| parti_0951_v2.png | png | PixelParti128 |
| parti_0952_v0.png | png | PixelParti128 |
| parti_0953_v0.png | png | PixelParti128 |
| parti_0954_v0.png | png | PixelParti128 |
| parti_0955_v0.png | png | PixelParti128 |
| parti_0956_v0.png | png | PixelParti128 |
| parti_0962_v0.png | png | PixelParti128 |
| parti_0963_v0.png | png | PixelParti128 |
| parti_0968_v0.png | png | PixelParti128 |
| parti_0969_v0.png | png | PixelParti128 |
| parti_0970_v0.png | png | PixelParti128 |
| parti_0972_v0.png | png | PixelParti128 |
| parti_0973_v0.png | png | PixelParti128 |
| parti_0975_v0.png | png | PixelParti128 |
| parti_0976_v0.png | png | PixelParti128 |
| parti_0977_v0.png | png | PixelParti128 |
| parti_1013_v0.png | png | PixelParti128 |
| parti_1043_v0.png | png | PixelParti128 |
| parti_1065_v0.png | png | PixelParti128 |
| parti_1068_v0.png | png | PixelParti128 |
| parti_1070_v0.png | png | PixelParti128 |
| parti_1075_v3.png | png | PixelParti128 |
| parti_1079_v0.png | png | PixelParti128 |
| parti_1080_v0.png | png | PixelParti128 |
| parti_1083_v0.png | png | PixelParti128 |
| parti_1085_v0.png | png | PixelParti128 |
| parti_1087_v0.png | png | PixelParti128 |
| parti_1089_v0.png | png | PixelParti128 |
| parti_1090_v0.png | png | PixelParti128 |
| parti_1092_v1.png | png | PixelParti128 |
| parti_1097_v0.png | png | PixelParti128 |
| parti_1098_v0.png | png | PixelParti128 |
| parti_1099_v0.png | png | PixelParti128 |
| parti_1100_v0.png | png | PixelParti128 |
| parti_1104_v0.png | png | PixelParti128 |
| parti_1107_v0.png | png | PixelParti128 |
| parti_1114_v0.png | png | PixelParti128 |
| parti_1116_v0.png | png | PixelParti128 |
| parti_1118_v0.png | png | PixelParti128 |
| parti_1119_v0.png | png | PixelParti128 |
| parti_1123_v0.png | png | PixelParti128 |
| parti_1124_v0.png | png | PixelParti128 |
| parti_1125_v0.png | png | PixelParti128 |
| parti_1126_v0.png | png | PixelParti128 |
| parti_1127_v0.png | png | PixelParti128 |
| parti_1128_v0.png | png | PixelParti128 |
| parti_1133_v0.png | png | PixelParti128 |
| parti_1143_v0.png | png | PixelParti128 |
| parti_1145_v0.png | png | PixelParti128 |
| parti_1150_v0.png | png | PixelParti128 |
| parti_1152_v0.png | png | PixelParti128 |
| parti_1154_v0.png | png | PixelParti128 |
| parti_1155_v0.png | png | PixelParti128 |
| parti_1157_v0.png | png | PixelParti128 |
| parti_1158_v0.png | png | PixelParti128 |
| parti_1159_v0.png | png | PixelParti128 |
| parti_1174_v0.png | png | PixelParti128 |
| parti_1175_v0.png | png | PixelParti128 |
| parti_1176_v0.png | png | PixelParti128 |
| parti_1177_v0.png | png | PixelParti128 |
| parti_1178_v0.png | png | PixelParti128 |
| parti_1179_v0.png | png | PixelParti128 |
| parti_1180_v0.png | png | PixelParti128 |
| parti_1187_v0.png | png | PixelParti128 |
| parti_1188_v1.png | png | PixelParti128 |
| parti_1194_v0.png | png | PixelParti128 |
| parti_1196_v0.png | png | PixelParti128 |
| parti_1197_v0.png | png | PixelParti128 |
| parti_1198_v0.png | png | PixelParti128 |
| parti_1200_v0.png | png | PixelParti128 |
| parti_1201_v0.png | png | PixelParti128 |
| parti_1202_v0.png | png | PixelParti128 |
| parti_1203_v0.png | png | PixelParti128 |
| parti_1204_v0.png | png | PixelParti128 |
| parti_1208_v0.png | png | PixelParti128 |
| multi_0500.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0515.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0516.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0518.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0527.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0549.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0551.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0560.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0570.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0573.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0578.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0579.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0581.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0585.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0586.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0592.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0596.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0620.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0631.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0638.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0654.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0655.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0656.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0657.jpg | jpg | Multi-Style Stylized Image Pairs |
| multi_0658.jpg | jpg | Multi-Style Stylized Image Pairs |
| mks_0002.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0006.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0007.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0010.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0013.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0018.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0019.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0026.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0027.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0033.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0035.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0036.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0040.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0043.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0044.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0051.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0053.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0057.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0058.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0060.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0064.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0066.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0068.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0076.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0078.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0079.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0087.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0088.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0096.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0099.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0102.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0105.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0108.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0115.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0122.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0123.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0128.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0135.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0136.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0141.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0146.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0153.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0160.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0168.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0173.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0187.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0191.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0192.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0195.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0196.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0222.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0234.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0241.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0250.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0257.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0277.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0283.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0285.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0294.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0306.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0338.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0347.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0354.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0371.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0399.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0400.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0430.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0460.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0479.png | png | Pixel Art Image Dataset (mks0813) |
| mks_0483.png | png | Pixel Art Image Dataset (mks0813) |
