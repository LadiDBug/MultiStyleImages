# Dataset per il LoRA Cyberpunk

Aggiornato al **17 novembre 2026**. Dataset locale di **400 immagini JPEG con
caption inglesi**, suddivise in **360 train** e **40 validation**, per LoRA della
U-Net di Stable Diffusion 1.5. **Il training non è stato eseguito.**

La selezione segue [Dataset_LoRA.md del Pixel Art](../pixel_art/Dataset_LoRA.md):
300 esempi vari e 100 scene aggiuntive con persone. Le due parti sono disgiunte;
le eventuali persone nelle 300 immagini non contribuiscono alle quote delle 100.
Il dataset Pixel Art non è stato modificato.

## Fonte e definizione dello stile

Fonte unica: [adrianmele/MegaStyle-1.4M](https://huggingface.co/datasets/adrianmele/MegaStyle-1.4M),
revisione fissata
[`141be95c7ba9724caeca38a973e9bdcd4c78bdac`](https://huggingface.co/datasets/adrianmele/MegaStyle-1.4M/tree/141be95c7ba9724caeca38a973e9bdcd4c78bdac).
La scheda degli autori descrive immagini sintetiche ottenute con Qwen-Image.
Una copia della scheda è in [SOURCE_README.md](SOURCE_README.md).

Sono stati esaminati 1.200 candidati provenienti da 150 varianti di stile che
contengono `cyberpunk` nella descrizione. Le varianti sono state individuate
nell'indice `style_indices.pkl` della revisione fissata; sono stati evitati
stili esplicitamente etichettati come pixel, anime, abstract, chibi, watercolor,
sketch o pop art. La selezione finale conserva 400 diversi `content_id`,
ricavati dalla componente `cNNNN` degli identificativi `sNNNN_cNNNN`.
Non sono state incluse più varianti dello stesso identificativo di contenuto.

L'aspetto ricercato combina materiali futuristici, superfici metalliche,
riflessi, accenti luminosi ciano/magenta o ambra e ombre atmosferiche.
Comprende città, interni, paesaggi, persone, animali, oggetti e cibo reinterpretati
in questo stile: non è una raccolta di soli ritratti di cyborg o strade notturne.
Sono presenti sia rendering realistici sia illustrazioni. Intensità del neon,
palette e resa dei materiali variano: non viene garantito uno stile uniforme.
Il termine Cyberpunk non indica qui esclusivamente il videogioco Cyberpunk 2077.

**Licenza della fonte:** [License Term of MegaStyle](SOURCE_LICENSE.txt),
dichiarata `other` nella card. Consente ricerca e didattica, esclude usi
commerciali e di produzione e richiede di mantenere l'avviso di copyright e
permesso. Il testo originale è incluso integralmente. Non è equivalente alle
licenze Apache-2.0/MIT delle fonti Pixel Art. Questa selezione replica i criteri
visivi e numerici; **non replica la disponibilità di una licenza permissiva**.
La licenza riportata è quella della fonte, senza verifica indipendente dei
diritti di ogni singola immagine.

Multi-Style GPTImage2, già usato per Pixel Art, è stato controllato ma non ha
una colonna Cyberpunk. Non sono state importate le sue altre stilizzazioni.

## Composizione e criteri di selezione

| Sottoinsieme | Train | Validation | Totale |
|---|---:|---:|---:|
| `base300`: contenuti vari | 270 | 30 | 300 |
| `people100`: ulteriori scene con persone | 90 | 10 | 100 |
| **Totale** | **360** | **40** | **400** |

| Composizione delle 100 scene aggiuntive | Train | Validation | Totale |
|---|---:|---:|---:|
| Folla, `crowd` | 9 | 1 | 10 |
| Piccolo gruppo, `small_group` | 38 | 4 | 42 |
| Persona principale con ambiente, `single_main` | 43 | 5 | 48 |

**24 delle 100** hanno `wide=1`: figure lontane o inquadratura ampia con
ambiente leggibile. È una proprietà trasversale alle tre categorie, non un
quarto gruppo. Le annotazioni sono qualitative. Una persona principale può
essere accompagnata da figure secondarie; per folla si intende anche un
pubblico sullo sfondo. Alcuni soggetti sono umanoidi con armature o protesi.

La selezione privilegia soggetti utili al trasferimento di fotografie:
paesaggi montani e costieri, architetture e monumenti, strade e infrastrutture,
interni, attività quotidiane, sport, trasporti, animali, piante, cibo e oggetti.
Sono stati scartati ritratti isolati dominanti, collage di volti, grafici,
immagini con messaggi di errore e molte varianti ridondanti. Rimangono mezzi
busti ambientati e oggetti singoli, come nel riferimento; non si pretende che
tutte le immagini contengano scene ampie. Restano occasionali scritte,
loghi sintetici ed errori anatomici o geometrici propri della generazione.

Le 300 immagini vengono congelate con questa prima selezione: non sono le
300 immagini Pixel Art trasformate. Non vengono replicate le proporzioni fra
le fonti originarie né i soggetti immagine per immagine. Le categorie di
contenuto delle 300 non avevano quote numeriche nel documento di riferimento;
non viene dichiarato un bilanciamento semantico esatto fra i due dataset.

## Immagini, caption e crop

Tutti i file selezionati sono **512×512**. Sono i JPEG esposti dal viewer
Hugging Face della revisione indicata, conservati senza ricompressione locale.
**Non si afferma l'identità dei byte con le immagini incorporate nei Parquet:**
il viewer può produrre una propria codifica JPEG. Gli SHA256 del manifest
identificano precisamente i file usati qui.

Le caption iniziano con `Cyberpunk style.` e descrivono il contenuto. Partono
dalla colonna `content` della fonte, con correzioni basate sulle anteprime
visive: nomi di celebrità e descrizioni non riscontrate non sono stati mantenuti
quando rilevati. Non viene copiata la lunga colonna tecnica `style`. Le caption
sorgenti sono conservate separatamente nel manifest per la tracciabilità,
comprese eventuali imprecisioni; **soltanto `text` entra nel training**.
Le caption contengono **10–19 token CLIP**, entro il limite di 77, senza
troncamento nel controllo locale.
La revisione visiva è avvenuta tramite tavole di contatto e ingrandimenti
mirati; non è una certificazione dell'accuratezza di ogni dettaglio.

Il loader esistente `PixelArtDataset` accetta questo formato senza modifiche.
I config Cyberpunk usano **Lanczos**, invece di nearest-neighbor utilizzato
per la griglia Pixel Art. Applica orientamento EXIF, RGB, resize proporzionale,
crop centrale 512×512 e normalizzazione in [-1,1]. Dato che le immagini sono
già quadrate alla risoluzione di training, **originale e crop coincidono**.
Non vengono applicati crop casuali o flip. Non servono coppie fotografia/target.
La validation non aggiorna i pesi; le fotografie di inferenza restano separate.

## Struttura e uso

```text
data/cyberpunk/
├── Dataset_LoRA.md
├── SOURCE_LICENSE.txt
├── SOURCE_README.md
├── selection.csv             # fonti, revisione, contenuto, caption, split e hash
├── validation_report.json    # controlli ripetibili
├── review.html               # galleria locale di tutte le immagini
├── train/                   # 360 JPEG + metadata.jsonl
└── validation/              # 40 JPEG + metadata.jsonl
```

Ogni riga dei metadata contiene `file_name` e `text`; il nome è relativo allo
split. Il trainer non legge questo Markdown. La galleria è consultabile
aprendo [review.html](review.html) in un browser locale.

Controllo offline della selezione congelata:

```bash
python3 scripts/build_cyberpunk_dataset.py
python3 main.py --lora lora_cyberpunk_rank4.toml train --check-data
```

Ripristino dei file mancanti dalla cache o dal viewer della fonte, con verifica
degli hash prima della scrittura:

```bash
python3 scripts/build_cyberpunk_dataset.py --download
```

Le URL degli asset nel manifest sono prive delle firme temporanee. In assenza
di cache lo script richiede una URL aggiornata al viewer e verifica ID,
revisione e hash; fallisce se la fonte cambia, diventa indisponibile o il viewer
rigenera JPEG differenti. Non scarica l'intero dataset originale.

Sono disponibili `lora_cyberpunk_rank4.toml`, `lora_cyberpunk_rank8.toml`,
`lora_cyberpunk_rank16.toml` e `lora_cyberpunk_rank32.toml`, con output separati
in `lora/cyberpunk_rank_*`. Lo stile è registrato in `configs/styles.toml`.
Per avviare successivamente un training:

```bash
python3 main.py --lora lora_cyberpunk_rank4.toml train
```

Usare gli stessi file, caption e split nei confronti fra rank.

## Integrità e limiti dei controlli

Revisione della selezione: **`cyberpunk-context400-2026-11-17`**.
I file immagine occupano circa **13,8 MB**.

Controlli completati:

- numerosità, quote 300+100, composizione delle persone e split 90/10;
- decodifica, dimensioni, hash dei byte e dei pixel RGB contro il manifest;
- 400 ID di contenuto distinti, senza varianti del medesimo ID fra split;
- nessun duplicato esatto dei byte o dei pixel RGB; il crop coincide con l'immagine;
- nessuna coppia con dHash ≤ 4 nell'intero dataset, quindi anche fra split;
- nessuna corrispondenza RGB esatta o dHash ≤ 4 con le 108 immagini locali
  leggibili in `data/test_images` al momento del controllo;
- lettura tramite `main.py ... train --check-data`, senza avviare il training;
- preprocessing effettivo di tutti i 400 esempi e tokenizzazione CLIP locale.

I controlli non escludono tutti i duplicati semantici, personaggi ricorrenti,
contaminazioni con benchmark esterni o errori delle caption. Gli ID distinguono
prompt di contenuto, non garantiscono scene semanticamente indipendenti.
La qualità del LoRA va misurata sulle immagini generate, non dedotta dai soli
conteggi o dalla loss.

Firme compatibili con il loader, comprendenti nomi, caption e byte:

- train: `6c2f0c762a29900a6ac88f5495b50cd0a80205acb96a23e222fc7c826f8d7b91`;
- validation: `b6141b0a74ac38aecc8e9b9e83789f30c7a851f7256f2ffcadf0a2a1d70130cc`;

## Inventario completo

Il nome locale conserva l'ID della fonte. `wide=0` nelle 300 immagini non è
un'annotazione negativa: la proprietà è stata assegnata solo alle 100 persone.
Fonte, descrizione di stile, ID di contenuto e hash sono nel CSV.

| File | Split | Sottoinsieme | Composizione | Wide |
|---|---|---|---|---:|
| cyber_s137734_c25785.jpg | train | base300 | — | — |
| cyber_s137734_c330387.jpg | train | base300 | — | — |
| cyber_s137734_c383992.jpg | train | base300 | — | — |
| cyber_s137737_c136436.jpg | train | base300 | — | — |
| cyber_s137737_c238052.jpg | train | base300 | — | — |
| cyber_s137737_c351513.jpg | train | people100 | single_main | 0 |
| cyber_s137737_c379437.jpg | train | people100 | single_main | 1 |
| cyber_s137755_c149651.jpg | train | base300 | — | — |
| cyber_s137755_c325594.jpg | train | people100 | single_main | 0 |
| cyber_s137755_c347906.jpg | train | base300 | — | — |
| cyber_s137758_c32981.jpg | train | people100 | small_group | 1 |
| cyber_s137758_c79257.jpg | train | base300 | — | — |
| cyber_s137779_c106491.jpg | train | people100 | small_group | 0 |
| cyber_s137779_c177128.jpg | train | people100 | small_group | 0 |
| cyber_s137779_c183145.jpg | train | people100 | small_group | 0 |
| cyber_s137779_c55816.jpg | train | base300 | — | — |
| cyber_s137779_c85449.jpg | validation | base300 | — | — |
| cyber_s85516_c115300.jpg | train | base300 | — | — |
| cyber_s85516_c394239.jpg | train | base300 | — | — |
| cyber_s85537_c336160.jpg | train | base300 | — | — |
| cyber_s85576_c166012.jpg | train | base300 | — | — |
| cyber_s85576_c282661.jpg | validation | people100 | single_main | 1 |
| cyber_s85576_c307472.jpg | train | people100 | single_main | 0 |
| cyber_s85608_c164129.jpg | train | base300 | — | — |
| cyber_s85608_c96519.jpg | train | base300 | — | — |
| cyber_s85608_c96952.jpg | train | base300 | — | — |
| cyber_s85639_c249339.jpg | train | base300 | — | — |
| cyber_s85639_c56622.jpg | train | base300 | — | — |
| cyber_s85679_c356339.jpg | train | base300 | — | — |
| cyber_s85716_c152277.jpg | train | base300 | — | — |
| cyber_s85716_c373881.jpg | train | people100 | crowd | 0 |
| cyber_s85716_c63236.jpg | validation | people100 | small_group | 1 |
| cyber_s85731_c399397.jpg | train | base300 | — | — |
| cyber_s85731_c47234.jpg | train | base300 | — | — |
| cyber_s85788_c167416.jpg | train | base300 | — | — |
| cyber_s85788_c250750.jpg | train | base300 | — | — |
| cyber_s85788_c277607.jpg | train | base300 | — | — |
| cyber_s85788_c363651.jpg | train | people100 | small_group | 0 |
| cyber_s85814_c290500.jpg | train | base300 | — | — |
| cyber_s85891_c123457.jpg | validation | base300 | — | — |
| cyber_s85891_c150705.jpg | train | base300 | — | — |
| cyber_s85891_c85210.jpg | train | base300 | — | — |
| cyber_s85891_c96688.jpg | train | base300 | — | — |
| cyber_s85909_c106900.jpg | train | base300 | — | — |
| cyber_s85909_c279757.jpg | train | base300 | — | — |
| cyber_s85909_c45926.jpg | train | base300 | — | — |
| cyber_s86011_c218011.jpg | validation | base300 | — | — |
| cyber_s86011_c329750.jpg | train | base300 | — | — |
| cyber_s86011_c52171.jpg | train | base300 | — | — |
| cyber_s86070_c171371.jpg | train | base300 | — | — |
| cyber_s86070_c256274.jpg | train | base300 | — | — |
| cyber_s86070_c293801.jpg | train | base300 | — | — |
| cyber_s86070_c48327.jpg | train | base300 | — | — |
| cyber_s86079_c243347.jpg | train | base300 | — | — |
| cyber_s86079_c334400.jpg | train | base300 | — | — |
| cyber_s86098_c26905.jpg | train | base300 | — | — |
| cyber_s86271_c121254.jpg | train | people100 | small_group | 1 |
| cyber_s86271_c149665.jpg | train | people100 | single_main | 1 |
| cyber_s86271_c208748.jpg | train | people100 | single_main | 0 |
| cyber_s86271_c216663.jpg | train | people100 | single_main | 0 |
| cyber_s86271_c75997.jpg | train | base300 | — | — |
| cyber_s86274_c106522.jpg | train | people100 | small_group | 0 |
| cyber_s86274_c190284.jpg | train | people100 | small_group | 0 |
| cyber_s86274_c292272.jpg | train | base300 | — | — |
| cyber_s86274_c396076.jpg | train | base300 | — | — |
| cyber_s86274_c48960.jpg | train | base300 | — | — |
| cyber_s86313_c110683.jpg | train | people100 | single_main | 0 |
| cyber_s86313_c265921.jpg | validation | people100 | single_main | 0 |
| cyber_s86313_c274115.jpg | validation | people100 | small_group | 0 |
| cyber_s86313_c274317.jpg | train | people100 | single_main | 0 |
| cyber_s86315_c27972.jpg | train | base300 | — | — |
| cyber_s86315_c92200.jpg | train | people100 | small_group | 0 |
| cyber_s86374_c25950.jpg | train | base300 | — | — |
| cyber_s86374_c367102.jpg | train | people100 | small_group | 0 |
| cyber_s86374_c63922.jpg | train | people100 | small_group | 0 |
| cyber_s86374_c87159.jpg | train | people100 | small_group | 0 |
| cyber_s86375_c191087.jpg | train | base300 | — | — |
| cyber_s86375_c289827.jpg | train | base300 | — | — |
| cyber_s86375_c294039.jpg | train | base300 | — | — |
| cyber_s86416_c360226.jpg | train | base300 | — | — |
| cyber_s86416_c396005.jpg | train | base300 | — | — |
| cyber_s86416_c84688.jpg | train | base300 | — | — |
| cyber_s86447_c153953.jpg | train | base300 | — | — |
| cyber_s86499_c251193.jpg | train | base300 | — | — |
| cyber_s86499_c55926.jpg | train | base300 | — | — |
| cyber_s86529_c143706.jpg | train | base300 | — | — |
| cyber_s86529_c162286.jpg | train | base300 | — | — |
| cyber_s86529_c250590.jpg | train | base300 | — | — |
| cyber_s86529_c295323.jpg | train | base300 | — | — |
| cyber_s86543_c11931.jpg | train | people100 | crowd | 0 |
| cyber_s86543_c327941.jpg | train | base300 | — | — |
| cyber_s86543_c386108.jpg | validation | base300 | — | — |
| cyber_s86565_c252980.jpg | validation | base300 | — | — |
| cyber_s86565_c285216.jpg | train | base300 | — | — |
| cyber_s86566_c207244.jpg | train | base300 | — | — |
| cyber_s86574_c313802.jpg | train | base300 | — | — |
| cyber_s86574_c334298.jpg | train | base300 | — | — |
| cyber_s86574_c76552.jpg | validation | base300 | — | — |
| cyber_s86578_c302503.jpg | train | base300 | — | — |
| cyber_s86578_c386265.jpg | train | base300 | — | — |
| cyber_s86581_c373867.jpg | train | base300 | — | — |
| cyber_s86581_c96409.jpg | train | base300 | — | — |
| cyber_s86615_c217943.jpg | train | people100 | single_main | 0 |
| cyber_s86615_c233135.jpg | train | base300 | — | — |
| cyber_s86615_c266263.jpg | train | people100 | small_group | 0 |
| cyber_s86615_c298183.jpg | validation | people100 | small_group | 0 |
| cyber_s86615_c300448.jpg | train | people100 | small_group | 1 |
| cyber_s86619_c396222.jpg | train | base300 | — | — |
| cyber_s86619_c73861.jpg | train | base300 | — | — |
| cyber_s86649_c233676.jpg | train | base300 | — | — |
| cyber_s86649_c255002.jpg | train | base300 | — | — |
| cyber_s86649_c396315.jpg | train | base300 | — | — |
| cyber_s86704_c156628.jpg | validation | base300 | — | — |
| cyber_s86704_c173652.jpg | train | base300 | — | — |
| cyber_s86704_c192983.jpg | train | base300 | — | — |
| cyber_s86704_c302872.jpg | train | base300 | — | — |
| cyber_s86706_c103204.jpg | validation | base300 | — | — |
| cyber_s86706_c118892.jpg | train | base300 | — | — |
| cyber_s86710_c242655.jpg | train | base300 | — | — |
| cyber_s86710_c332152.jpg | train | base300 | — | — |
| cyber_s86710_c389337.jpg | train | base300 | — | — |
| cyber_s86719_c12328.jpg | train | people100 | small_group | 0 |
| cyber_s86719_c130738.jpg | train | people100 | single_main | 0 |
| cyber_s86719_c142353.jpg | train | base300 | — | — |
| cyber_s86719_c162227.jpg | train | base300 | — | — |
| cyber_s86719_c272010.jpg | train | people100 | small_group | 0 |
| cyber_s86719_c272935.jpg | train | people100 | single_main | 0 |
| cyber_s86719_c279986.jpg | train | base300 | — | — |
| cyber_s86719_c373093.jpg | train | people100 | crowd | 0 |
| cyber_s86740_c177412.jpg | train | base300 | — | — |
| cyber_s86740_c276897.jpg | train | base300 | — | — |
| cyber_s86808_c202686.jpg | train | base300 | — | — |
| cyber_s86808_c37502.jpg | train | base300 | — | — |
| cyber_s86835_c169128.jpg | validation | base300 | — | — |
| cyber_s86835_c295335.jpg | train | base300 | — | — |
| cyber_s86835_c375030.jpg | validation | base300 | — | — |
| cyber_s86844_c252827.jpg | train | base300 | — | — |
| cyber_s86844_c35287.jpg | train | base300 | — | — |
| cyber_s86886_c248810.jpg | train | base300 | — | — |
| cyber_s86886_c350307.jpg | train | base300 | — | — |
| cyber_s86886_c394716.jpg | train | base300 | — | — |
| cyber_s86906_c148946.jpg | train | base300 | — | — |
| cyber_s86906_c213478.jpg | train | base300 | — | — |
| cyber_s86906_c66912.jpg | train | base300 | — | — |
| cyber_s86918_c1598.jpg | train | people100 | single_main | 1 |
| cyber_s86918_c236147.jpg | validation | people100 | single_main | 1 |
| cyber_s86918_c266994.jpg | train | people100 | small_group | 0 |
| cyber_s86918_c386876.jpg | validation | people100 | single_main | 0 |
| cyber_s86918_c48492.jpg | train | base300 | — | — |
| cyber_s86918_c93299.jpg | train | people100 | single_main | 0 |
| cyber_s86928_c10544.jpg | train | people100 | crowd | 0 |
| cyber_s86928_c122889.jpg | train | base300 | — | — |
| cyber_s86928_c309380.jpg | train | people100 | single_main | 1 |
| cyber_s86928_c97289.jpg | train | base300 | — | — |
| cyber_s86972_c180232.jpg | train | base300 | — | — |
| cyber_s86972_c259675.jpg | train | base300 | — | — |
| cyber_s86982_c120191.jpg | train | base300 | — | — |
| cyber_s86982_c204153.jpg | train | base300 | — | — |
| cyber_s86982_c77616.jpg | train | base300 | — | — |
| cyber_s87023_c263536.jpg | train | people100 | single_main | 1 |
| cyber_s87023_c367710.jpg | train | base300 | — | — |
| cyber_s87023_c49865.jpg | train | base300 | — | — |
| cyber_s87103_c326884.jpg | train | base300 | — | — |
| cyber_s87103_c350265.jpg | train | base300 | — | — |
| cyber_s87103_c65666.jpg | train | base300 | — | — |
| cyber_s87136_c120193.jpg | train | base300 | — | — |
| cyber_s87136_c212453.jpg | train | base300 | — | — |
| cyber_s87136_c268267.jpg | train | base300 | — | — |
| cyber_s87136_c378032.jpg | validation | base300 | — | — |
| cyber_s87247_c276382.jpg | train | base300 | — | — |
| cyber_s87247_c341966.jpg | train | base300 | — | — |
| cyber_s87247_c73127.jpg | train | base300 | — | — |
| cyber_s87292_c122494.jpg | validation | base300 | — | — |
| cyber_s87292_c341782.jpg | validation | people100 | crowd | 0 |
| cyber_s87300_c147949.jpg | validation | base300 | — | — |
| cyber_s87300_c166993.jpg | train | base300 | — | — |
| cyber_s87300_c170319.jpg | train | base300 | — | — |
| cyber_s87300_c314019.jpg | train | base300 | — | — |
| cyber_s87300_c347979.jpg | validation | base300 | — | — |
| cyber_s87325_c144781.jpg | train | base300 | — | — |
| cyber_s87325_c177201.jpg | validation | base300 | — | — |
| cyber_s87325_c331875.jpg | train | base300 | — | — |
| cyber_s87389_c234084.jpg | train | base300 | — | — |
| cyber_s87450_c140239.jpg | train | base300 | — | — |
| cyber_s87450_c216922.jpg | train | people100 | small_group | 0 |
| cyber_s87519_c192155.jpg | train | base300 | — | — |
| cyber_s87519_c293253.jpg | train | base300 | — | — |
| cyber_s87676_c117779.jpg | train | people100 | single_main | 0 |
| cyber_s87676_c125842.jpg | train | base300 | — | — |
| cyber_s87676_c352857.jpg | train | people100 | single_main | 0 |
| cyber_s87683_c150522.jpg | train | base300 | — | — |
| cyber_s87683_c282597.jpg | train | base300 | — | — |
| cyber_s87695_c149023.jpg | train | people100 | single_main | 0 |
| cyber_s87695_c162662.jpg | train | base300 | — | — |
| cyber_s87695_c188938.jpg | train | people100 | single_main | 0 |
| cyber_s87695_c198618.jpg | train | people100 | single_main | 0 |
| cyber_s87695_c268070.jpg | train | base300 | — | — |
| cyber_s87695_c290027.jpg | train | base300 | — | — |
| cyber_s87695_c70464.jpg | train | people100 | small_group | 0 |
| cyber_s87797_c144174.jpg | train | base300 | — | — |
| cyber_s87797_c291089.jpg | train | base300 | — | — |
| cyber_s87797_c41120.jpg | train | base300 | — | — |
| cyber_s87971_c185049.jpg | train | people100 | single_main | 0 |
| cyber_s87971_c29370.jpg | train | people100 | single_main | 1 |
| cyber_s87971_c78343.jpg | validation | base300 | — | — |
| cyber_s87979_c137517.jpg | train | base300 | — | — |
| cyber_s87979_c289651.jpg | validation | base300 | — | — |
| cyber_s87979_c32680.jpg | train | people100 | single_main | 1 |
| cyber_s87979_c328979.jpg | train | base300 | — | — |
| cyber_s87979_c351701.jpg | train | people100 | single_main | 1 |
| cyber_s87986_c298205.jpg | train | base300 | — | — |
| cyber_s88079_c88454.jpg | train | base300 | — | — |
| cyber_s88152_c193344.jpg | train | people100 | single_main | 1 |
| cyber_s88152_c209359.jpg | train | people100 | single_main | 1 |
| cyber_s88152_c244199.jpg | train | people100 | single_main | 1 |
| cyber_s88152_c362950.jpg | train | people100 | small_group | 0 |
| cyber_s88152_c59075.jpg | train | people100 | small_group | 0 |
| cyber_s88152_c80026.jpg | train | people100 | single_main | 0 |
| cyber_s88158_c25841.jpg | train | base300 | — | — |
| cyber_s88163_c339549.jpg | train | base300 | — | — |
| cyber_s88163_c61078.jpg | train | people100 | small_group | 0 |
| cyber_s88192_c295046.jpg | train | base300 | — | — |
| cyber_s88192_c379528.jpg | train | base300 | — | — |
| cyber_s88251_c147433.jpg | train | people100 | small_group | 0 |
| cyber_s88251_c191639.jpg | train | people100 | small_group | 0 |
| cyber_s88251_c347951.jpg | train | base300 | — | — |
| cyber_s88251_c54151.jpg | train | base300 | — | — |
| cyber_s88251_c85523.jpg | train | base300 | — | — |
| cyber_s88286_c145866.jpg | train | base300 | — | — |
| cyber_s88299_c10567.jpg | train | people100 | small_group | 0 |
| cyber_s88299_c78582.jpg | train | base300 | — | — |
| cyber_s88369_c10713.jpg | train | base300 | — | — |
| cyber_s88369_c157420.jpg | train | base300 | — | — |
| cyber_s88369_c384007.jpg | train | base300 | — | — |
| cyber_s88369_c384243.jpg | train | base300 | — | — |
| cyber_s88402_c112425.jpg | validation | people100 | small_group | 0 |
| cyber_s88402_c171965.jpg | validation | base300 | — | — |
| cyber_s88402_c249554.jpg | train | base300 | — | — |
| cyber_s88402_c329929.jpg | train | base300 | — | — |
| cyber_s88500_c143053.jpg | train | base300 | — | — |
| cyber_s88500_c66290.jpg | train | base300 | — | — |
| cyber_s88515_c120136.jpg | train | base300 | — | — |
| cyber_s88535_c396633.jpg | train | base300 | — | — |
| cyber_s88535_c76402.jpg | train | base300 | — | — |
| cyber_s88557_c201500.jpg | train | people100 | small_group | 0 |
| cyber_s88557_c295579.jpg | validation | people100 | single_main | 1 |
| cyber_s88557_c297464.jpg | train | people100 | small_group | 0 |
| cyber_s88557_c329319.jpg | train | people100 | single_main | 1 |
| cyber_s88557_c363613.jpg | train | people100 | small_group | 0 |
| cyber_s88557_c396284.jpg | train | base300 | — | — |
| cyber_s88581_c340331.jpg | train | base300 | — | — |
| cyber_s88581_c346833.jpg | train | base300 | — | — |
| cyber_s88581_c41668.jpg | train | base300 | — | — |
| cyber_s88581_c64089.jpg | train | people100 | single_main | 1 |
| cyber_s88611_c206017.jpg | train | base300 | — | — |
| cyber_s88611_c248364.jpg | train | base300 | — | — |
| cyber_s88611_c364268.jpg | train | people100 | small_group | 0 |
| cyber_s88615_c252442.jpg | validation | base300 | — | — |
| cyber_s88615_c253219.jpg | train | base300 | — | — |
| cyber_s88615_c84846.jpg | train | base300 | — | — |
| cyber_s88635_c343707.jpg | train | base300 | — | — |
| cyber_s88635_c35010.jpg | train | base300 | — | — |
| cyber_s88635_c42526.jpg | train | base300 | — | — |
| cyber_s88635_c56763.jpg | train | base300 | — | — |
| cyber_s88704_c195628.jpg | validation | base300 | — | — |
| cyber_s88704_c336256.jpg | train | base300 | — | — |
| cyber_s88713_c238087.jpg | train | base300 | — | — |
| cyber_s88713_c284724.jpg | train | base300 | — | — |
| cyber_s88713_c321622.jpg | train | people100 | single_main | 0 |
| cyber_s88713_c323462.jpg | train | people100 | single_main | 0 |
| cyber_s88713_c350420.jpg | train | base300 | — | — |
| cyber_s88717_c299578.jpg | train | base300 | — | — |
| cyber_s88717_c44120.jpg | train | base300 | — | — |
| cyber_s88748_c162419.jpg | train | base300 | — | — |
| cyber_s88748_c205221.jpg | train | base300 | — | — |
| cyber_s88748_c332574.jpg | train | base300 | — | — |
| cyber_s88792_c270994.jpg | train | people100 | small_group | 0 |
| cyber_s88792_c303493.jpg | train | people100 | small_group | 0 |
| cyber_s88792_c382689.jpg | train | base300 | — | — |
| cyber_s88817_c254857.jpg | train | base300 | — | — |
| cyber_s88817_c51348.jpg | train | base300 | — | — |
| cyber_s88915_c254483.jpg | train | base300 | — | — |
| cyber_s88915_c32880.jpg | train | base300 | — | — |
| cyber_s89013_c337678.jpg | train | people100 | crowd | 0 |
| cyber_s89013_c348352.jpg | train | base300 | — | — |
| cyber_s89060_c295080.jpg | train | base300 | — | — |
| cyber_s89060_c351028.jpg | train | base300 | — | — |
| cyber_s89060_c45371.jpg | train | base300 | — | — |
| cyber_s89061_c136656.jpg | train | base300 | — | — |
| cyber_s89061_c160987.jpg | train | base300 | — | — |
| cyber_s89061_c256014.jpg | train | base300 | — | — |
| cyber_s89061_c395881.jpg | train | base300 | — | — |
| cyber_s89115_c282538.jpg | train | base300 | — | — |
| cyber_s89115_c350836.jpg | train | base300 | — | — |
| cyber_s89212_c120732.jpg | train | base300 | — | — |
| cyber_s89212_c172264.jpg | train | base300 | — | — |
| cyber_s89212_c281969.jpg | train | base300 | — | — |
| cyber_s89212_c328977.jpg | train | base300 | — | — |
| cyber_s89235_c24280.jpg | train | base300 | — | — |
| cyber_s89235_c278508.jpg | train | base300 | — | — |
| cyber_s89237_c114128.jpg | train | people100 | single_main | 0 |
| cyber_s89237_c184801.jpg | train | people100 | small_group | 0 |
| cyber_s89237_c187794.jpg | train | people100 | single_main | 0 |
| cyber_s89237_c250690.jpg | train | base300 | — | — |
| cyber_s89237_c33967.jpg | train | people100 | single_main | 0 |
| cyber_s89237_c55964.jpg | train | base300 | — | — |
| cyber_s89242_c294239.jpg | train | base300 | — | — |
| cyber_s89242_c322457.jpg | train | people100 | crowd | 0 |
| cyber_s89242_c75003.jpg | train | base300 | — | — |
| cyber_s89253_c332151.jpg | validation | base300 | — | — |
| cyber_s89295_c343018.jpg | train | base300 | — | — |
| cyber_s89396_c196705.jpg | train | base300 | — | — |
| cyber_s89396_c384230.jpg | validation | base300 | — | — |
| cyber_s89396_c96374.jpg | train | base300 | — | — |
| cyber_s89414_c123663.jpg | train | base300 | — | — |
| cyber_s89414_c312635.jpg | train | base300 | — | — |
| cyber_s89414_c41294.jpg | train | base300 | — | — |
| cyber_s89550_c74454.jpg | train | base300 | — | — |
| cyber_s89550_c86929.jpg | train | base300 | — | — |
| cyber_s89555_c340688.jpg | train | base300 | — | — |
| cyber_s89555_c382176.jpg | train | base300 | — | — |
| cyber_s89724_c147619.jpg | train | base300 | — | — |
| cyber_s89724_c280181.jpg | train | base300 | — | — |
| cyber_s89788_c270331.jpg | train | base300 | — | — |
| cyber_s89788_c292562.jpg | train | base300 | — | — |
| cyber_s89824_c139486.jpg | validation | base300 | — | — |
| cyber_s89824_c235979.jpg | train | base300 | — | — |
| cyber_s89856_c160965.jpg | train | base300 | — | — |
| cyber_s89935_c146400.jpg | train | base300 | — | — |
| cyber_s89935_c271854.jpg | train | people100 | small_group | 0 |
| cyber_s89935_c62997.jpg | train | people100 | small_group | 0 |
| cyber_s89981_c328141.jpg | train | base300 | — | — |
| cyber_s89981_c331958.jpg | train | base300 | — | — |
| cyber_s89997_c279509.jpg | train | base300 | — | — |
| cyber_s89997_c384623.jpg | train | base300 | — | — |
| cyber_s89998_c17126.jpg | train | people100 | single_main | 1 |
| cyber_s89998_c255725.jpg | train | base300 | — | — |
| cyber_s89998_c374834.jpg | train | base300 | — | — |
| cyber_s89998_c62849.jpg | train | people100 | single_main | 1 |
| cyber_s89998_c70773.jpg | validation | base300 | — | — |
| cyber_s90031_c249437.jpg | train | base300 | — | — |
| cyber_s90031_c276022.jpg | train | base300 | — | — |
| cyber_s90031_c295089.jpg | train | base300 | — | — |
| cyber_s90031_c305564.jpg | train | base300 | — | — |
| cyber_s90031_c307354.jpg | train | people100 | crowd | 0 |
| cyber_s90031_c67234.jpg | train | base300 | — | — |
| cyber_s90036_c122005.jpg | train | people100 | small_group | 0 |
| cyber_s90036_c138427.jpg | validation | base300 | — | — |
| cyber_s90036_c151391.jpg | train | people100 | crowd | 0 |
| cyber_s90036_c32261.jpg | train | base300 | — | — |
| cyber_s90036_c367349.jpg | train | people100 | small_group | 0 |
| cyber_s90072_c390109.jpg | train | base300 | — | — |
| cyber_s90086_c277276.jpg | train | base300 | — | — |
| cyber_s90086_c290886.jpg | train | base300 | — | — |
| cyber_s90098_c161542.jpg | validation | base300 | — | — |
| cyber_s90098_c391123.jpg | validation | base300 | — | — |
| cyber_s90142_c260471.jpg | train | people100 | crowd | 0 |
| cyber_s90142_c77095.jpg | train | base300 | — | — |
| cyber_s90213_c245159.jpg | train | base300 | — | — |
| cyber_s90213_c327744.jpg | train | base300 | — | — |
| cyber_s90213_c85991.jpg | train | base300 | — | — |
| cyber_s90234_c399727.jpg | train | base300 | — | — |
| cyber_s90378_c48742.jpg | train | base300 | — | — |
| cyber_s90431_c348554.jpg | train | base300 | — | — |
| cyber_s90431_c54127.jpg | validation | base300 | — | — |
| cyber_s90460_c159950.jpg | train | base300 | — | — |
| cyber_s90460_c189016.jpg | train | people100 | single_main | 0 |
| cyber_s90460_c29557.jpg | train | base300 | — | — |
| cyber_s90510_c145235.jpg | train | base300 | — | — |
| cyber_s90541_c350136.jpg | train | base300 | — | — |
| cyber_s90541_c391275.jpg | train | base300 | — | — |
| cyber_s90541_c42395.jpg | train | base300 | — | — |
| cyber_s90606_c116828.jpg | train | base300 | — | — |
| cyber_s90606_c325261.jpg | validation | base300 | — | — |
| cyber_s90606_c337762.jpg | train | people100 | small_group | 0 |
| cyber_s90616_c234920.jpg | train | base300 | — | — |
| cyber_s90616_c23591.jpg | train | people100 | single_main | 1 |
| cyber_s90616_c308483.jpg | train | people100 | single_main | 1 |
| cyber_s90616_c389598.jpg | train | base300 | — | — |
| cyber_s90616_c40759.jpg | train | base300 | — | — |
| cyber_s90616_c97687.jpg | train | people100 | small_group | 0 |
| cyber_s90632_c334348.jpg | train | base300 | — | — |
| cyber_s90632_c379977.jpg | train | base300 | — | — |
| cyber_s90655_c119922.jpg | train | base300 | — | — |
| cyber_s90655_c187420.jpg | train | people100 | single_main | 0 |
| cyber_s90655_c252768.jpg | train | base300 | — | — |
| cyber_s90655_c357651.jpg | train | people100 | single_main | 0 |
| cyber_s90655_c40078.jpg | train | base300 | — | — |
| cyber_s90686_c119561.jpg | train | base300 | — | — |
| cyber_s90686_c173383.jpg | train | base300 | — | — |
| cyber_s90686_c327254.jpg | train | base300 | — | — |
| cyber_s90686_c330507.jpg | train | base300 | — | — |
| cyber_s90687_c154083.jpg | train | base300 | — | — |
| cyber_s90687_c55864.jpg | train | base300 | — | — |
| cyber_s90714_c65994.jpg | train | base300 | — | — |
| cyber_s90823_c83646.jpg | train | base300 | — | — |
| cyber_s90898_c103283.jpg | train | base300 | — | — |
| cyber_s90898_c118141.jpg | train | base300 | — | — |
| cyber_s90898_c156004.jpg | validation | base300 | — | — |
| cyber_s90898_c205391.jpg | train | base300 | — | — |
