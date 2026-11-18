# Dataset per il LoRA Fumetto con retino

Aggiornato al **18 novembre 2026**. Contiene **400 immagini JPEG con caption**,
suddivise in **360 train** e **40 validation**, per LoRA di Stable Diffusion 1.5.
La selezione riprende i criteri del [dataset GTA5](../gta5/Dataset_LoRA.md).
**Nessun training è stato eseguito durante questa preparazione.**

## Fonte e stile

Fonte: [adrianmele/MegaStyle-1.4M](https://huggingface.co/datasets/adrianmele/MegaStyle-1.4M),
revisione fissata
[`141be95c7ba9724caeca38a973e9bdcd4c78bdac`](https://huggingface.co/datasets/adrianmele/MegaStyle-1.4M/tree/141be95c7ba9724caeca38a973e9bdcd4c78bdac).
Gli autori descrivono immagini sintetiche ottenute con Qwen-Image; la scheda
originale è conservata in [SOURCE_README.md](SOURCE_README.md).

Dall'indice ufficiale `style_indices.pkl` sono state individuate descrizioni
contenenti `halftone` o `ben-day`, riferite a comic o pop art. Per la raccolta
sono state privilegiate quelle esplicitamente comic, evitando etichette noir,
monocromatiche, black and white, manga e anime. Sono stati scaricati **960
candidati da 120 descrizioni di stile**, quindi rivisti tramite tavole di contatto.
Le immagini selezionate sono esempi già stilizzati: non sono fotografie raw.

La resa ricercata combina contorni neri a inchiostro, campiture piatte,
retino puntinato e ombre nette. Sono presenti palette ciano/rosso/giallo,
verde/rosso e ocra/crema, retini con diversa densità e alcune scene quasi
monocromatiche nonostante l'etichetta sorgente. Alcune immagini contengono
raggi grafici o fondali teatrali: il dataset **non garantisce un'unica palette,
una frequenza uniforme dei puntini o la replica esatta dell'esempio generato
in chat**. Un'etichetta sorgente non certifica da sola la resa di ogni immagine.
La galleria consente di valutare direttamente questa variabilità prima del training.

**Licenza:** [License Term of MegaStyle](SOURCE_LICENSE.txt), dichiarata `other`
nella fonte. Consente ricerca e didattica, esclude uso commerciale e di produzione
e richiede di mantenere copyright e avviso di permesso. Il testo è incluso
integralmente. Non è una licenza Apache/MIT e non certifica i diritti di ogni
singola immagine. Rispetto a GTA5 cambia la fonte e cambia il regime di licenza.

Altre fonti esaminate non sono state importate: Comic Strips di bghira raccoglie
strisce eterogenee con licenze delle immagini non uniformi; Halftone Revolution
riguarda immagini storiche di giornale, non specificamente fumetti con retino.

## Composizione e criteri

| Sottoinsieme | Train | Validation | Totale |
|---|---:|---:|---:|
| `base300`, contenuti vari | 270 | 30 | 300 |
| `people100`, scene aggiuntive con persone | 90 | 10 | 100 |
| **Totale** | **360** | **40** | **400** |

| Composizione delle 100 scene aggiuntive | Train | Validation | Totale |
|---|---:|---:|---:|
| Folla, `crowd` | 9 | 1 | 10 |
| Piccolo gruppo, `small_group` | 38 | 4 | 42 |
| Persona principale ambientata, `single_main` | 43 | 5 | 48 |

**24 delle 100** hanno `wide=1`: figure lontane o inquadratura ampia con
ambiente leggibile. È una proprietà trasversale, non un quarto gruppo.
Le etichette sono qualitative; una scena con persona principale può contenere
figure secondarie. Le persone già presenti nelle 300 non contribuiscono alle 100.

Contenuti: paesaggi montani, coste e corsi d'acqua, campagna, città e monumenti,
interni e infrastrutture, persone in attività, sport, lavoro, piccoli gruppi,
pubblici e mercati, veicoli, animali, piante, cibo e oggetti.
Sono stati evitati ritratti isolati dominanti, collage, grafici, lettere isolate,
scritte predominanti e molte varianti ridondanti. Restano alcuni testi, marchi
sintetici, soggetti fantastici e errori anatomici o geometrici della generazione.

**Replicato da GTA5:** 400 esempi, split 90/10, struttura 300+100, quote delle
persone, 24 scene ampie, caption di contenuto, controllo visivo, tracciabilità
e controlli contro duplicati e contaminazione degli split.
Le 300 immagini vengono congelate con questa prima selezione.

**Adattamenti necessari:** le illustrazioni costituiscono qui il target, mentre
GTA5 escludeva illustrazioni promozionali per privilegiare screenshot in-game.
Non si riproducono i soggetti immagine per immagine né le proporzioni delle
gallerie GTA5. Non è un confronto semanticamente bilanciato fra stili.
Non servono coppie fotografia/target per il trainer.

## Immagini, caption e crop

Tutti i JPEG sono **512×512**, scaricati dal viewer Hugging Face della revisione
indicata e conservati senza ricompressione locale. Il viewer può ricodificare
le immagini: non si dichiara l'identità con i byte dei Parquet originali.
SHA256, ID sorgente, indice di riga e URL degli asset sono nel manifest.
Le URL sono archiviate senza firme temporanee.

Le caption inglesi iniziano con `Halftone comic illustration.`. Partono dalla
colonna `content`, con correzioni rispetto alle anteprime visive quando sono
state rilevate discrepanze. Le descrizioni originali e la colonna tecnica
`style` restano nel manifest; **soltanto `text` entra nel training**.
La revisione tramite tavole di contatto non certifica ogni dettaglio delle caption.

Il loader `PixelArtDataset` accetta gli split senza modifiche. I config usano
**Lanczos**, come GTA5, con EXIF/RGB, resize proporzionale e crop centrale a
512×512, normalizzazione in [-1,1], senza flip o crop casuali. Poiché le immagini
sono già quadrate a questa risoluzione, **immagine e crop coincidono**.
La validation non aggiorna i pesi; le fotografie di inferenza sono separate.

## Struttura e utilizzo

```text
data/halftone_comic/
├── Dataset_LoRA.md
├── SOURCE_LICENSE.txt
├── SOURCE_README.md
├── selection.csv
├── validation_report.json
├── review.html
├── train/                   # 360 JPEG + metadata.jsonl
└── validation/              # 40 JPEG + metadata.jsonl
```

I metadata contengono `file_name` relativo allo split e `text`. Il Markdown
non viene letto dal training. [selection.csv](selection.csv) congela file,
caption, split, quote, ID di contenuto, fonte e hash. [review.html](review.html)
mostra tutte le immagini e le annotazioni nel browser locale.

Verifica senza training:

```bash
python3 scripts/build_halftone_dataset.py
python3 main.py --lora lora_halftone_rank4.toml train --check-data
```

Ripristino delle sole immagini o metadata mancanti, verificando gli hash:

```bash
python3 scripts/build_halftone_dataset.py --download
```

Il ripristino usa la cache locale oppure aggiorna la URL firmata dal viewer;
controlla revisione e ID e fallisce se i byte cambiano. Disponibilità e codifica
degli asset del viewer non sono garantite nel tempo: mantenere una copia locale.

I config `lora_halftone_rank4.toml`, `lora_halftone_rank8.toml`,
`lora_halftone_rank16.toml`, `lora_halftone_rank32.toml` usano questo dataset,
lo stile `halftone_comic` e output separati in `lora/halftone_comic_rank_*`.
Un training si avvia successivamente con:

```bash
python3 main.py --lora lora_halftone_rank4.toml train
```

## Integrità e riproducibilità

Revisione: **`halftone-context400-2026-11-18`**. Circa **20,5 MB** di immagini.

Controlli automatici completati:

- conteggi 360/40, 300+100, quote delle persone e 24 scene ampie;
- 400 diversi ID di contenuto `cNNNN`, senza varianti del medesimo ID fra split;
- decodifica, dimensioni e SHA256 dei byte e dei pixel contro il manifest;
- nessun duplicato esatto dei byte o dei pixel RGB; il crop coincide con l'immagine;
- nessuna coppia con dHash ≤ 4 nell'intero dataset, quindi anche fra split;
- nessuna corrispondenza RGB esatta o dHash ≤ 4 con le 108 foto locali di inferenza;
- audit attraverso `main.py ... train --check-data` senza avviare il training;
- preprocessing effettivo dei 400 esempi: tensori 3×512×512 finiti in [-1,1];
- tokenizzazione CLIP locale: caption di **13–25 token**, entro il limite 77,
  senza troncamento; esito conservato in `loader_validation.json`.

Gli ID distinti e dHash non escludono ogni duplicato semantico, personaggio
ricorrente, scena simile o contaminazione con benchmark esterni.
Usare gli stessi file, caption e split per confrontare i rank 4/8/16/32.
I controlli di integrità non dimostrano la qualità del futuro LoRA.

Firme del loader, comprendenti nomi, caption e byte:

- train: `37284cbf132a02f7a50f57ecb56abf840c7750792e5857e03004ce7034fc08ed`;
- validation: `0e18b8acef7c99b4f6a733c15fd94a8f6ca67af08fc82a66e01e1436349cbebd`;

## Inventario completo

Fonte e descrizioni originali sono nel CSV. `wide` è annotato solo per le 100
scene con persone; un trattino nelle 300 non significa assenza di viste ampie.

| File | Split | Sottoinsieme | Composizione | Wide |
|---|---|---|---|---:|
| halftone_s166074_c172942.jpg | train | base300 | — | — |
| halftone_s166090_c112120.jpg | train | base300 | — | — |
| halftone_s166090_c266548.jpg | train | base300 | — | — |
| halftone_s166090_c271823.jpg | train | people100 | crowd | 0 |
| halftone_s166090_c286074.jpg | train | people100 | small_group | 0 |
| halftone_s166091_c227898.jpg | train | base300 | — | — |
| halftone_s166091_c273238.jpg | train | base300 | — | — |
| halftone_s166091_c79287.jpg | train | people100 | small_group | 0 |
| halftone_s166097_c343907.jpg | train | base300 | — | — |
| halftone_s166097_c373552.jpg | train | people100 | crowd | 0 |
| halftone_s166097_c38287.jpg | train | people100 | small_group | 0 |
| halftone_s166097_c70185.jpg | train | people100 | single_main | 1 |
| halftone_s166098_c104173.jpg | validation | base300 | — | — |
| halftone_s166098_c1470.jpg | train | base300 | — | — |
| halftone_s166098_c259578.jpg | train | base300 | — | — |
| halftone_s166118_c63467.jpg | train | people100 | small_group | 0 |
| halftone_s166118_c78389.jpg | train | base300 | — | — |
| halftone_s166136_c154023.jpg | train | base300 | — | — |
| halftone_s166136_c214995.jpg | validation | base300 | — | — |
| halftone_s166136_c295527.jpg | train | people100 | single_main | 1 |
| halftone_s166136_c307181.jpg | validation | base300 | — | — |
| halftone_s166136_c340177.jpg | train | base300 | — | — |
| halftone_s166136_c384772.jpg | train | people100 | crowd | 0 |
| halftone_s166166_c154217.jpg | validation | base300 | — | — |
| halftone_s166166_c399948.jpg | train | base300 | — | — |
| halftone_s166171_c112141.jpg | train | base300 | — | — |
| halftone_s166171_c122661.jpg | train | base300 | — | — |
| halftone_s166171_c184739.jpg | train | people100 | small_group | 0 |
| halftone_s166171_c242405.jpg | train | people100 | small_group | 0 |
| halftone_s166172_c361954.jpg | train | base300 | — | — |
| halftone_s166172_c393769.jpg | train | base300 | — | — |
| halftone_s166176_c236519.jpg | train | base300 | — | — |
| halftone_s166176_c42044.jpg | train | base300 | — | — |
| halftone_s166176_c89737.jpg | train | base300 | — | — |
| halftone_s166177_c255402.jpg | train | base300 | — | — |
| halftone_s166177_c320482.jpg | train | base300 | — | — |
| halftone_s166177_c40806.jpg | train | base300 | — | — |
| halftone_s166195_c198031.jpg | train | base300 | — | — |
| halftone_s166195_c199852.jpg | train | base300 | — | — |
| halftone_s166195_c313002.jpg | train | base300 | — | — |
| halftone_s166195_c358930.jpg | train | base300 | — | — |
| halftone_s166195_c363538.jpg | train | base300 | — | — |
| halftone_s166219_c114483.jpg | train | base300 | — | — |
| halftone_s166219_c25381.jpg | validation | base300 | — | — |
| halftone_s166219_c39752.jpg | train | base300 | — | — |
| halftone_s166219_c40912.jpg | train | base300 | — | — |
| halftone_s166220_c143252.jpg | train | base300 | — | — |
| halftone_s166220_c73387.jpg | train | base300 | — | — |
| halftone_s166222_c115769.jpg | train | base300 | — | — |
| halftone_s166222_c210175.jpg | train | people100 | single_main | 0 |
| halftone_s166222_c248795.jpg | train | base300 | — | — |
| halftone_s166222_c270131.jpg | train | base300 | — | — |
| halftone_s166222_c67820.jpg | train | base300 | — | — |
| halftone_s166222_c75808.jpg | train | base300 | — | — |
| halftone_s166252_c251322.jpg | validation | base300 | — | — |
| halftone_s166252_c254228.jpg | train | base300 | — | — |
| halftone_s166252_c272307.jpg | train | base300 | — | — |
| halftone_s166252_c38818.jpg | train | base300 | — | — |
| halftone_s166283_c21096.jpg | train | base300 | — | — |
| halftone_s166283_c240581.jpg | train | base300 | — | — |
| halftone_s166283_c338427.jpg | train | base300 | — | — |
| halftone_s166283_c68720.jpg | train | base300 | — | — |
| halftone_s166286_c280863.jpg | train | base300 | — | — |
| halftone_s166312_c155144.jpg | train | base300 | — | — |
| halftone_s166312_c242821.jpg | train | base300 | — | — |
| halftone_s166318_c167001.jpg | train | base300 | — | — |
| halftone_s166318_c97538.jpg | train | base300 | — | — |
| halftone_s166323_c198483.jpg | validation | base300 | — | — |
| halftone_s166323_c209180.jpg | train | base300 | — | — |
| halftone_s166323_c285047.jpg | validation | base300 | — | — |
| halftone_s166324_c224392.jpg | train | base300 | — | — |
| halftone_s166324_c354307.jpg | train | base300 | — | — |
| halftone_s166324_c367753.jpg | train | base300 | — | — |
| halftone_s166324_c7871.jpg | train | base300 | — | — |
| halftone_s166326_c157962.jpg | train | base300 | — | — |
| halftone_s166326_c203353.jpg | train | base300 | — | — |
| halftone_s166326_c340245.jpg | train | base300 | — | — |
| halftone_s166340_c140553.jpg | validation | base300 | — | — |
| halftone_s166340_c200529.jpg | train | base300 | — | — |
| halftone_s166340_c34208.jpg | train | base300 | — | — |
| halftone_s166340_c395592.jpg | train | base300 | — | — |
| halftone_s166342_c110684.jpg | validation | base300 | — | — |
| halftone_s166342_c26740.jpg | train | base300 | — | — |
| halftone_s166342_c31997.jpg | train | base300 | — | — |
| halftone_s166343_c14340.jpg | validation | people100 | small_group | 1 |
| halftone_s166343_c154465.jpg | validation | base300 | — | — |
| halftone_s166343_c158031.jpg | train | people100 | single_main | 0 |
| halftone_s166343_c168843.jpg | validation | people100 | single_main | 0 |
| halftone_s166343_c240391.jpg | train | people100 | single_main | 1 |
| halftone_s166343_c64788.jpg | train | people100 | single_main | 1 |
| halftone_s166343_c72587.jpg | train | base300 | — | — |
| halftone_s166344_c295926.jpg | validation | base300 | — | — |
| halftone_s166344_c308485.jpg | train | base300 | — | — |
| halftone_s166344_c391667.jpg | train | base300 | — | — |
| halftone_s166344_c396654.jpg | train | base300 | — | — |
| halftone_s166346_c127795.jpg | train | people100 | small_group | 0 |
| halftone_s166346_c203614.jpg | train | base300 | — | — |
| halftone_s166346_c242435.jpg | train | people100 | single_main | 1 |
| halftone_s166346_c44984.jpg | train | base300 | — | — |
| halftone_s166347_c304325.jpg | train | base300 | — | — |
| halftone_s166347_c359110.jpg | train | base300 | — | — |
| halftone_s166347_c60640.jpg | train | base300 | — | — |
| halftone_s166347_c99639.jpg | train | base300 | — | — |
| halftone_s166348_c281151.jpg | train | base300 | — | — |
| halftone_s166348_c347279.jpg | train | base300 | — | — |
| halftone_s166350_c103952.jpg | train | people100 | single_main | 1 |
| halftone_s166350_c140624.jpg | train | base300 | — | — |
| halftone_s166350_c35875.jpg | train | base300 | — | — |
| halftone_s166350_c389665.jpg | train | base300 | — | — |
| halftone_s166350_c70524.jpg | train | people100 | small_group | 0 |
| halftone_s166350_c93370.jpg | train | people100 | single_main | 0 |
| halftone_s166351_c76064.jpg | train | base300 | — | — |
| halftone_s166353_c277692.jpg | train | base300 | — | — |
| halftone_s166353_c297725.jpg | train | base300 | — | — |
| halftone_s166353_c392960.jpg | train | base300 | — | — |
| halftone_s166353_c67242.jpg | train | base300 | — | — |
| halftone_s166357_c393646.jpg | train | base300 | — | — |
| halftone_s166361_c268309.jpg | train | base300 | — | — |
| halftone_s166361_c338009.jpg | train | people100 | crowd | 0 |
| halftone_s166361_c365569.jpg | train | base300 | — | — |
| halftone_s166363_c156473.jpg | train | base300 | — | — |
| halftone_s166363_c284279.jpg | train | people100 | single_main | 1 |
| halftone_s166363_c373848.jpg | train | base300 | — | — |
| halftone_s166374_c112052.jpg | train | base300 | — | — |
| halftone_s166374_c279989.jpg | train | base300 | — | — |
| halftone_s166374_c358763.jpg | train | people100 | small_group | 0 |
| halftone_s166374_c65333.jpg | train | base300 | — | — |
| halftone_s166387_c211153.jpg | train | base300 | — | — |
| halftone_s166387_c22348.jpg | validation | people100 | single_main | 1 |
| halftone_s166387_c246590.jpg | train | people100 | small_group | 0 |
| halftone_s166395_c163384.jpg | train | people100 | single_main | 1 |
| halftone_s166395_c31250.jpg | validation | people100 | single_main | 0 |
| halftone_s166395_c99960.jpg | validation | people100 | small_group | 0 |
| halftone_s166401_c120154.jpg | validation | base300 | — | — |
| halftone_s166401_c128384.jpg | train | base300 | — | — |
| halftone_s166401_c234670.jpg | train | base300 | — | — |
| halftone_s166447_c106365.jpg | train | people100 | single_main | 0 |
| halftone_s166447_c248011.jpg | train | base300 | — | — |
| halftone_s166447_c280955.jpg | train | people100 | small_group | 0 |
| halftone_s166447_c337256.jpg | train | people100 | crowd | 0 |
| halftone_s166482_c373487.jpg | validation | base300 | — | — |
| halftone_s166483_c117473.jpg | train | base300 | — | — |
| halftone_s166483_c182655.jpg | train | base300 | — | — |
| halftone_s166483_c64288.jpg | train | people100 | small_group | 0 |
| halftone_s166525_c206588.jpg | train | base300 | — | — |
| halftone_s166525_c22239.jpg | train | people100 | small_group | 0 |
| halftone_s166525_c25346.jpg | train | base300 | — | — |
| halftone_s166525_c264870.jpg | train | base300 | — | — |
| halftone_s166525_c271061.jpg | train | people100 | small_group | 0 |
| halftone_s166586_c330165.jpg | train | base300 | — | — |
| halftone_s166586_c354181.jpg | train | base300 | — | — |
| halftone_s166586_c393390.jpg | train | base300 | — | — |
| halftone_s166636_c150955.jpg | train | base300 | — | — |
| halftone_s166636_c272370.jpg | train | base300 | — | — |
| halftone_s166636_c374193.jpg | train | base300 | — | — |
| halftone_s166664_c120124.jpg | train | base300 | — | — |
| halftone_s166664_c232305.jpg | train | base300 | — | — |
| halftone_s166664_c291334.jpg | train | base300 | — | — |
| halftone_s166664_c306307.jpg | train | people100 | crowd | 0 |
| halftone_s166664_c342542.jpg | train | base300 | — | — |
| halftone_s166664_c3798.jpg | train | base300 | — | — |
| halftone_s166668_c140290.jpg | train | base300 | — | — |
| halftone_s166668_c184719.jpg | validation | people100 | small_group | 0 |
| halftone_s166668_c80957.jpg | train | people100 | small_group | 0 |
| halftone_s166668_c8697.jpg | train | people100 | small_group | 0 |
| halftone_s166671_c34481.jpg | train | base300 | — | — |
| halftone_s166677_c12524.jpg | train | people100 | small_group | 0 |
| halftone_s166677_c52512.jpg | train | base300 | — | — |
| halftone_s166677_c59643.jpg | train | people100 | small_group | 0 |
| halftone_s166680_c174008.jpg | train | base300 | — | — |
| halftone_s166680_c380077.jpg | train | base300 | — | — |
| halftone_s166681_c171132.jpg | train | base300 | — | — |
| halftone_s166681_c268459.jpg | train | base300 | — | — |
| halftone_s166681_c286523.jpg | train | base300 | — | — |
| halftone_s166707_c24785.jpg | train | base300 | — | — |
| halftone_s166716_c105891.jpg | train | people100 | small_group | 0 |
| halftone_s166716_c266847.jpg | train | base300 | — | — |
| halftone_s166716_c314378.jpg | train | base300 | — | — |
| halftone_s166716_c351977.jpg | train | base300 | — | — |
| halftone_s166716_c374833.jpg | train | people100 | small_group | 0 |
| halftone_s166716_c42386.jpg | train | base300 | — | — |
| halftone_s166723_c308852.jpg | train | base300 | — | — |
| halftone_s166723_c330107.jpg | train | base300 | — | — |
| halftone_s166770_c130090.jpg | train | people100 | single_main | 1 |
| halftone_s166770_c273104.jpg | train | people100 | single_main | 0 |
| halftone_s166770_c79532.jpg | train | people100 | single_main | 0 |
| halftone_s166774_c369456.jpg | train | base300 | — | — |
| halftone_s166774_c377788.jpg | train | base300 | — | — |
| halftone_s166778_c7032.jpg | validation | people100 | small_group | 0 |
| halftone_s166782_c116218.jpg | train | base300 | — | — |
| halftone_s166812_c216015.jpg | train | base300 | — | — |
| halftone_s166812_c233728.jpg | train | base300 | — | — |
| halftone_s166812_c68490.jpg | validation | base300 | — | — |
| halftone_s166812_c93909.jpg | train | base300 | — | — |
| halftone_s166822_c334233.jpg | train | base300 | — | — |
| halftone_s166827_c114850.jpg | train | people100 | small_group | 0 |
| halftone_s166827_c140193.jpg | validation | people100 | single_main | 1 |
| halftone_s166827_c236780.jpg | train | base300 | — | — |
| halftone_s166827_c242640.jpg | train | base300 | — | — |
| halftone_s166827_c337596.jpg | train | people100 | crowd | 0 |
| halftone_s166832_c112019.jpg | train | people100 | single_main | 0 |
| halftone_s166832_c171518.jpg | train | people100 | single_main | 0 |
| halftone_s166832_c197573.jpg | train | people100 | single_main | 0 |
| halftone_s166832_c212917.jpg | train | base300 | — | — |
| halftone_s166832_c331448.jpg | train | people100 | single_main | 0 |
| halftone_s166837_c22381.jpg | train | base300 | — | — |
| halftone_s166837_c231291.jpg | train | base300 | — | — |
| halftone_s166837_c259451.jpg | train | base300 | — | — |
| halftone_s166837_c278636.jpg | train | base300 | — | — |
| halftone_s166837_c348114.jpg | train | base300 | — | — |
| halftone_s166850_c144720.jpg | validation | people100 | crowd | 0 |
| halftone_s166850_c299857.jpg | train | base300 | — | — |
| halftone_s166886_c243751.jpg | train | base300 | — | — |
| halftone_s166886_c288804.jpg | train | base300 | — | — |
| halftone_s166886_c380632.jpg | train | base300 | — | — |
| halftone_s166886_c41668.jpg | train | base300 | — | — |
| halftone_s166909_c197233.jpg | train | base300 | — | — |
| halftone_s166909_c303153.jpg | train | base300 | — | — |
| halftone_s166909_c75628.jpg | train | base300 | — | — |
| halftone_s166914_c281492.jpg | validation | base300 | — | — |
| halftone_s166914_c57864.jpg | train | people100 | small_group | 0 |
| halftone_s166920_c193708.jpg | train | base300 | — | — |
| halftone_s166920_c27898.jpg | train | people100 | crowd | 0 |
| halftone_s166920_c94338.jpg | train | base300 | — | — |
| halftone_s166935_c144250.jpg | train | base300 | — | — |
| halftone_s166935_c267271.jpg | train | people100 | crowd | 0 |
| halftone_s166935_c284466.jpg | train | base300 | — | — |
| halftone_s166952_c161739.jpg | train | base300 | — | — |
| halftone_s166952_c164015.jpg | train | base300 | — | — |
| halftone_s166995_c183607.jpg | train | base300 | — | — |
| halftone_s166995_c213324.jpg | train | base300 | — | — |
| halftone_s166995_c312831.jpg | train | base300 | — | — |
| halftone_s166996_c185123.jpg | train | base300 | — | — |
| halftone_s166996_c241923.jpg | train | base300 | — | — |
| halftone_s166996_c349085.jpg | train | base300 | — | — |
| halftone_s166996_c352140.jpg | train | base300 | — | — |
| halftone_s166996_c44702.jpg | train | base300 | — | — |
| halftone_s167003_c143784.jpg | train | base300 | — | — |
| halftone_s167003_c382008.jpg | train | base300 | — | — |
| halftone_s167003_c72514.jpg | train | base300 | — | — |
| halftone_s167003_c80038.jpg | train | base300 | — | — |
| halftone_s167015_c330199.jpg | train | base300 | — | — |
| halftone_s167015_c393886.jpg | validation | base300 | — | — |
| halftone_s167035_c41864.jpg | train | base300 | — | — |
| halftone_s167042_c255602.jpg | train | base300 | — | — |
| halftone_s167042_c362532.jpg | train | base300 | — | — |
| halftone_s167053_c171703.jpg | train | base300 | — | — |
| halftone_s167053_c186462.jpg | train | base300 | — | — |
| halftone_s167053_c281028.jpg | train | base300 | — | — |
| halftone_s167053_c63469.jpg | validation | base300 | — | — |
| halftone_s167054_c109292.jpg | validation | base300 | — | — |
| halftone_s167054_c209956.jpg | train | base300 | — | — |
| halftone_s167054_c242895.jpg | train | base300 | — | — |
| halftone_s167057_c131030.jpg | train | base300 | — | — |
| halftone_s167057_c181526.jpg | validation | base300 | — | — |
| halftone_s167057_c206180.jpg | train | base300 | — | — |
| halftone_s167057_c246498.jpg | train | base300 | — | — |
| halftone_s167057_c380606.jpg | train | base300 | — | — |
| halftone_s167057_c42428.jpg | train | base300 | — | — |
| halftone_s167057_c84228.jpg | train | base300 | — | — |
| halftone_s167058_c39552.jpg | train | base300 | — | — |
| halftone_s167068_c204486.jpg | train | base300 | — | — |
| halftone_s167068_c277683.jpg | train | base300 | — | — |
| halftone_s167068_c282820.jpg | train | base300 | — | — |
| halftone_s167217_c136959.jpg | train | people100 | small_group | 0 |
| halftone_s167217_c346522.jpg | train | base300 | — | — |
| halftone_s167217_c376644.jpg | train | base300 | — | — |
| halftone_s167217_c377390.jpg | train | base300 | — | — |
| halftone_s167217_c44882.jpg | train | base300 | — | — |
| halftone_s167224_c129297.jpg | train | base300 | — | — |
| halftone_s167225_c41576.jpg | train | base300 | — | — |
| halftone_s167225_c80081.jpg | train | base300 | — | — |
| halftone_s167240_c177776.jpg | train | base300 | — | — |
| halftone_s167247_c226088.jpg | train | base300 | — | — |
| halftone_s167247_c23346.jpg | train | base300 | — | — |
| halftone_s167247_c375982.jpg | train | base300 | — | — |
| halftone_s167247_c83739.jpg | train | base300 | — | — |
| halftone_s167250_c118472.jpg | train | base300 | — | — |
| halftone_s167250_c192660.jpg | train | people100 | single_main | 0 |
| halftone_s167250_c198904.jpg | train | people100 | single_main | 0 |
| halftone_s167250_c351034.jpg | train | base300 | — | — |
| halftone_s167250_c352728.jpg | train | people100 | single_main | 0 |
| halftone_s167257_c216087.jpg | validation | base300 | — | — |
| halftone_s167257_c335206.jpg | train | base300 | — | — |
| halftone_s167257_c81798.jpg | train | people100 | small_group | 0 |
| halftone_s167263_c127030.jpg | train | people100 | single_main | 1 |
| halftone_s167263_c222848.jpg | validation | people100 | single_main | 0 |
| halftone_s167263_c385393.jpg | train | base300 | — | — |
| halftone_s167284_c139565.jpg | train | base300 | — | — |
| halftone_s167284_c23873.jpg | train | base300 | — | — |
| halftone_s167284_c240970.jpg | train | base300 | — | — |
| halftone_s167284_c379312.jpg | train | base300 | — | — |
| halftone_s167287_c164334.jpg | train | base300 | — | — |
| halftone_s167287_c328405.jpg | train | base300 | — | — |
| halftone_s167287_c49885.jpg | train | base300 | — | — |
| halftone_s167300_c243662.jpg | train | people100 | single_main | 0 |
| halftone_s167300_c24660.jpg | train | people100 | single_main | 0 |
| halftone_s167300_c248396.jpg | validation | base300 | — | — |
| halftone_s167300_c307943.jpg | train | people100 | single_main | 0 |
| halftone_s167300_c349654.jpg | train | base300 | — | — |
| halftone_s167302_c231641.jpg | validation | base300 | — | — |
| halftone_s167302_c378113.jpg | train | base300 | — | — |
| halftone_s167303_c117897.jpg | validation | base300 | — | — |
| halftone_s167303_c159252.jpg | train | people100 | single_main | 0 |
| halftone_s167303_c298269.jpg | train | people100 | single_main | 1 |
| halftone_s167303_c300554.jpg | train | people100 | small_group | 0 |
| halftone_s167303_c358617.jpg | train | people100 | single_main | 1 |
| halftone_s167303_c63846.jpg | train | people100 | small_group | 1 |
| halftone_s167303_c87069.jpg | train | people100 | small_group | 0 |
| halftone_s167310_c150659.jpg | train | people100 | small_group | 0 |
| halftone_s167310_c164373.jpg | train | people100 | single_main | 1 |
| halftone_s167310_c183873.jpg | train | people100 | single_main | 1 |
| halftone_s167310_c219433.jpg | train | people100 | single_main | 0 |
| halftone_s167310_c250338.jpg | train | base300 | — | — |
| halftone_s167322_c115320.jpg | train | base300 | — | — |
| halftone_s167322_c216685.jpg | train | base300 | — | — |
| halftone_s167322_c95387.jpg | train | base300 | — | — |
| halftone_s167331_c141973.jpg | validation | base300 | — | — |
| halftone_s167331_c233038.jpg | validation | base300 | — | — |
| halftone_s167331_c242872.jpg | train | base300 | — | — |
| halftone_s167331_c350423.jpg | train | base300 | — | — |
| halftone_s167350_c1604.jpg | train | people100 | small_group | 0 |
| halftone_s167350_c238390.jpg | train | base300 | — | — |
| halftone_s167350_c382159.jpg | train | base300 | — | — |
| halftone_s167350_c385164.jpg | train | base300 | — | — |
| halftone_s167353_c219024.jpg | train | base300 | — | — |
| halftone_s167357_c146147.jpg | train | base300 | — | — |
| halftone_s167357_c184173.jpg | train | base300 | — | — |
| halftone_s167357_c365891.jpg | train | base300 | — | — |
| halftone_s167361_c131326.jpg | train | people100 | single_main | 1 |
| halftone_s167361_c251687.jpg | train | base300 | — | — |
| halftone_s167361_c66333.jpg | train | base300 | — | — |
| halftone_s167366_c180243.jpg | train | base300 | — | — |
| halftone_s167366_c307698.jpg | train | base300 | — | — |
| halftone_s167366_c321096.jpg | train | base300 | — | — |
| halftone_s167366_c354533.jpg | validation | base300 | — | — |
| halftone_s167378_c102206.jpg | train | base300 | — | — |
| halftone_s167382_c243398.jpg | train | base300 | — | — |
| halftone_s167382_c273691.jpg | train | people100 | single_main | 1 |
| halftone_s167382_c339032.jpg | train | base300 | — | — |
| halftone_s167382_c378510.jpg | train | people100 | small_group | 0 |
| halftone_s167384_c289749.jpg | train | base300 | — | — |
| halftone_s167384_c362938.jpg | train | people100 | small_group | 0 |
| halftone_s167384_c368379.jpg | train | people100 | small_group | 0 |
| halftone_s167384_c91549.jpg | train | base300 | — | — |
| halftone_s167386_c13194.jpg | train | base300 | — | — |
| halftone_s167386_c204290.jpg | train | base300 | — | — |
| halftone_s167386_c366362.jpg | train | people100 | small_group | 0 |
| halftone_s167388_c358637.jpg | train | people100 | single_main | 1 |
| halftone_s167388_c375004.jpg | train | base300 | — | — |
| halftone_s167388_c92026.jpg | train | people100 | single_main | 0 |
| halftone_s167393_c234459.jpg | train | base300 | — | — |
| halftone_s167393_c2900.jpg | train | base300 | — | — |
| halftone_s167393_c311286.jpg | train | base300 | — | — |
| halftone_s167393_c364184.jpg | train | people100 | small_group | 0 |
| halftone_s167393_c394454.jpg | train | base300 | — | — |
| halftone_s167393_c85009.jpg | train | base300 | — | — |
| halftone_s167394_c149331.jpg | train | people100 | single_main | 0 |
| halftone_s167394_c164546.jpg | train | people100 | single_main | 1 |
| halftone_s167394_c16986.jpg | train | base300 | — | — |
| halftone_s167394_c387595.jpg | train | base300 | — | — |
| halftone_s167394_c71186.jpg | train | people100 | single_main | 0 |
| halftone_s167397_c146112.jpg | train | base300 | — | — |
| halftone_s167397_c265890.jpg | train | base300 | — | — |
| halftone_s167397_c294932.jpg | train | base300 | — | — |
| halftone_s167397_c86525.jpg | train | base300 | — | — |
| halftone_s167400_c26132.jpg | train | base300 | — | — |
| halftone_s167400_c358358.jpg | train | base300 | — | — |
| halftone_s167400_c40913.jpg | train | base300 | — | — |
| halftone_s167404_c183791.jpg | train | base300 | — | — |
| halftone_s167404_c212035.jpg | train | base300 | — | — |
| halftone_s167404_c280253.jpg | validation | base300 | — | — |
| halftone_s167404_c284130.jpg | train | base300 | — | — |
| halftone_s167404_c284170.jpg | train | base300 | — | — |
| halftone_s167404_c295744.jpg | train | base300 | — | — |
| halftone_s167407_c38284.jpg | train | people100 | small_group | 0 |
| halftone_s167407_c391690.jpg | train | base300 | — | — |
| halftone_s167407_c81700.jpg | train | people100 | small_group | 0 |
| halftone_s167411_c312748.jpg | train | base300 | — | — |
| halftone_s167411_c345890.jpg | train | base300 | — | — |
| halftone_s167411_c397626.jpg | train | base300 | — | — |
| halftone_s20386_c120898.jpg | train | base300 | — | — |
| halftone_s20386_c244533.jpg | validation | base300 | — | — |
| halftone_s20386_c306829.jpg | train | base300 | — | — |
| halftone_s20386_c365073.jpg | train | base300 | — | — |
| halftone_s20387_c246659.jpg | train | people100 | small_group | 0 |
| halftone_s20387_c251282.jpg | validation | base300 | — | — |
| halftone_s20387_c25812.jpg | train | base300 | — | — |
| halftone_s20387_c379378.jpg | train | base300 | — | — |
| halftone_s20387_c97599.jpg | train | base300 | — | — |
| halftone_s20388_c159436.jpg | train | people100 | single_main | 0 |
| halftone_s20388_c325770.jpg | train | people100 | single_main | 1 |
| halftone_s20388_c357964.jpg | train | people100 | single_main | 1 |
| halftone_s20388_c379828.jpg | train | people100 | single_main | 0 |
| halftone_s20388_c82944.jpg | train | people100 | small_group | 0 |
| halftone_s44294_c104935.jpg | train | base300 | — | — |
| halftone_s44294_c27652.jpg | train | base300 | — | — |
| halftone_s44294_c305911.jpg | train | base300 | — | — |
| halftone_s44294_c308898.jpg | train | base300 | — | — |
| halftone_s44294_c35619.jpg | train | base300 | — | — |
