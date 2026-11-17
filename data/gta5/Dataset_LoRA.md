# Dataset per il LoRA GTA5

Aggiornato al **17 novembre 2026**. Contiene **400 immagini JPEG con caption**,
suddivise in **360 train** e **40 validation**, per addestrare adattatori LoRA
di Stable Diffusion 1.5 nello stile **GTA V in-game**.

La selezione segue il riferimento [Pixel Art](../pixel_art/Dataset_LoRA.md):
300 esempi di contenuto vario e 100 ulteriori scene con persone nel contesto.
Le immagini e gli split Pixel Art non sono stati modificati. I file GTA5 sono
già presenti localmente; **nessun LoRA GTA5 è stato addestrato** in questa fase.

## Stile e fonti

La definizione adottata è quella di `configs/styles.toml`: resa tridimensionale
del gioco, materiali e texture, illuminazione e profondità atmosferica.
Non è un dataset delle illustrazioni delle schermate di caricamento.
Le immagini provengono da screenshot GTA V e GTA Online pubblicati nelle
seguenti gallerie GTABase, che li presentano come screenshot ufficiali Rockstar:

| Galleria | Train | Validation | Totale |
|---|---:|---:|---:|
| [PS3 / Xbox 360](https://www.gtabase.com/grand-theft-auto-v/screenshots/ps3-360-screenshots/) | 126 | 13 | 139 |
| [PS4 / Xbox One / PC](https://www.gtabase.com/grand-theft-auto-v/screenshots/ps4-xb1-pc-screenshots/) | 128 | 18 | 146 |
| [GTA Online](https://www.gtabase.com/grand-theft-auto-v/screenshots/gta-online/) | 106 | 9 | 115 |
| **Totale** | **360** | **40** | **400** |

Si tratta di tre raccolte dello stesso universo visivo, non di tre dataset
indipendenti. Delle 572 immagini candidate scaricate sono state mantenute
400 dopo revisione visiva di originali e anteprime del crop centrale.

Le URL dei JPEG, gli ID della galleria, i titoli sorgente, le dimensioni e gli
hash SHA256 sono in [selection.csv](selection.csv). La galleria è una fonte
web mutabile: non dispone di una revisione immutabile equivalente ai commit
Hugging Face del dataset Pixel Art. Gli hash congelano i file effettivamente
usati e consentono di rilevare eventuali cambiamenti a un successivo download.

**Licenze:** le pagine consultate non dichiarano per questi screenshot una
licenza aperta Apache-2.0/MIT equivalente a quelle delle fonti Pixel Art.
La disponibilità pubblica non costituisce una concessione esplicita per
training o redistribuzione; i diritti restano ai rispettivi titolari.
Questa selezione non certifica autorizzazioni d'uso delle singole immagini.
Se il protocollo richiede esclusivamente immagini con licenza aperta esplicita,
questa fonte non soddisfa tale requisito e occorre sostituirla.

Le altre fonti esaminate non sono state importate:

- [GTA Image Captioning](https://huggingface.co/datasets/viplismism/GTA-Image-Captioning-Dataset):
  il primo shard ispezionato è dominato da fotogrammi stradali consecutivi;
  le caption campionate presentano errori e la card non esplicita una licenza.
- [Multi-Style](https://huggingface.co/datasets/yufan/image_style_transfer_GPTImage2):
  non contiene una colonna GTA5.
- [GTA-HDR](https://github.com/HrishavBakulBarua/GTA-HDR):
  il repository richiede l'invio di un accordo agli autori per ottenere i dati.
  Non sono state inviate richieste né accettati accordi a nome dell'utente.
- [Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/index.html):
  fonte pertinente per la resa del gioco, ma centrata sulle scene stradali.

## Composizione e criteri di selezione

| Sottoinsieme | Train | Validation | Totale |
|---|---:|---:|---:|
| Contenuto vario, `base300` | 270 | 30 | 300 |
| Scene aggiuntive con persone, `people100` | 90 | 10 | 100 |
| **Totale** | **360** | **40** | **400** |

Le 100 scene aggiuntive sono disgiunte dalle 300 di base, anche quando queste
ultime contengono persone. Riprendono esattamente le quote del riferimento:

| Composizione nelle 100 aggiuntive | Train | Validation | Totale |
|---|---:|---:|---:|
| Folla | 9 | 1 | 10 |
| Piccolo gruppo | 38 | 4 | 42 |
| Persona principale con ambiente riconoscibile | 43 | 5 | 48 |

**24 delle 100** hanno etichetta `wide=1`: figure distanti o un'inquadratura
ampia con contesto leggibile. È una caratteristica trasversale, non un quarto
gruppo da sommare al totale. Le etichette sono qualitative; alcune viste ampie
originali diventano inquadrature più ravvicinate nel crop quadrato, e alcune
scene con persona principale contengono figure secondarie.

La selezione privilegia contenuti utili al trasferimento di fotografie:

- paesaggi montani e costieri, fiumi, laghi, foreste, deserti e campagna;
- skyline, ponti, infrastrutture, edifici, quartieri e strade;
- interni abitati, uffici, studi di registrazione, laboratori e spazi ricreativi;
- persone in attività, sport, piccoli gruppi, folle e figure nel paesaggio;
- veicoli terrestri, aerei e imbarcazioni, animali e alcuni oggetti di scena;
- luce diurna, tramonti, notte, nebbia, pioggia e temporali.

Sono stati esclusi menu delle impostazioni, collage promozionali evidenti,
poster e illustrazioni dominanti, primi piani senza sufficiente contesto e
numerose varianti ridondanti. Restano mezzi busti ambientati e scene d'azione,
come nel riferimento restano esempi non esclusivamente panoramici.
Non sono state aggiunte fotografie raw, altre stilizzazioni o mappe semantiche.
Non servono coppie fotografia/target per il trainer.

**Cosa viene replicato:** numerosità, rapporto 90/10, struttura 300+100,
quote di composizione delle persone, caption di contenuto, controllo visivo
del crop, tracciabilità e controlli contro duplicati e contaminazione degli split.
Le 300 immagini GTA5 vengono congelate con questa prima selezione: non sono
le 300 immagini Pixel Art trasformate e non condividono gli stessi soggetti.

**Differenze:** non si riproducono le proporzioni delle fonti Pixel Art né una
corrispondenza immagine per immagine. GTA5 contiene più trasporti, personaggi
maschili e scene d'azione; cibo, piante isolate e varietà di animali sono meno
rappresentati. Le due raccolte non sono semanticamente bilanciate per un
confronto causale fra stili. Licenze e immutabilità delle fonti differiscono.

## Caption e compatibilità con il trainer

Le caption inglesi descrivono il contenuto visibile e iniziano con
`GTA V in-game rendering.`. Sono state redatte tramite revisione visiva,
con correzioni quando il crop perdeva elementi laterali. Non derivano da una
sostituzione testuale delle caption Pixel Art né dalle caption automatiche del
campione scartato. I nomi commerciali dei veicoli e i nomi dei personaggi non
sono necessari a rappresentare il contenuto.

Le caption effettive sono nei `metadata.jsonl`: ogni riga contiene `file_name`
e `text`, con il nome relativo allo split. Il manifest ne conserva una copia
per ricostruire il dataset. Controllo con il tokenizer CLIP locale del backbone:
**16–25 token per caption**, entro il limite di **77**, senza troncamento.

```text
data/gta5/
├── Dataset_LoRA.md
├── selection.csv             # selezione congelata, caption, split, fonti e hash
├── validation_report.json    # risultati dei controlli automatici
├── review.html               # galleria locale con anteprime del crop centrale
├── train/                   # 360 JPEG + metadata.jsonl
└── validation/              # 40 JPEG + metadata.jsonl
```

I JPEG conservano i byte scaricati, senza ricompressione locale. Sono le varianti
esposte dalla galleria, non una garanzia degli originali Rockstar alla massima
risoluzione. Le dimensioni vanno da 640×360 a 1920×1080; il report elenca tutti
i formati. Alcuni file contengono loghi o scritte: il crop centrale esclude molti
marchi d'angolo ma non garantisce l'assenza di watermark, loghi o testo nel gioco.

Il loader esistente `PixelArtDataset`, nonostante il nome, accetta anche questi
file senza modifiche al codice. Applica orientamento EXIF, RGB, resize
proporzionale e crop centrale a **512×512**, con normalizzazione in [-1, 1].
I config GTA5 impostano **Lanczos**, supportato dal loader, al posto del nearest
usato per conservare la griglia Pixel Art. Questa è una differenza di
preprocessing esplicita, non di selezione. Non si applicano crop casuali o flip.

## Integrità, split e riproducibilità

Revisione: **`gta5-context400-2026-11-17`**. I 400 file occupano circa **69,8 MB**.

Controlli completati con `scripts/build_gta5_dataset.py`:

- conteggi 360/40 e 300+100, comprese le quote delle persone nel validation;
- corrispondenza di tutti i file con dimensioni e hash del manifest;
- nessun duplicato esatto dei byte, dei pixel RGB o dei crop di training;
- nessuna coppia con distanza dHash ≤ 4, su originali o crop centrali;
- 345 gruppi di scene/varianti note, ciascuno confinato a un solo split;
- confronto con le **108 foto di inferenza** attualmente in `data/test_images`:
  nessun duplicato RGB esatto e nessuna coppia con dHash ≤ 4;
- decodifica e preprocessing di tutte le immagini con il loader del progetto;
- audit attraverso `main.py ... train --check-data`, senza caricare modelli.

I gruppi comprendono, fra gli altri, varianti di missioni, viste del medesimo
ponte, del lungomare e di alcune stanze. Il seed usato per allocare i gruppi
è 42; l'assegnazione finale è fissata nel manifest. Lo split non corrisponde a
uno split ufficiale delle fonti. Questi controlli non escludono tutte le somiglianze
semantiche, le identità ricorrenti o i luoghi condivisi dello stesso mondo di gioco.

Firme del loader, che includono nomi, caption e byte:

- train: `cf7128ed5f80282ca09c568032890149a9905963cd6be292e580b290e99ea2f2`;
- validation: `8440ae3ee9b9d27507ec16199073ea3f217eed6194a437514d3e2398e2298fd0`;

Le stesse immagini, caption e assegnazioni vanno mantenute per i rank 4/8/16/32.
Le foto di inferenza restano separate e non partecipano all'addestramento.
La qualità del futuro LoRA andrà valutata sulle generazioni: il superamento
dell'audit non è una verifica della qualità del modello.

Verifica del dataset già presente:

```bash
python3 scripts/build_gta5_dataset.py
python3 main.py --lora lora_gta5_rank4.toml train --check-data
```

Ripristino dei soli file mancanti, con verifica obbligatoria degli hash:

```bash
python3 scripts/build_gta5_dataset.py --download
```

Il ripristino usa la cache se disponibile, altrimenti le URL del manifest;
richiede accesso alla rete e fallisce se i byte della fonte sono cambiati.
`scripts/collect_gta5_candidates.py` conserva il procedimento di raccolta dei
candidati; non deve essere rieseguito per usare o ripristinare i 400 selezionati.

Sono pronti i config `lora_gta5_rank4.toml`, `lora_gta5_rank8.toml`,
`lora_gta5_rank16.toml` e `lora_gta5_rank32.toml`, con output separati in
`lora/gta5_rank_*`. Un eventuale training si avvia esplicitamente con:

```bash
python3 main.py --lora lora_gta5_rank4.toml train
```

## Elenco completo delle immagini selezionate

`gta5_NNNNN.jpg` usa l'ID numerico della galleria, non un indice sequenziale
di selezione. Il manifest riporta anche la URL diretta del singolo JPEG.
`wide` è annotato solo per il sottoinsieme delle 100 persone: `0` nelle 300
non significa che queste siano prive di vedute ampie.

| File | Split | Raccolta | Sottoinsieme | Composizione | Wide |
|---|---|---|---|---|---|
| gta5_00001.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00002.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00003.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00004.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00005.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00006.jpg | validation | PS3 / 360 | people100 | single_main | 1 |
| gta5_00007.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00008.jpg | validation | PS3 / 360 | people100 | small_group | 0 |
| gta5_00009.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00011.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00013.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00014.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00016.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00017.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00018.jpg | train | PS3 / 360 | people100 | crowd | 0 |
| gta5_00019.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00021.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00022.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00024.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00025.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00028.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00030.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00031.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00033.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00035.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00036.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00037.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00038.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00039.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00042.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00044.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00045.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00047.jpg | train | PS3 / 360 | people100 | crowd | 0 |
| gta5_00049.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00051.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00055.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00058.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00060.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00061.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00065.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00066.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00069.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00070.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00071.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00073.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00074.jpg | validation | PS3 / 360 | people100 | single_main | 1 |
| gta5_00075.jpg | train | PS3 / 360 | people100 | crowd | 0 |
| gta5_00076.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00078.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00079.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00083.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00084.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00085.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00086.jpg | validation | PS3 / 360 | people100 | single_main | 1 |
| gta5_00090.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00092.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00096.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00100.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00103.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00104.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00106.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00107.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00110.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00111.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00115.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00116.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00117.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00122.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00124.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00128.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00135.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00137.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00139.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00140.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00144.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00145.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00146.jpg | validation | PS3 / 360 | people100 | single_main | 0 |
| gta5_00148.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00151.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00152.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00153.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00155.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00157.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00159.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00160.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00166.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00168.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00171.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00172.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00174.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00175.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00176.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00177.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00178.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00179.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00180.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00181.jpg | train | PS3 / 360 | people100 | crowd | 1 |
| gta5_00182.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00183.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00184.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00185.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00187.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00188.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00190.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00192.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00193.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00194.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00195.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00196.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00197.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00198.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00199.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00201.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00202.jpg | train | PS3 / 360 | people100 | single_main | 0 |
| gta5_00203.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00204.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00205.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00206.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00209.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00210.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00211.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00212.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00213.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00214.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00215.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00224.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00225.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00226.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00228.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00230.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00231.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00232.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00233.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00234.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00236.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00237.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00239.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00240.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00242.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00244.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00245.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00246.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00247.jpg | train | PS4 / XB1 / PC | people100 | single_main | 0 |
| gta5_00248.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00249.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00250.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00251.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00252.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00253.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00254.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00255.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00256.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00257.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00258.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00259.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00260.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00261.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00262.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00263.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00264.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00265.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00266.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00267.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00268.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00269.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00270.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00271.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00272.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00273.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00274.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00277.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00279.jpg | train | PS4 / XB1 / PC | people100 | single_main | 0 |
| gta5_00280.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00282.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00284.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00287.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00288.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00291.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00293.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00296.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00297.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00300.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00304.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00306.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00307.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00309.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00310.jpg | train | PS4 / XB1 / PC | people100 | single_main | 0 |
| gta5_00313.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00314.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00318.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00319.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00320.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00322.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00323.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00326.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00328.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00329.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00330.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00331.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00333.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00335.jpg | validation | PS4 / XB1 / PC | people100 | crowd | 0 |
| gta5_00336.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_00338.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00339.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_00340.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00341.jpg | validation | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00342.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00343.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00344.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00346.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00348.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00349.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00350.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_00351.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00352.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00354.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00356.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00357.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00358.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00359.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00361.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00362.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00363.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_00714.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00720.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00721.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00722.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00724.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00725.jpg | train | GTA Online | base300 | — | — |
| gta5_00726.jpg | train | GTA Online | base300 | — | — |
| gta5_00727.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_00729.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_00730.jpg | validation | GTA Online | base300 | — | — |
| gta5_00731.jpg | train | GTA Online | base300 | — | — |
| gta5_00746.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_00747.jpg | train | GTA Online | base300 | — | — |
| gta5_00749.jpg | train | GTA Online | base300 | — | — |
| gta5_00750.jpg | train | GTA Online | base300 | — | — |
| gta5_00751.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_00752.jpg | train | GTA Online | base300 | — | — |
| gta5_00827.jpg | train | GTA Online | base300 | — | — |
| gta5_00841.jpg | train | GTA Online | base300 | — | — |
| gta5_00978.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_00980.jpg | train | GTA Online | base300 | — | — |
| gta5_00990.jpg | validation | PS3 / 360 | base300 | — | — |
| gta5_00993.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00994.jpg | train | PS3 / 360 | people100 | small_group | 0 |
| gta5_00995.jpg | train | PS3 / 360 | people100 | single_main | 1 |
| gta5_00997.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00998.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_00999.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_01000.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_01001.jpg | train | PS3 / 360 | base300 | — | — |
| gta5_01002.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01003.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_01004.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01005.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01006.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01007.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01009.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01010.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01014.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01015.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01016.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01017.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01018.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01019.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01020.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01021.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01022.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01023.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01024.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01025.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01027.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01028.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01029.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01031.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01032.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_01034.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01037.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01038.jpg | validation | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_01039.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_01041.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01050.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01051.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01052.jpg | train | PS4 / XB1 / PC | people100 | small_group | 0 |
| gta5_01054.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01055.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01057.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01058.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01060.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01062.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01063.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01064.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01065.jpg | train | PS4 / XB1 / PC | people100 | single_main | 1 |
| gta5_01067.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01068.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01069.jpg | validation | PS4 / XB1 / PC | base300 | — | — |
| gta5_01070.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_01195.jpg | validation | GTA Online | base300 | — | — |
| gta5_01219.jpg | train | GTA Online | base300 | — | — |
| gta5_01227.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_01234.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_01235.jpg | train | GTA Online | base300 | — | — |
| gta5_01238.jpg | train | GTA Online | base300 | — | — |
| gta5_01389.jpg | validation | GTA Online | people100 | small_group | 0 |
| gta5_01390.jpg | train | GTA Online | base300 | — | — |
| gta5_01393.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_01400.jpg | train | GTA Online | base300 | — | — |
| gta5_01447.jpg | train | GTA Online | base300 | — | — |
| gta5_01454.jpg | train | GTA Online | base300 | — | — |
| gta5_01670.jpg | train | GTA Online | people100 | single_main | 1 |
| gta5_01671.jpg | train | GTA Online | base300 | — | — |
| gta5_02042.jpg | train | GTA Online | base300 | — | — |
| gta5_02052.jpg | train | GTA Online | base300 | — | — |
| gta5_02053.jpg | train | GTA Online | base300 | — | — |
| gta5_02054.jpg | train | GTA Online | base300 | — | — |
| gta5_02260.jpg | train | GTA Online | base300 | — | — |
| gta5_03608.jpg | train | GTA Online | base300 | — | — |
| gta5_03664.jpg | train | GTA Online | base300 | — | — |
| gta5_03958.jpg | train | GTA Online | base300 | — | — |
| gta5_03959.jpg | train | GTA Online | base300 | — | — |
| gta5_03960.jpg | train | GTA Online | base300 | — | — |
| gta5_03961.jpg | train | GTA Online | base300 | — | — |
| gta5_03962.jpg | train | GTA Online | base300 | — | — |
| gta5_03963.jpg | train | GTA Online | base300 | — | — |
| gta5_03964.jpg | train | GTA Online | base300 | — | — |
| gta5_03965.jpg | train | GTA Online | base300 | — | — |
| gta5_03966.jpg | train | GTA Online | base300 | — | — |
| gta5_03967.jpg | train | GTA Online | base300 | — | — |
| gta5_03968.jpg | train | GTA Online | base300 | — | — |
| gta5_03969.jpg | train | GTA Online | people100 | crowd | 0 |
| gta5_03971.jpg | train | GTA Online | base300 | — | — |
| gta5_03972.jpg | train | GTA Online | base300 | — | — |
| gta5_03973.jpg | train | GTA Online | base300 | — | — |
| gta5_03974.jpg | train | GTA Online | base300 | — | — |
| gta5_03975.jpg | train | GTA Online | base300 | — | — |
| gta5_03976.jpg | train | GTA Online | base300 | — | — |
| gta5_03977.jpg | train | GTA Online | base300 | — | — |
| gta5_03978.jpg | train | GTA Online | base300 | — | — |
| gta5_03979.jpg | train | GTA Online | base300 | — | — |
| gta5_03980.jpg | train | GTA Online | base300 | — | — |
| gta5_03981.jpg | validation | GTA Online | base300 | — | — |
| gta5_03982.jpg | train | GTA Online | people100 | crowd | 0 |
| gta5_03984.jpg | train | GTA Online | base300 | — | — |
| gta5_03985.jpg | train | GTA Online | base300 | — | — |
| gta5_03986.jpg | train | GTA Online | base300 | — | — |
| gta5_03988.jpg | train | GTA Online | base300 | — | — |
| gta5_03990.jpg | train | GTA Online | base300 | — | — |
| gta5_03993.jpg | train | GTA Online | base300 | — | — |
| gta5_03995.jpg | train | GTA Online | people100 | single_main | 0 |
| gta5_03997.jpg | validation | GTA Online | base300 | — | — |
| gta5_03999.jpg | train | GTA Online | base300 | — | — |
| gta5_04000.jpg | train | GTA Online | base300 | — | — |
| gta5_04003.jpg | train | GTA Online | base300 | — | — |
| gta5_04004.jpg | train | GTA Online | base300 | — | — |
| gta5_04041.jpg | train | GTA Online | base300 | — | — |
| gta5_04044.jpg | train | GTA Online | base300 | — | — |
| gta5_04069.jpg | train | GTA Online | base300 | — | — |
| gta5_04116.jpg | train | GTA Online | base300 | — | — |
| gta5_04384.jpg | train | GTA Online | base300 | — | — |
| gta5_04402.jpg | validation | GTA Online | base300 | — | — |
| gta5_04403.jpg | train | GTA Online | base300 | — | — |
| gta5_06536.jpg | train | GTA Online | base300 | — | — |
| gta5_06537.jpg | train | GTA Online | base300 | — | — |
| gta5_06538.jpg | train | GTA Online | base300 | — | — |
| gta5_06874.jpg | train | GTA Online | base300 | — | — |
| gta5_06880.jpg | train | GTA Online | base300 | — | — |
| gta5_06881.jpg | train | GTA Online | people100 | crowd | 0 |
| gta5_06890.jpg | train | GTA Online | base300 | — | — |
| gta5_06974.jpg | train | GTA Online | base300 | — | — |
| gta5_06991.jpg | train | GTA Online | base300 | — | — |
| gta5_07010.jpg | train | GTA Online | base300 | — | — |
| gta5_07061.jpg | train | GTA Online | base300 | — | — |
| gta5_07136.jpg | train | GTA Online | base300 | — | — |
| gta5_07221.jpg | train | GTA Online | people100 | crowd | 1 |
| gta5_07263.jpg | train | GTA Online | base300 | — | — |
| gta5_08413.jpg | train | GTA Online | base300 | — | — |
| gta5_08414.jpg | validation | GTA Online | people100 | single_main | 0 |
| gta5_08415.jpg | train | GTA Online | base300 | — | — |
| gta5_08416.jpg | train | GTA Online | base300 | — | — |
| gta5_08417.jpg | train | GTA Online | base300 | — | — |
| gta5_08687.jpg | train | GTA Online | base300 | — | — |
| gta5_08758.jpg | train | GTA Online | base300 | — | — |
| gta5_08800.jpg | validation | GTA Online | base300 | — | — |
| gta5_11281.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_11297.jpg | train | PS4 / XB1 / PC | base300 | — | — |
| gta5_11354.jpg | train | GTA Online | base300 | — | — |
| gta5_11355.jpg | train | GTA Online | people100 | crowd | 0 |
| gta5_11366.jpg | train | GTA Online | base300 | — | — |
| gta5_11367.jpg | train | GTA Online | base300 | — | — |
| gta5_11368.jpg | train | GTA Online | base300 | — | — |
| gta5_11369.jpg | train | GTA Online | base300 | — | — |
| gta5_11370.jpg | train | GTA Online | base300 | — | — |
| gta5_11371.jpg | train | GTA Online | base300 | — | — |
| gta5_11372.jpg | train | GTA Online | base300 | — | — |
| gta5_11373.jpg | validation | GTA Online | base300 | — | — |
| gta5_11374.jpg | train | GTA Online | base300 | — | — |
| gta5_11522.jpg | train | GTA Online | people100 | small_group | 0 |
| gta5_11523.jpg | train | GTA Online | base300 | — | — |
