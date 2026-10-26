# MultiStyleImages — Roadmap operativa del progetto

> Documento di lavoro per il progetto magistrale di Deep Learning (team di 2 persone).  
> **Scopo:** sapere cosa fare, in quale ordine, come controllare che funzioni e quali decisioni non prendere prematuramente.  
> **Aggiornato:** 26 ottobre 2026 — stile LoRA per RQ2 fissato a **watercolor**.  
> **Stato:** piano proposto; le configurazioni numeriche sono **punti di partenza da validare**, non risultati sperimentali.

## 0. Visione del progetto

**Obiettivo applicativo:** data una fotografia e uno stile visivo (es. watercolor, GTA V, cyberpunk, pixel art), generare un'immagine stilizzata cercando di preservare contenuto e struttura della scena.

**Problema scientifico:** la fedeltà allo stile può entrare in tensione con la preservazione del contenuto. Il professore ha richiesto research question esplicite, esperimenti controllati, ablation study e metriche distinte per stile, contenuto e struttura.

### Research questions

**RQ1 — Structural Conditioning**  
*How do semantic segmentation, depth maps, and edge maps affect the trade-off between content preservation and style fidelity in structurally conditioned image generation?*

- Confrontare ControlNet **Semantic**, **Depth**, **Canny/Edge**.
- Stable Diffusion 1.5 e checkpoint preaddestrati **congelati**: nessun training richiesto per questa RQ.
- Studiare eventualmente `controlnet_conditioning_scale` come ablation secondaria.

**RQ2 — LoRA Rank**  
*How does LoRA rank affect the trade-off between style fidelity and content preservation in structurally conditioned image generation?*

- Fine-tuning LoRA su Stable Diffusion 1.5, **un rango alla volta**.
- Confronto **nessuna LoRA** vs ranghi **4, 8, 16, 32**.
- **Stile scelto: watercolor (acquerello)**. Addestrare quattro LoRA dello stesso stile, con ranghi diversi; altri stili fuori dal confronto principale.
- Nella comparazione primaria di RQ2 usare **un solo tipo di conditioning fisso**; confronto tra conditioning opzionale successivo.


---

## 0.1 Pipeline operativa aggiornata: Watercolor LoRA

**Due percorsi distinti: non confondere dataset di training e fotografie da stilizzare.**

```text
TRAINING RQ2 (una volta per rank)
Dipinti watercolor + caption
        │
        ▼
Dataset / dataloader → VAE encoder → latente z₀
                                      │
                           campiona timestep t e rumore ε
                                      │
                                      ▼
                               latente rumoroso zₜ
                                      │
Caption → CLIP text encoder ───────────┤
                                      ▼
                   U-Net SD 1.5 congelata + LoRA trainabile
                                      │
                                      ▼
                    predizione rumore ε̂ → MSE(ε̂, ε)
                                      │
                                      ▼
                    aggiorna SOLO i parametri LoRA
                                      │
                                      ▼
                     salva watercolor_rankXX.safetensors

INFERENZA RQ1 + RQ2 (per ogni fotografia nuova)
Fotografia → semantic / depth / canny map ──┐
                                           ▼
Prompt watercolor → SD 1.5 + ControlNet + LoRA opzionale
                                           │
                                           ▼
                                  immagine watercolor
```

- **Nel training LoRA non è necessaria la fotografia da stilizzare** e **non addestriamo il ControlNet**. Il dataset contiene *immagini già watercolor*, non fotografie originali da convertire.
- Per il primo esperimento usiamo solo `wtrclrstyle` (scelto convenzionalmente) nelle caption, sempre accompagnato da una descrizione del **soggetto**. La LoRA standard non crea un nuovo token di tokenizer.
- La prima LoRA da addestrare è **watercolor rank 8**, con poche immagini/step per lo smoke test; poi un training pilota adeguato.
- Se il caricamento funziona, ripetere con **rank 4, 16, 32** usando **lo stesso split watercolor**, budget di training, learning rate, moduli target, seed e protocollo di inferenza.
- Nell'**ablation primaria** mantenere **un solo ControlNet**, preferibilmente Semantic (già funzionante). RQ1 confronta invece i tre ControlNet **senza LoRA**.
- Il confronto `no-LoRA` è SD 1.5 + ControlNet + prompt watercolor, non una LoRA di rango zero.

**Esempio di dati di training:**

```text
data/style_watercolor/
├── train/
│   ├── 001.png
│   ├── 002.png
│   └── metadata.jsonl
└── validation/
    ├── 301.png
    └── metadata.jsonl
```

```jsonl
{"file_name":"001.png","text":"A forest landscape, in wtrclrstyle"}
{"file_name":"002.png","text":"An indoor cafe with tables and chairs, in wtrclrstyle"}
```

**Attenzione ai dati:** il campo `file_name` si riferisce ai file nella stessa directory del rispettivo metadata; validare il formato supportato dal `dataset.py` definitivo. Non usare nel training le foto del benchmark né le immagini di validazione per aggiornare i pesi.

**Criterio di successo del pilota:** checkpoint LoRA realmente caricabile nella pipeline `StableDiffusionControlNetPipeline`, generazioni watercolor su soggetti diversi e log completi. La qualità quantitativa verrà misurata più avanti nell'evaluation; un valore basso della training loss non basta a dimostrare il trasferimento dello stile.

---

## 1. Decisioni già prese vs ancora aperte

| Tema | Decisione / stato |
|---|---|
| Base generator | Stable Diffusion 1.5 |
| Baseline | SD 1.5 + ControlNet Seg + UPerNet ADE20K + prompt statico; nessuna LoRA |
| RQ1 | Semantic vs Depth vs Canny (senza fine-tuning) |
| RQ2 | LoRA **watercolor** sui rank 4/8/16/32, baseline no-LoRA |
| Training LoRA | Dataset di **dipinti watercolor con caption**, U-Net LoRA con loss di diffusion standard; ControlNet congelato e usato **solo in inferenza** |
| Evaluation principale | CLIP image-image (CLIP-I), CLIP image-text (CLIP-T), semantic consistency mIoU |
| Evaluation complementare | Confronti visivi e piccola valutazione umana; tempi, parametri e VRAM per RQ2 |
| Prompt builder | Già sviluppato, ma **non includerlo nei confronti primari**: mantenere i prompt fissi |
| Img2Img | Possibile estensione, non mescolare con l'ablation primaria |
| Demo | Alla fine, dopo avere congelato gli esperimenti |

---

## 2. Ordine di esecuzione: checklist

### Fase A — Congelare e verificare la baseline

- [ ] Assicurarsi che il codice della baseline funzioni su un'immagine singola.
- [ ] Annotare versioni librerie, modello SD 1.5, ControlNet, scheduler, dimensione output, numero di inference step, CFG, scale/end ControlNet, seed e preprocessing.
- [ ] Salvare **immagine input**, **control map**, **output** e JSON metadati per ogni run.
- [ ] Congelare le **100 foto di test** e salvarne l'elenco (`test_manifest.csv`); non usarle nel training LoRA né nella selezione dei checkpoint.
- [ ] Selezionare un **piccolo validation set distinto** per debug e selezione dei parametri.
- [ ] Creare un commit/tag Git per la baseline funzionante.

**Completata quando:** puoi rigenerare il medesimo risultato, in uno stesso ambiente, con stessa config e seed; i file sono associabili per ID.

### Fase B — Organizzare il progetto

Struttura suggerita (adattare alla repo già esistente; **non riscrivere i moduli che funzionano**):

```text
MultiStyleImages/
├── main.py                         # orchestratore inferenza / benchmark
├── configs/
│   ├── generation.toml
│   ├── lora_rank_04.toml
│   ├── lora_rank_08.toml
│   ├── lora_rank_16.toml
│   └── lora_rank_32.toml
├── src/
│   ├── generation/pipeline.py      # SD 1.5 + ControlNet; LoRA opzionale
│   ├── conditioning/
│   │   ├── semantic.py
│   │   ├── depth.py
│   │   └── canny.py
│   ├── training/
│   │   ├── dataset.py              # immagini + caption
│   │   └── train_lora.py          # training LoRA
│   └── evaluation/
│       └── evaluate.py            # CLIP-I, CLIP-T, mIoU
├── data/
│   ├── style_watercolor/train/
│   ├── style_watercolor/validation/
│   └── content_test/              # 100 foto, bloccate
├── outputs/
│   ├── lora_rank_04/
│   ├── lora_rank_08/
│   ├── lora_rank_16/
│   └── lora_rank_32/
├── results/
└── demo/app.py
```

- [ ] Separare **training** e **inferenza**: la generation pipeline non deve gestire l'optimizer.
- [ ] Usare TOML per configurare esperimenti (invece di cambiare numeri nel codice).
- [ ] Fare in modo che `load_lora_weights(...)` sia opzionale; senza adattatore la pipeline deve restare baseline.

### Fase C — Dataset per la prima LoRA

**Stile scelto per RQ2: watercolor (acquerello).** Non serve un dataset appaiato foto→acquerello: il fine-tuning text-to-image LoRA usa immagini già in stile watercolor con caption. Le fotografie da trasformare entrano solo durante l’inferenza tramite ControlNet.

- [ ] Reperire **dipinti/illustrazioni watercolor** con uso consentito e soggetti vari: città, strade, persone, paesaggi, foreste, interni, oggetti; partire indicativamente da **200–500 immagini** di buona qualità, poi adattare ai dati disponibili.
- [ ] Rimuovere duplicati/quasi-duplicati e immagini corrotte.
- [ ] Creare caption descrittive di **contenuto + stile**. Evitare che tutte le caption contengano solo `GTA V style`.
- [ ] Dividere le immagini di stile in **train/validation** senza overlap; il dataset di fotografie input per la valutazione resta separato.
- [ ] Controllare manualmente 20 coppie immagine/caption.
- [ ] Implementare `dataset.py` e un test che carichi un mini-batch con tensor shape attesa.

Esempio di caption: `A mountain road surrounded by trees, in wtrclrstyle` (oppure un trigger testuale coerente e stabile).

**Nota:** `wtrclrstyle` è una stringa/identificatore nelle caption; una LoRA standard **non aggiunge automaticamente un nuovo token al tokenizer**.

**Completata quando:** `dataset.py` riesce a leggere tutto il train split e generare un batch senza errori; train e validation non si sovrappongono.

### Fase D — Primo training pilota watercolor (LoRA r=8)

Usare come riferimento lo script di Diffusers per text-to-image LoRA:  
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

- [ ] Scegliere e **fissare le versioni** compatibili di `diffusers`, `transformers`, `accelerate`, `peft`, `torch`.
- [ ] Caricare SD 1.5 (specificare checkpoint/revisione).
- [ ] Congelare VAE, text encoder e pesi originali U-Net; allenare soltanto i parametri LoRA della U-Net.
- [ ] Fissare `target_modules` per tutti gli esperimenti (per esempio `to_q`, `to_k`, `to_v`, `to_out.0`, **se compatibili con lo script scelto**).
- [ ] Salvare training loss, validation loss, learning rate, seed, config, checkpoint, immagini di validazione e tempi.
- [ ] Eseguire un **smoke test** di pochi step su un mini-dataset; verificare che solo la LoRA riceva gradienti.
- [ ] Eseguire il training pilota completo con rank 8.
- [ ] Caricare la LoRA nella `generation/pipeline.py` con ControlNet preaddestrato, senza addestrare ControlNet.
- [ ] Generare poche foto di **validazione** con/senza LoRA, a prompt e seed fissati.

Configurazione **solo iniziale**, da adattare alla GPU Kaggle e ai risultati del pilota:

```toml
[training]
rank = 8
alpha = 8
learning_rate = 1e-4
resolution = 512
batch_size = 1
gradient_accumulation_steps = 4
max_train_steps = 1000
seed = 42
validation_every_steps = 250
```

**Completata quando:** il training termina, salva pesi caricabili, la pipeline produce immagini con l'adattatore e non ci sono errori di shape/VRAM.

### Fase E — Ablation sul rango LoRA (RQ2)

Configurazioni **sullo stesso dataset watercolor**: **no-LoRA, rank 4, 8, 16, 32**.

- [ ] Replicare la configurazione del training pilota e cambiare **solo** `rank` e `alpha` se si sceglie la convenzione `alpha=rank`.
- [ ] Tenere costanti immagini, caption, split, learning rate, numero di update, batch effettivo, moduli LoRA, scheduler training e risoluzione.
- [ ] Registrare il numero di parametri addestrabili, tempo e picco VRAM per ciascun rango.
- [ ] Conservare checkpoint intermedi o almeno uno finale e log.
- [ ] Verificare su un **validation set**, non sul test set, se servono cambi di training budget o scelta del checkpoint.
- [ ] Dopo aver fissato il protocollo, generare output sul **test set bloccato**.

**Nota metodologica:** `alpha=rank` mantiene `alpha/r=1` ma non garantisce che l'effetto del rank sia isolato perfettamente: cambiano numero di parametri e dinamiche di ottimizzazione. Il confronto a pari update non equivale a costo computazionale identico.

### Fase F — RQ1 e integrazione con RQ2

- [ ] Con la **LoRA disattivata**, confrontare ControlNet Semantic / Depth / Canny sulle stesse immagini e prompt.
- [ ] Usare moduli che producano mappe nel formato atteso dal **rispettivo checkpoint** ControlNet.
- [ ] Tenere fissi SD 1.5, seed, scheduler, risoluzione, CFG e inference steps; documentare i diversi checkpoint ControlNet.
- [ ] Se si vuole studiare la forza del controllo, ripetere su un piccolo insieme con `controlnet_conditioning_scale` variabile, mantenendo fisso il resto.
- [ ] Per RQ2 primaria selezionare **un conditioning fisso**, poi eventualmente provare gli altri con uno o due rank selezionati.

**Importante:** cambiare da Semantic a Depth o Canny cambia sia la rappresentazione **sia il checkpoint ControlNet**. Nel report parlare di confronto tra *sistemi di condizionamento*, non di un'unica rete alla quale si passa solo un'immagine diversa.

### Fase G — Evaluation condivisa

Tre misure quantitative fondamentali:

| Metrica | Input | Interpretazione | Limite |
|---|---|---|---|
| **CLIP-I** ↑ | Foto originale + output | Similarità semantica globale | Non garantisce identità o geometria |
| **CLIP-T** ↑ | Output + testo dello stile | Allineamento testo/immagine | Proxy imperfetto della fedeltà stilistica |
| **Semantic consistency mIoU** ↑ | Segmentazione originale + output | Overlap delle classi nella scena | Dipende da UPerNet; bias e difficoltà sulle immagini stilizzate |

- [ ] Implementare `evaluate.py` **separato** dalla generazione; nessuna nuova generazione per ricalcolare le metriche.
- [ ] Usare lo **stesso checkpoint CLIP e stessi prompt di valutazione** per tutti i metodi.
- [ ] Usare lo stesso protocollo UPerNet per tutte le immagini; allineare mappe e risoluzioni con nearest-neighbor prima di mIoU.
- [ ] Definire come gestire classi assenti e se riportare macro/micro; fissare la decisione prima del confronto finale.
- [ ] Salvare un `per_image_metrics.csv` e un `summary_metrics.csv` con media e deviazione standard per condizionamento, stile e rango.
- [ ] Aggiungere una piccola **valutazione umana** cieca (stile/contenuto/artefatti, scala 1–5) su immagini rappresentative.
- [ ] Ispezionare gli errori per categorie di scena, incluse le **foreste**, che la baseline gestisce male.
- [ ] Creare un grafico **CLIP-T vs CLIP-I** e/o **CLIP-T vs mIoU** per visualizzare il trade-off, senza condensare tutto in un unico numero arbitrario.

**Per FID/KID:** non sono prioritarie. Necessitano di **dataset di riferimento per ogni stile**; con circa 100 output per stile le stime distributive sono poco affidabili.

### Fase H — Ablation e report

Esperimenti essenziali:

| ID | Conditioning | LoRA | Prompt | Cosa isola |
|---|---|---|---|---|
| A | Semantic | No | Statico | Baseline |
| B | Depth | No | Statico | Conditioning RQ1 |
| C | Canny | No | Statico | Conditioning RQ1 |
| D | Semantic | Rank 4 | Statico | Rank RQ2 |
| E | Semantic | Rank 8 | Statico | Rank RQ2 |
| F | Semantic | Rank 16 | Statico | Rank RQ2 |
| G | Semantic | Rank 32 | Statico | Rank RQ2 |

- [ ] Formulare ipotesi **prima** di misurare (non scegliere l'ipotesi dopo aver visto i grafici).
- [ ] Confrontare differenze medie e dispersione; considerare intervalli di confidenza bootstrap appaiati sulle foto, se fattibile.
- [ ] Selezionare casi qualitativi: successo, errore di struttura, errore di contenuto, stile non convincente.
- [ ] Documentare limiti dei valutatori automatici, oltre ai possibili artefatti dovuti a bassa risoluzione.
- [ ] Scrivere `Method`, `Experimental Setup`, `Metrics`, `RQ1 Results`, `RQ2 Results`, `Ablations`, `Limitations`.

### Fase I — Demo finale (dopo gli esperimenti)

- [ ] Implementare interfaccia semplice (ad es. Gradio).
- [ ] Upload foto e selezione stile.
- [ ] Selezione del conditioning (Semantic/Depth/Canny).
- [ ] Selezione LoRA (off / rank 4 / 8 / 16 / 32 se disponibili).
- [ ] Visualizzazione input, control map, output e metadati.
- [ ] Impostazioni predefinite coerenti con gli esperimenti; non promettere prestazioni in real time senza test.

---

## 3. Checklist prima di ogni training Kaggle

- [ ] Dataset montato e numero immagini verificato (`/kaggle/input/...`).
- [ ] GPU abilitata e memoria disponibile.
- [ ] Checkpoint corretto e percorso di output in `/kaggle/working/...`.
- [ ] Nome run univoco, ad es. `watercolor_rank08_seed42`.
- [ ] Config, seed, split, moduli addestrabili e versioni librerie registrati.
- [ ] Verificare che il salvataggio dei checkpoint sia periodico.
- [ ] **Save & Run All** solo dopo smoke test; verificare poi lo stato della versione e gli output salvati.
- [ ] Copiare/download dei checkpoint finali e log importanti (non affidarsi soltanto a file temporanei della sessione).

## 4. Checklist prima di ogni confronto

- [ ] Stesse fotografie e stesso ordine dei file.
- [ ] Stesso stile, prompt, negative prompt e seed.
- [ ] Stessa risoluzione, num inference steps, CFG, scheduler.
- [ ] Stessa modalità generativa: non mescolare txt2img con img2img in una RQ che non li sta confrontando.
- [ ] Per RQ2: **stesso ControlNet** per tutti i rank.
- [ ] Annotare differenze inevitabili (checkpoint tra condizionamenti, costi diversi per rank).
- [ ] Output non sovrascritti e metadati completi.

## 5. Piano di lavoro consigliato per due persone

| Lavoro | Persona A | Persona B | Insieme |
|---|---|---|---|
| Baseline / condizionamenti | Responsabile | Supporto | Verifica |
| Dataset watercolor + caption | Supporto | Responsabile | Controllo qualità |
| Training LoRA | Supporto | Responsabile | Config e test |
| Integrazione LoRA in generazione | Responsabile | Supporto | Smoke test |
| Evaluation / analisi | Implementazione metriche | Analisi training | Interpretazione e report |
| Demo | Implementazione | Integrazione checkpoint | Test finale |

La divisione è indicativa: entrambi devono comprendere pipeline, training e metodologia per poter discutere il progetto all'esame.

## 6. Cosa fare nella **prossima sessione di lavoro**

1. Aprire il repository e verificare che **baseline e benchmark siano riproducibili** (non modificare la baseline congelata).
2. Creare `data/style_watercolor/train` e `data/style_watercolor/validation`, oppure definire i corrispondenti percorsi Kaggle.
3. Selezionare un primo insieme **piccolo di dipinti watercolor**, controllare licenze, varietà e formati, poi scrivere caption contenuto + `wtrclrstyle`.
4. Creare `src/training/dataset.py` con un caricamento di test.
5. Preparare `src/training/train_lora.py` partendo dallo script Diffusers, **senza ancora lanciare quattro training**.
6. Eseguire uno smoke test con rank 8, salvare log e verificare caricamento del checkpoint.

**Regola di avanzamento:** passare al passo successivo soltanto quando quello corrente produce un artefatto verificabile (dataset pulito, batch corretto, checkpoint caricabile, CSV metriche, grafico).

## 7. Risorse di riferimento

- Diffusers LoRA training: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
- Stable Diffusion 1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Diffusers ControlNet: https://huggingface.co/docs/diffusers/using-diffusers/controlnet
- ConsisLoRA (related work, **non implementazione da copiare automaticamente**): https://consislora.github.io/

---

**Stato della roadmap:** non ancora eseguita. Aggiorna le checkbox dopo ogni sessione e annota in Git le decisioni su dataset, modello, conditioning e protocollo di valutazione.
