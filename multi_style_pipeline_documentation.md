# Real Image → Multi-Style Translation (GTA V, Mario Kart, Cyberpunk, Ghibli) — Pipeline del Progetto

## 1. Obiettivo

Il progetto sviluppa un sistema generativo di **image-to-image translation multi-stile** che trasforma un'immagine reale di una scena urbana o paesaggistica in diversi stili visivi (GTA V, Mario Kart, Cyberpunk 2077, Studio Ghibli, ecc.), preservando rigorosamente la struttura e la composizione spaziale della scena originale.

### Input
- Immagine reale (foto di una scena urbana, veicolo, paesaggio)
- Selezione dello stile desiderato (es. GTA V, Cyberpunk, Mario Kart)

### Output
- La stessa scena trasformata nello stile visivo selezionato

Il sistema si colloca nell'ambito di:
- Image-to-image translation (unpaired)
- Generative modeling (Diffusion Models)
- Representation / Semantic conditioning
- Modularity & Style Hot-Swapping

---

# 2. Architettura Generale del Sistema

Per supportare più stili senza addestrare o duplicare modelli da miliardi di parametri per ogni dominio, l'architettura si basa su una **struttura modulare ad aggancio dinamico (Style Hot-Swapping)**.

```text
                     REAL IMAGE (Input)
                             │
                             ▼
                    ControlNet (Segmentation) ──┐
                             │                  │ (Struttura semantica invariata)
                             ▼                  │
┌─────────────────────────────────────────────┐ │
│         STABLE DIFFUSION BASE MODEL         │ │
└─────────────────────────────────────────────┘ │
                             │                  │
           ┌─────────────────┼─────────────────┐│
           ▼                 ▼                 ▼│
    [LoRA: GTA V]   [LoRA: Mario Kart]  [LoRA: Cyberpunk] ... (Hot-Swapping)
           │                 │                 ││
           └─────────────────┼─────────────────┘│
                             │                  │
                             ▼                  │
                   GENERATED MULTI-STYLE IMAGE ◄┘
```

### Componenti Principali:
1. **Model Backbone (Frozen):** Stable Diffusion 1.5 come base generativa.
2. **Structural Conditioning (Frozen/Fine-tuned):** ControlNet guidato da segmentation map semantica per la preservazione della composizione.
3. **Style Adapters (Dynamic LoRAs):** Micro-moduli LoRA (Low-Rank Adaptation) caricati ed intercambiati al volo in memoria a seconda dello stile scelto dall'utente.

---

# 3. Dataset e Stili Target

Trattandosi di un approccio **unpaired** (non esistono coppie di immagini "foto reale ↔ screenshot del gioco/anime per la stessa identica scena"), per ogni stile viene creato/utilizzato un mini-dataset dedicato.

| Stile | Fonte Dataset | Caratteristiche Visive Dominanti | Prompt Trigger |
| :--- | :--- | :--- | :--- |
| **GTA V** | Dataset GTA5 (Richter et al.) / Game screenshots | Colori saturi, illuminazione californiana, estetica video-ludica semi-realistica | `"a GTA V style urban street scene"` |
| **Mario Kart** | Frame da Mario Kart 8 / Simulatori | Palette accesa, forme arrotondate, texture pulite, estetica 3D cartoon | `"a Mario Kart style track, vibrant colors, 3d render"` |
| **Cyberpunk 2077**| Screenshot di Night City / Concept art | Luci al neon (magenta/ciano), asfalto bagnato con riflessi, pioggia, atmosfera sci-fi | `"a Cyberpunk 2077 Night City style scene, neon lights, rainy street"` |
| **Studio Ghibli** | Frame da anime (es. *Your Name*, *La città incantata*) | Dipinto a mano, colori pastello, cieli dettagliati con nuvole soffici, estetica 2D | `"Studio Ghibli style, hand-drawn anime, soft lighting, detailed sky"` |

---

# 4. Pipeline Tecnica di Addestramento e Inferenza

## 4.1 Preprocessing e Segmentazione Semantica

L'immagine reale di input viene ridimensionata (es. $512 \times 512$) e passata a una rete di segmentazione (es. UniFormer / SegFormer) per produrre la **segmentation map**.

```text
              Semantic Map

       ┌──────────────────────┐
       │        SKY           │
       │                      │
       │ BUILDING    BUILDING │
       │                      │
       │   CAR     CAR        │
       │──────────────────────│
       │        ROAD          │
       └──────────────────────┘
```

Questo isola il **contenuto** (posizione di auto, strada, palazzi, vegetazione) dallo **stile** (colori, texture, illuminazione).

## 4.2 Training delle Singole LoRA (Style Adaptors)

Per ciascuno stile supportato:
1. Si raccolgono 20-50 immagini rappresentative del dominio target.
2. Si associano le caption descrittive contenenti la *trigger word* specifica.
3. Si addestramo solo le matrici ad alto rango (LoRA) della UNet con un rank $r = 8, 16 \text{ o } 32$.
4. **Risultato:** File `.safetensors` leggero (~10 MB - 100 MB) per ogni stile.

## 4.3 Formulation della Loss di Diffusion

Durante il denoiser, la loss di addestramento è definita da:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c_{\text{text}}, c_{\text{seg}}) \|^2 \right]$$

Dove:
- $x_t$: Immagine al passo temporale $t$ con rumore applicato.
- $c_{\text{text}}$: Prompt di condizionamento testuale associato allo stile.
- $c_{\text{seg}}$: Mappa di segmentazione semantica passata via ControlNet.

---

# 5. Modulo Avanzato: LoRA Blending & Weight Tuning

Il sistema permette non solo di cambiare stile, ma anche di **fondere due stili contemporaneamente** combinando linearmente i pesi delle matrici LoRA durante la Cross-Attention:

$$W_{\text{blended}} = W_0 + \alpha \Delta W_{\text{StyleA}} + \beta \Delta W_{\text{StyleB}}$$

Dove $\alpha$ e $\beta$ sono gli **Style Weights** (es. $0.6 \text{ Cyberpunk} + 0.4 \text{ Mario Kart}$).

---

# 6. Esperimenti e Valutazione Quantitativa

Per validare la bontà del sistema, la pipeline prevede 3 metriche principali:

1. **FID (Fréchet Inception Distance):** Misura la qualità visiva e la vicinanza della distribuzione delle immagini generate rispetto al dataset reale dello stile selezionato.
2. **mIoU (mean Intersection over Union):** Si passa l'immagine generata a una rete di segmentazione esterna e si confronta la mappa risultante con la mappa semantica dell'input originale per verificare la preservazione della struttura.
3. **LPIPS (Learned Perceptual Image Patch Similarity):** Valuta quanto la composizione e i contorni dell'immagine d'origine siano stati conservati dopo il trasferimento di stile.

---

# 7. Struttura del Progetto nel Repository

```text
multi-style-transfer/
├── loras/                  # File .safetensors delle LoRA (gta5.safetensors, cyberpunk.safetensors, ecc.)
├── src/                    # Codice sorgente modulare
│   ├── __init__.py
│   ├── segmentation.py     # Estrazione e gestione segmentation map per ControlNet
│   ├── lora_manager.py     # Caricamento, caching e blending dinamico dei pesi LoRA
│   └── pipeline.py         # Ingegneria della pipeline Diffusion + ControlNet
├── scripts/                # Script di utilità
│   ├── train_lora.sh       # Script per l'addestramento di una nuova LoRA stile
│   └── evaluate.py         # Calcolo metriche (FID, mIoU, LPIPS)
├── app.py                  # Interfaccia grafica utente (Gradio Demo)
├── requirements.txt        # Dipendenze Python
└── README.md               # Documentazione e guida rapida
```

---

# 8. Interfaccia Utente (Gradio Demo)

L'interfaccia utente fornisce i seguenti controlli:

- **Input Image Upload:** Selezione foto da trasformare.
- **Style Selector Dropdown:** Menu a tendina per cambiare stile (GTA V, Mario Kart, Cyberpunk, Ghibli, ecc.).
- **Style Strength (Slider):** Impostazione del peso della LoRA ($\alpha$).
- **ControlNet Conditioning Scale (Slider):** Grado di aderenza alla geometria della foto originale.
- **Toggle Box:** `[x] Show Semantic Segmentation Map`.
- **Output View:** Visualizzazione affiancata di (Immagine Reale | Mappa Semantica | Risultato Stilizzato).