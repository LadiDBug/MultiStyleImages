# Demo web MultiStyleImages

Apri `demo/MultiStyleImages_Demo.ipynb` in Colab o Jupyter con GPU CUDA ed esegui le celle in ordine. Il sito viene servito dal runtime del notebook: non serve un backend separato.

1. Rendi disponibile il repository completo nel runtime e imposta `PROJECT_ROOT` nella prima cella. In Colab puoi clonare il repository oppure montare Google Drive e indicare la cartella del progetto.
2. Installa le dipendenze con la cella dedicata (Python ≥ 3.11; PyTorch deve supportare la GPU del runtime).
3. Copia eventuali checkpoint LoRA nel runtime. Puoi usare i percorsi dei TOML oppure `LORA_PATHS` nel notebook. I file LoRA sono ignorati da Git e non vengono scaricati automaticamente.
4. Esegui il caricamento dei modelli e attendi `Modelli pronti`.
5. Avvia il sito e apri il link stampato. In Colab usa `SHARE = True`; in locale puoi impostarlo a `False`. `AUTH` permette di proteggere il link pubblico con username e password. Il kernel deve rimanere attivo.

## Funzioni

- Upload di una foto senza preprocessing fotografico: solo conversione RGB, senza crop o resize preliminari. La demo usa `use_original_image=True`, come il runner con `--no-preprocess`; gli adattamenti dimensionali interni sono lasciati a Diffusers.
- Mappa: Nessuna, Canny, Depth, Segmentation.
- Stile: GTA5, Pixel art, Watercolor, con i prompt esistenti di `configs/styles.toml`.
- Solo per Pixel art: LoRA opzionale e scelta del rank tra i checkpoint disponibili. Il rank identifica pesi già addestrati; non modifica il rank di una LoRA a inferenza.
- Risultato scaricabile in PNG, anteprima della mappa utilizzata e dettagli di generazione con seed.

## Modelli e memoria

Il notebook precarica un solo backbone SD 1.5, tre ControlNet, DPT/MiDaS, UperNet/ADE20K e le LoRA disponibili. Le pipeline condividono UNet, VAE e text encoder usando `from_pipe`. Nessun download di modelli avviene durante una generazione.

I modelli inattivi restano in RAM CPU. Durante una richiesta CUDA, la pipeline selezionata usa il model CPU offloading di Diffusers: i componenti vengono trasferiti sulla GPU quando servono. Al termine, anche in caso di errore, gli hook vengono rimossi e i modelli riportati sulla CPU, così le pipeline possono condividere i componenti senza interferenze. Il VAE tiling riduce il picco di memoria nelle fasi di codifica e decodifica; sulle immagini grandi può introdurre lievi variazioni tonali tra le aree elaborate. I preprocessori depth e segmentation lavorano su CPU. Sono consigliati almeno 16 GB di RAM e una GPU con circa 8–16 GB di VRAM per immagini di circa 512×512; i requisiti effettivi vanno verificati sul runtime e con i checkpoint usati. La demo usa le dimensioni della foto caricata: se la GPU esaurisce la memoria, carica una foto più piccola. `width` e `height` in `configs/base.toml` non determinano il formato della foto nella demo. I trasferimenti CPU/GPU e i preprocessori CPU incidono sui tempi.

Una coda e un lock serializzano le generazioni, dato che backbone e adattatori sono condivisi. La LoRA viene disabilitata dopo ogni richiesta e non viene applicata agli altri stili. Parametri, scheduler ed estrazione delle mappe riutilizzano il codice del progetto. Le mappe vengono estratte dalla foto RGB alle sue dimensioni originali. Il preprocessing fotografico in `src/preprocessing/img_preprocess.py` resta disponibile per gli altri flussi, ma non viene chiamato dalla demo.

Per aggiungere un rank, passa una voce `rank: percorso` in `LORA_PATHS` e riesegui il caricamento (su runtime con poca RAM, riavvia prima il kernel). Una directory deve contenere `pytorch_lora_weights.safetensors`; è ammesso anche il percorso diretto a un file `.safetensors`.

## Avvio senza notebook

Dalla radice del repository, in un ambiente CUDA:

```bash
python -m pip install -r demo/requirements.txt
python -m demo.app
# Con link pubblico temporaneo:
python -m demo.app --share
```

## Verifiche senza pesi o GPU

```bash
python -m unittest discover -s demo/tests -v
```

I test usano pipeline simulate per verificare instradamento, mappe e isolamento LoRA; non misurano qualità o compatibilità dei checkpoint. L'inferenza reale deve essere verificata eseguendo il notebook sul runtime CUDA.

Riferimenti API: [Gradio Blocks e launch](https://www.gradio.app/docs/gradio/blocks), [coda Gradio](https://www.gradio.app/guides/queuing), [condivisione dei componenti Diffusers](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/loading#reuse-a-pipeline).
