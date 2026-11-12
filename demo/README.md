# Demo web MultiStyleImages

Apri `demo/MultiStyleImages_Demo.ipynb` in Colab o Jupyter con GPU CUDA ed esegui le celle in ordine. Il sito viene servito dal runtime del notebook: non serve un backend separato.

1. Rendi disponibile il repository completo nel runtime e imposta `PROJECT_ROOT` nella prima cella. In Colab puoi clonare il repository oppure montare Google Drive e indicare la cartella del progetto.
2. Installa le dipendenze con la cella dedicata (Python ≥ 3.11; PyTorch deve supportare la GPU del runtime).
3. Copia eventuali checkpoint LoRA nel runtime. Puoi usare i percorsi dei TOML oppure `LORA_PATHS` nel notebook. I file LoRA sono ignorati da Git e non vengono scaricati automaticamente.
4. Inizializza la demo: questa cella verifica il runtime e rileva le LoRA senza caricare modelli.
5. Avvia il sito e apri il link stampato. In Colab usa `SHARE = True`; in locale puoi impostarlo a `False`. `AUTH` permette di proteggere il link pubblico con username e password. Il kernel deve rimanere attivo.

## Funzioni

- Upload di una foto senza preprocessing fotografico: solo conversione RGB, senza crop o resize preliminari. La demo usa `use_original_image=True`, come il runner con `--no-preprocess`; gli adattamenti dimensionali interni sono lasciati a Diffusers.
- Mappa: Nessuna, Canny, Depth, Segmentation.
- Stile: GTA5, Pixel art, Watercolor, con i prompt esistenti di `configs/styles.toml`.
- Solo per Pixel art: LoRA opzionale e scelta del rank tra i checkpoint disponibili. Il rank identifica pesi già addestrati; non modifica il rank di una LoRA a inferenza.
- Risultato scaricabile in PNG, anteprima della mappa utilizzata e dettagli di generazione con seed.

## Modelli e memoria

Il sito si avvia senza modelli in memoria. Ogni pressione di **Genera** esegue questo ciclo, con una sola richiesta alla volta:

1. Con Nessuna non viene estratta alcuna mappa; Canny non richiede modelli. Per Depth o Segmentation viene caricato soltanto il relativo estrattore su CPU e rilasciato subito dopo aver prodotto la mappa.
2. Viene caricata una sola pipeline SD 1.5, con il solo ControlNet scelto, oppure senza ControlNet per Nessuna.
3. Soltanto se Pixel art e LoRA sono attivi, viene caricato il checkpoint del rank selezionato. Le altre LoRA non vengono caricate né validate contro il backbone di questa richiesta.
4. Su CUDA si usa model CPU offloading. VAE slicing e tiling riducono i picchi di memoria; il tiling può introdurre lievi variazioni tonali sulle immagini grandi.
5. Al termine, anche dopo un errore, vengono rimossi gli hook, eliminati i riferimenti ai modelli, eseguita la garbage collection e svuotata la cache CUDA inutilizzata. Anche i frame dei traceback vengono ripuliti dai riferimenti che potrebbero trattenere tensori.

I pesi restano nella cache su disco di Hugging Face: ogni richiesta li ricarica in memoria, ma li scarica soltanto se non sono già in cache. Il primo utilizzo di una modalità può quindi richiedere un download; Kaggle deve avere accesso a Internet, oppure i checkpoint devono essere già in cache. La coda serializza caricamento, generazione e rilascio.

La demo conserva le dimensioni della foto caricata, con sola conversione RGB. La memoria richiesta durante l'inferenza dipende ancora dalla risoluzione e dalla modalità: il caricamento su richiesta non garantisce che qualsiasi immagine rientri nella VRAM. Se necessario, carica una foto più piccola. `width` e `height` in `configs/base.toml` non determinano il formato della foto nella demo. Il preprocessing fotografico in `src/preprocessing/img_preprocess.py` resta disponibile per gli altri flussi e non viene chiamato dalla demo.

Per aggiungere un rank, passa una voce `rank: percorso` in `LORA_PATHS` e reinizializza runtime e sito. Una directory deve contenere `pytorch_lora_weights.safetensors`; è ammesso anche un percorso diretto a un file `.safetensors`.

Dopo l'aggiornamento dalla versione con precaricamento, **riavvia la sessione Kaggle** per rimuovere i vecchi modelli e importare il nuovo codice, poi riesegui le celle.

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

Riferimenti API: [Gradio Blocks e launch](https://www.gradio.app/docs/gradio/blocks), [coda Gradio](https://www.gradio.app/guides/queuing), [gestione della memoria Diffusers](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory).
