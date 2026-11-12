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

## Esplora i test

La pagina include una sezione per scegliere una foto di `our_ds`, vedere l'originale e confrontare i risultati già salvati. I filtri permettono di combinare esperimenti, stili e mappe; il selettore dei risultati permette di mostrarne uno, più di uno oppure tutti con **Mostra tutti i risultati filtrati**. Ogni immagine mostra esperimento, stile, mappa, seed e run. Cliccando una miniatura si apre l'anteprima ingrandita.

Nel notebook configura **`TEST_IMAGE_PATHS`** per le foto e **`TEST_RESULT_PATHS`** per i sette esperimenti:

```python
TEST_IMAGE_PATHS = {'our_ds': 'our_ds'}
TEST_RESULT_PATHS = {
    'baseline_seed_42': 'outputs/baseline_seed_42',
    'baseline_seed_123': 'outputs/baseline_seed_123',
    'baseline_seed_456': 'outputs/baseline_seed_456',
    'rank_4': 'outputs/rank_4',
    'rank_8': 'outputs/rank_8',
    'rank_16': 'outputs/rank_16',
    'rank_32': 'outputs/rank_32',
}
```

I percorsi relativi partono da `PROJECT_ROOT`; sono ammessi anche percorsi assoluti Kaggle o Drive. Per esempio, da `/kaggle/working/MultiStyleImages`, `../../input/datasets/<utente>/<dataset>/rank_4` indica una cartella nel dataset Kaggle. Le chiavi identificano gli esperimenti nel sito e i valori indicano le loro cartelle effettive. Puoi aggiungere altre voci senza modificare il sito.

Il lettore riconosce la struttura `outputs/<run>/samples/<immagine>/images/<stile>/<mappa>/<nome>_seed<seed>.png`. Puoi indicare l'intera esportazione come quella in `outputs/rank_4`, la sua directory `outputs`, oppure un singolo run o la cartella `samples`. Le directory delle immagini di test possono contenere sottocartelle. L'associazione mantiene il percorso relativo dell'immagine; usa il solo nome file come alternativa soltanto quando non è ambiguo.

Se manca `our_ds`, viene usata l'eventuale copia `input_original.png` salvata nel test. Le cartelle mancanti vengono segnalate nella pagina: puoi caricare solo parte dei dataset e continuare a usare il sito. I metadati con vecchi percorsi assoluti Kaggle non sono necessari: l'indice usa i file realmente presenti e non carica pesi né rigenera risultati. Per nuovi dataset o file, ricrea `test_gallery` e riavvia il sito con la cella di avvio.

Il notebook passa a Gradio `allowed_paths=test_gallery.allowed_paths`: sono autorizzati solo i file immagine indicizzati, non l'intera esportazione contenente codice e checkpoint. Mantieni questo parametro anche in un avvio personalizzato per servire i dataset esterni alla cartella del progetto.

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

I test della galleria creano un dataset temporaneo con i sette esperimenti e verificano filtri, selezioni multiple, cartelle mancanti, percorsi portabili e corrispondenza degli originali. Non caricano modelli e non misurano la qualità delle generazioni. L'inferenza reale deve essere verificata eseguendo il notebook sul runtime CUDA.

Riferimenti API: [Gradio Blocks e launch](https://www.gradio.app/docs/gradio/blocks), [coda Gradio](https://www.gradio.app/guides/queuing), [gestione della memoria Diffusers](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory).
