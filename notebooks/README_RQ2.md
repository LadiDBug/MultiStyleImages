# Evaluation RQ2 su Kaggle

Materiale preparato il 24 novembre 2026:

- `RQ2_EVALUATION_RANKS_KAGGLE.ipynb`: notebook con i quattro percorsi del dataset `dianapamfile/lorarankresult` già impostati.
- `rq2_evaluation_code.zip`: solo codice di evaluation e protocollo; nessuna immagine o peso.
- `build_rq2_notebook.py`: rigenera notebook e bundle dai sorgenti, aggiornando gli hash attesi.

## Avvio

1. Importare il notebook `.ipynb` su Kaggle.
2. Aggiungere il dataset dei risultati e un secondo dataset con `rq2_evaluation_code.zip`. Il codice supporta anche il bundle estratto automaticamente da Kaggle: riconosce `rq2_bundle_manifest.json` e verifica tutti i file.
3. Attivare GPU e Internet. Il notebook conserva la build PyTorch di Kaggle e installa soltanto le dipendenze di evaluation.
4. Eseguire le celle di configurazione, codice e audit. Le sottocartelle vengono esplorate automaticamente: sono supportati anche run distinti per condizione. ZIP di risultati ancora compressi non vengono estratti dal notebook: serve la struttura di cartelle mostrata dal dataset.
5. Controllare il riepilogo dell'audit. Foto, seed, condizioni, hash input, geometria, prompt e parametri devono coincidere nei quattro rank. Duplicati, file mancanti o metadati incompatibili fermano il notebook; nessuna intersezione automatica delle sole foto valide.
6. Congelare il protocollo e avviare la cella GPU. Facoltativamente impostare `RQ1_PROTOCOL` al TOML congelato RQ1 allegato, per riutilizzarne esattamente le revisioni dei valutatori. Altrimenti vengono risolte una volta all'avvio RQ2.
7. Eseguire aggregazione ed esportazione. Scaricare lo ZIP `COMPLETE` da `/kaggle/working/`.

Il notebook non è stato eseguito sul dataset remoto o sull'ambiente reale Kaggle. Sono state verificate localmente la sintassi delle celle, la coerenza del bundle e le funzioni su dati sintetici, senza scaricare modelli o eseguire inferenze.

I sei test RQ2 offline sono superati. Nella suite esistente di evaluation sono passati 11 controlli; tre controlli non hanno potuto concludersi perché il criterio di controllo applicazioni di Windows blocca il caricamento di una DLL PyTorch (`torch._C`). Questo impedisce la verifica locale dei valutatori basati su modelli; la compatibilità effettiva verrà verificata su Kaggle.

## Risultati e interpretazione

Per ciascun rank: metriche per output, riepiloghi, delta ControlNet rispetto a `none`, configurazioni effettive dei valutatori e provenienza. L'aggregazione produce:

- `all_rank_metrics.csv`: tutti gli output e i quattro punteggi.
- `summary_by_rank.csv`: medie e intervalli per rank, categoria e condizione.
- `paired_between_ranks.csv`: tutti i sei confronti, delta rank alto meno rank basso, per categoria/condizione/metrica.

Prima si mediano i seed per foto, poi le foto con uguale peso. Gli intervalli al 95% usano bootstrap sulle foto (2000 campioni, seed 2026): sono descrittivi, senza correzione per confronti multipli. Con una sola foto non viene stimato un intervallo. Non misurano variabilità fra repliche di training.

Non è presente una baseline senza LoRA: `none` significa senza ControlNet e mantiene attivo l'adattatore del rank. CLIP-T non è una percentuale di fedeltà allo stile. L'audit registra hash adattatori e firme training, ma non sostituisce la verifica separata dei dati e degli iperparametri dei training.

## Interruzioni

I rank sono valutati in processi separati. Nella stessa sessione, la cella GPU salta i tentativi completati solo dopo averne verificato matrice, punteggi finiti, hash e provenienza. Se un tentativo fallisce, incrementare soltanto il suo valore in `ATTEMPTS` e rieseguire la cella GPU; i tentativi precedenti restano salvati. Non c'è resume all'interno del rank.

Se si riesegue la prima cella nella stessa sessione, impostare `RESUME_WORK` alla cartella stampata. Dopo un audit fallito non viene salvato un piano completo: correggere i percorsi e creare una nuova cartella WORK. Dopo reset Kaggle occorre ricostruire l'ambiente; non è implementato il recupero automatico da uno ZIP esportato.

L'ultima cella può esportare anche dopo un errore: l'archivio viene marcato `PARTIAL`. Include audit, piano, log, codice e CSV già presenti, senza immagini o pesi. Scaricarlo prima di chiudere la sessione.
