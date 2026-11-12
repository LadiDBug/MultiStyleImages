# Evaluation RQ1 su Kaggle — 104 foto comuni

Materiale preparato, **non eseguito su Kaggle**:

- [Notebook](RQ1_EVALUATION_COMMON104_KAGGLE.ipynb).
- [Bundle di codice e configurazioni](rq1_evaluation_code.zip), da allegare come dataset al notebook insieme ai risultati originali.

Il bundle contiene soltanto i moduli di evaluation, il protocollo delle metriche e la selezione congelata. Non contiene immagini, pesi, training o pipeline di generazione. Il notebook ne verifica l'hash prima di estrarlo: se si aggiorna il bundle, aggiornare anche l'hash atteso nel notebook dopo aver verificato le modifiche.

## Che cosa fa

1. Richiede Python >= 3.11 e una sessione Kaggle. I percorsi `BUNDLE_PATH` e `SOURCE_ROOT` vanno adattati ai dataset allegati.
2. Estrae il codice in una nuova directory sotto `/kaggle/working/`, installa le sole dipendenze di evaluation e registra ambiente e GPU disponibile. Non sostituisce PyTorch con un'altra build.
3. Risolve le revisioni Hugging Face dei tre valutatori in commit immutabili e congela il TOML per tutti i seed. Serve Internet; i testi CLIP-T restano invariati.
4. Verifica la matrice di **104 foto × 3 stili × 4 condizioni × 3 seed**, compresi i metadati e gli hash degli input. Controlla lo spazio e prepara tre copie separate da **1.248 output** ciascuna, lasciando intatti gli originali.
5. Esegue l'audit delle tre copie e ne controlla gli hash. Nessun modello viene usato in questa fase.
6. **Si ferma prima dell'inferenza.** Solo dopo il successivo OK dell'utente si cambia `APPROVA_INFERENZA = True` nella cella di approvazione.
7. Calcola CLIP-I, CLIP-T, DINO-I e mIoU semantica, un seed alla volta, sulla GPU Kaggle. Non genera nuove immagini e non carica SD, ControlNet o LoRA.
8. Verifica che siano presenti tutti i punteggi attesi, senza errori o valori non finiti, e prepara uno ZIP scaricabile.

Lo ZIP contiene i CSV individuali, i riepiloghi e i delta vs `none` **per seed**, audit, log, selezione, provenienza, protocollo congelato, ambiente e codice. Non include immagini né pesi. L'aggregazione finale fra seed e i confronti diretti fra ControlNet saranno un passaggio successivo.

## Come procedere

Importare il notebook in Kaggle, allegare la struttura completa di `result_rq1/` e lo ZIP del codice, poi impostare i due percorsi nella prima cella. Selezionare GPU e Internet nelle impostazioni. Eseguire le celle in ordine fino al termine dell'audit e fermarsi al blocco di approvazione. La preparazione usa CPU, disco e rete di Kaggle; nessuna GPU locale.

L'evaluator non ha resume all'interno di un seed. Se un tentativo si interrompe, conservarne i file e incrementare `ATTEMPT` soltanto per quel seed; gli altri tentativi conclusi e verificati vengono saltati. La cella di esportazione può salvare anche risultati parziali, marcandoli `PARTIAL`. Non ripetere la prima cella per riprendere una sessione ancora attiva: creerebbe una nuova cartella di lavoro. Dopo il reset della sessione Kaggle occorre ricostruire l'ambiente; non è implementato il recupero automatico da uno ZIP esportato.

Verifiche locali eseguite: sintassi di tutte le celle, corrispondenza del bundle ai sorgenti, blocco dell'inferenza di default e controllo dei risultati su CSV sintetici completi/incompleti e con valori NaN. **Non sono stati installati pacchetti, scaricati pesi o eseguiti valutatori per testare il notebook localmente.** Compatibilità dell'ambiente Kaggle, VRAM e durata saranno verificate nella sessione reale.

Piano e registro: [PIANO_ABLATION_STUDY_RQ1.md](../docs/RQ1/PIANO_ABLATION_STUDY_RQ1.md).
