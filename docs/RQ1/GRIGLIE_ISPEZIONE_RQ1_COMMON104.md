# RQ1 — Griglie per l'ispezione delle immagini esistenti

**Preparazione completata il 12 novembre 2026, dopo OK dell'utente.** Sono pronte 30 griglie comparative, con originale e quattro condizioni, più una galleria HTML con moduli di risposta, salvataggio locale ed esportazione/importazione. Il codice non ha assegnato giudizi percettivi: le risposte saranno compilate dai revisori.

Aprire la [galleria dei confronti](../../result_evaluation_rq1/visual_review_common104_v1/index.html) nel browser. Il lavoro è locale: nessun nuovo notebook Kaggle, modello o GPU è necessario. Se VS Code mostra il sorgente HTML, aprire `index.html` con il browser tramite Esplora file.

## 1. Che cosa contiene ogni griglia

Le colonne sono sempre **originale → none → canny → depth → segmentation**. Fotografia, stile e seed sono gli stessi in ogni riga; il seed visualizzato è **42**, fissato prima di aprire gli output. Non abbiamo scelto per ogni metodo il seed visivamente più favorevole.

Le quattro metriche sotto ogni output sono quelle di **quell'immagine con seed 42**. I casi diagnostici sono stati selezionati usando i delta medi sui seed 42, 123 e 456: il singolo seed mostrato può quindi avere un comportamento diverso dalla media usata per scegliere il caso. Non confondere questi due livelli di analisi.

Gli originali non vengono ritagliati, alterati o sovrascritti. Le griglie sono figure di confronto ridotte per la lettura; le immagini sono mostrate intere con interpolazione nearest-neighbor. Per dettagli fini, pixel, bordi o texture, usare i link ai **PNG sorgenti a piena risoluzione** nella galleria. L'anteprima non è una nuova immagine generata dal modello.

## 2. Campione casuale: sei foto comuni ai tre stili

Campionamento: `random.Random(2026).sample(sorted(image_ids), 6)` sulla selezione congelata di 104 input. La lista è stata salvata prima della composizione delle griglie e non è stata cambiata dopo l'apertura delle immagini.

| Foto | Griglia GTA V | Griglia Pixel Art | Griglia Watercolor |
|---|---|---|---|
| `0131x2.png` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0131x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0131x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0131x2_seed42.png) |
| `0433x2.png` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0433x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0433x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0433x2_seed42.png) |
| `4klsdb_0303.jpg` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_4klsdb_0303_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_4klsdb_0303_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_4klsdb_0303_seed42.png) |
| `4klsdb_0334.jpg` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_4klsdb_0334_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_4klsdb_0334_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_4klsdb_0334_seed42.png) |
| `flickr2k_part2_0173.png` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_flickr2k_part2_0173_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_flickr2k_part2_0173_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_flickr2k_part2_0173_seed42.png) |
| `0084x2.png` | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0084x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0084x2_seed42.png) | [Apri](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0084x2_seed42.png) |

Sono **18 griglie casuali = 6 foto × 3 stili**. Questo piccolo campione serve a iniziare l'ispezione, non a misurare con precisione la frequenza di difetti o la preferenza umana sull'intero benchmark. Non è stratificato per scena; le categorie di A06 sono ancora da definire.

## 3. Casi diagnostici: domande specifiche poste dai numeri

I casi sotto sono scelti in base ai punteggi e vanno mantenuti separati dal campione casuale. Servono a spiegare un fenomeno, non a dimostrare che quel fenomeno sia frequente. Ogni griglia mostra comunque tutte le quattro condizioni, anche quando il criterio riguarda una sola coppia.

| Stile | Criterio fissato | Foto selezionata / griglia |
|---|---|---|
| GTA V | Massimo delta DINO-I per Canny − none | [0731x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0731x2_seed42.png) |
| GTA V | Delta CLIP-T più negativo per Canny − none | [lsdir_0404.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_lsdir_0404_seed42.png) |
| GTA V | DINO-I positivo ma mIoU negativa per Canny − none; massimo delta DINO-I fra i casi ammissibili | [0262x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0262x2_seed42.png) |
| GTA V | Massimo guadagno mIoU di segmentation rispetto a depth | [0236x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/gta5_0236x2_seed42.png) |
| Pixel Art | Massimo delta DINO-I per Canny − none | [flickr2k_part2_0383.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_flickr2k_part2_0383_seed42.png) |
| Pixel Art | Delta CLIP-T più negativo per Canny − none | [0132x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0132x2_seed42.png) |
| Pixel Art | DINO-I positivo ma mIoU negativa per Canny − none; massimo delta DINO-I fra i casi ammissibili | [0691x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0691x2_seed42.png) |
| Pixel Art | Depth guadagna DINO-I e perde CLIP-T rispetto a segmentation; massimo delta DINO-I fra i casi ammissibili | [0209x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/pixel_art_0209x2_seed42.png) |
| Watercolor | Massimo delta DINO-I per Canny − none | [0235x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0235x2_seed42.png) |
| Watercolor | Delta CLIP-T più negativo per Canny − none | [0132x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0132x2_seed42.png) |
| Watercolor | DINO-I positivo ma mIoU negativa per Canny − none; massimo delta DINO-I fra i casi ammissibili | [0211x2.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_0211x2_seed42.png) |
| Watercolor | Canny guadagna DINO-I e mIoU vs depth; minimo valore assoluto del delta CLIP-T fra i casi ammissibili | [flickr2k_part2_0003.png](../../result_evaluation_rq1/visual_review_common104_v1/grids/watercolor_flickr2k_part2_0003_seed42.png) |

Sono **12 griglie diagnostiche**, quattro per stile. Nei pareggi si usa l'ordine lessicografico di `image_id`. Due criteri possono selezionare la stessa fotografia: non si cambia caso per rendere gli esempi più vari. Qui `0132x2.png` ricorre in due stili. Complessivamente le 30 griglie riguardano **17 fotografie distinte**, non 30 foto indipendenti.

## 4. Come leggere e annotare le griglie

**Si compila tutto nell'HTML, una griglia alla volta. Non serve modificare il CSV.**

1. Aprire [index.html](../../result_evaluation_rq1/visual_review_common104_v1/index.html) nel browser e inserire nome o pseudonimo; premere **Apri le mie risposte**.
2. Se lavorate in due, scegliere **Parte A** oppure **Parte B**: numeri di griglia dispari/pari, 15 griglie e cinque per stile a persona. Potete anche scegliere tutte le griglie o compilare entrambi la stessa: i nomi mantengono distinti i giudizi.
3. Per ogni griglia rispondere alle quattro domande testuali elencate sotto. Le risposte si salvano automaticamente nel browser. È possibile lasciare una bozza e riprendere più tardi.
4. Premere **Segna questa griglia completata** dopo aver risposto a tutte e quattro le domande. Sono ammesse parità, dubbi e assenza di difetti evidenti: non occorre forzare un vincitore. Una modifica successiva riapre la scheda come bozza.
5. A fine sessione premere **Esporta risposte JSON** e conservare il file scaricato. Questo è il backup da condividere o riaprire su un altro computer. Controllare che il download sia effettivamente presente.
6. Per riunire i lavori premere **Importa risposte JSON**, anche con più file selezionati. Risposte di revisori diversi restano separate; versioni differenti dello stesso revisore/griglia vengono conservate e richiedono una scelta esplicita. Entrambe le versioni rimangono nel JSON.

### Le quattro domande standard

| Domanda | Che cosa scrivere |
|---|---|
| **Contenuto e struttura: cosa rimane e cosa cambia?** | Confrontare i metodi con l'originale: oggetti, sagome, posizioni, prospettiva; elementi aggiunti o rimossi. Citare i nomi dei metodi e dettagli concreti. |
| **Stile richiesto: in quali output lo riconosci?** | Dire quali segni visivi dello stile si vedono e in quali output. È valida anche una risposta che dichiari uno stile poco riconoscibile. Gli aiuti cambiano per GTA V, Pixel Art e Watercolor. |
| **Artefatti: quali difetti o dettagli inventati noti?** | Deformazioni, duplicazioni, texture o elementi incoerenti, specificando il metodo. Se non si notano difetti evidenti alla risoluzione esaminata, scriverlo. |
| **Compromesso: quale output sceglieresti e perché?** | Considerare insieme contenuto, stile e difetti; motivare una scelta, una parità o nessuna preferenza. |

Non sono richiesti voti numerici. Il confronto riguarda tutti e quattro gli output della griglia, con **una risposta per domanda e per revisore**, anziché quattro schede di metodo separate.

### Salvataggio e collaborazione

Il salvataggio automatico usa `localStorage`: rimane nello stesso browser/profilo/computer e può dipendere dal percorso del file locale. **Non modifica l'HTML sul disco, non scrive il vecchio CSV e non sincronizza due computer.** Per cambiare computer, browser o percorso, importare il JSON esportato. Se il browser impedisce il salvataggio locale, viene mostrato un avviso e si può comunque esportare il contenuto tenuto in memoria prima di chiudere la pagina. Cancellare i dati del browser può eliminare le bozze: esportare regolarmente.

Per condividere la galleria usare [galleria_rq1_da_condividere.zip](../../result_evaluation_rq1/visual_review_common104_v1/galleria_rq1_da_condividere.zip). Contiene HTML, griglie e i 137 PNG sorgenti necessari ai link a piena risoluzione. Il collega deve **estrarre tutto lo ZIP** e aprire `index.html`. Il pacchetto contiene la galleria, non le vostre risposte: quelle vanno esportate in JSON e passate separatamente. Il medesimo identificatore della galleria consente di riunire i lavori fra la versione locale e quella portabile.

**Esporta riepilogo CSV** produce una riga per griglia/revisore con le quattro risposte attive. Per il backup e la riunione usare il JSON, che conserva anche le versioni alternative. Il vecchio [annotations.csv](../../result_evaluation_rq1/visual_review_common104_v1/annotations.csv), predisposto prima del modulo, rimane intatto e non è più il punto di raccolta richiesto per questo percorso.

Per interpretare le risposte, mantenere separati questi aspetti:

| Aspetto | Domande |
|---|---|
| Contenuto/layout | Oggetti, silhouette, posizioni e prospettiva sono riconoscibili? Sono comparsi elementi nuovi o spariti dettagli importanti? |
| Stile | Si riconoscono le caratteristiche richieste, oltre a un semplice cambio di colore o texture? |
| Artefatti/qualità | Ci sono deformazioni, duplicazioni, superfici incoerenti o difetti locali? |

Per Pixel Art controllare anche palette, blocchi, bordi e pixel a piena risoluzione. Per Watercolor osservare lavature, granulosità e passaggi dei bordi. Per GTA V distinguere una resa da videogioco da una foto quasi invariata o da un'illustrazione generica.

I nomi dei metodi e i punteggi sono visibili: queste griglie sono diagnostiche e **non costituiscono H01, la valutazione umana cieca**. Le osservazioni dell'assistente o dell'autore dovranno essere dichiarate come tali; non sono giudizi di valutatori indipendenti.

## 5. Artefatti e verifiche

Directory: [`result_evaluation_rq1/visual_review_common104_v1/`](../../result_evaluation_rq1/visual_review_common104_v1/).

| File | Contenuto |
|---|---|
| `index.html` | Galleria con navigazione di una griglia alla volta, quattro domande, revisori, salvataggio locale, import/export e conflitti |
| `grids/*.png` | 30 figure comparative esportabili, 3000 × 870 pixel |
| `visual_selection.json` | Lista fissata, criteri, seed, percorsi e hash sorgenti/artefatti |
| `displayed_metrics.csv` | 120 righe con i punteggi effettivi del seed 42 mostrati nelle figure |
| `annotations.csv` | Vecchio schema per metodo, conservato senza modifiche; il modulo esporta risposte per griglia/revisore |
| `index_before_forms.html` | Copia della galleria precedente ai moduli |
| `galleria_rq1_da_condividere.zip` | Pacchetto portabile con HTML, 30 griglie e 137 immagini sorgenti; nessuna risposta personale |

La galleria usa file locali, senza CDN, servizi remoti o caricamento di dati online. Per aprire i sorgenti tramite i link, conservare la posizione relativa di `result_rq1/` rispetto alla cartella della galleria. I PNG delle griglie sono autonomi e possono essere condivisi singolarmente.

Verificati: selezione congelata, presenza e configurazione dei file sorgenti, corrispondenza degli hash delle immagini mostrate con quelle valutate, criteri del campionamento su casi sintetici, numero di griglie e schede, link locali e leggibilità dei tre stili su una fotografia di controllo. Questo controllo visivo della composizione non equivale a una revisione percettiva completa delle 120 immagini.

Script delle griglie: [`src/evaluation/visual_review_rq1.py`](../../src/evaluation/visual_review_rq1.py). Usa Pillow e Matplotlib, senza Torch, e crea anche il modulo HTML. Per rigenerare in una directory nuova:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.visual_review_rq1 `
  --output-dir result_evaluation_rq1/visual_review_common104_replica
```

Per aggiornare soltanto il modulo di una galleria esistente e creare lo ZIP portabile, senza ridisegnare griglie o modificare risposte:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.review_gallery `
  --gallery-dir result_evaluation_rq1/visual_review_common104_v1 --share-zip
```

Test aggiunti: persistenza e ripristino dell'interfaccia in un DOM isolato, import/export, separazione dei revisori, scelta esplicita dei conflitti e conservazione delle versioni, validazione atomica dei file incompatibili, fallback senza salvataggio locale e integrità del pacchetto portabile. I test non hanno aperto il browser personale o caricato dati online.

## 6. Stato e prossimo passo

- [x] Preparare il campione casuale e i casi diagnostici senza scegliere il seed migliore.
- [x] Creare griglie, galleria e schede vuote con provenienza verificabile.
- [x] Predisporre quattro domande standard per tutte le 30 griglie, salvataggio e divisione del lavoro richiesta dall'utente.
- [ ] Compilare e raccogliere le risposte qualitative dei revisori tramite JSON.
- [ ] Collegare i casi osservati ai risultati quantitativi, senza generalizzare dai soli estremi.
- [ ] Definire A06 per scene e H01 per giudizi umani ciechi, se necessari alle conclusioni.

**Prossimo passo concordato:** i revisori compilano tutte le 30 griglie nell'HTML, eventualmente dividendosi Parte A e Parte B; poi passano i JSON per riunire e analizzare le risposte. Nessun giudizio viene inventato o assegnato automaticamente, e nessuna nuova generazione viene avviata.
