# per preparare i dati al trianing

# legge i metadati: costruisce l'elenco delle coppie immagine-caption
# apre l'img: la converte in RGB 
# Prepara i pixel per normalizzarli nel tensore atteso dal VAE
# prepara la caption: ovvero usa il tokenizer per convertirla in identificatori numerici. 