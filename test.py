import spacy
import crosslingual_coreference


text_file = open("matsutake.txt")
text = ""
for line in text_file:
    text += line
    
DEVICE = 0

#add coreference resolution model
coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})

print(coref(text)._.resolved_text)
