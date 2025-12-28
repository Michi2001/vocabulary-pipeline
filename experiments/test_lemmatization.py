import spacy

# Spanisches Modell laden
nlp = spacy.load("es_core_news_sm")

text = "Los estudiantes estaban estudiando y hablaron con sus profesores."
doc = nlp(text)

for token in doc:
    print(f"\n\n\n{token}\n\n\n")
    print(token.text, "→", token.lemma_)
