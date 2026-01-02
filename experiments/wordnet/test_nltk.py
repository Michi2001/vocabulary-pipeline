import nltk
from nltk.corpus import wordnet as wn

# Einmaliger Download der Datenbanken
nltk.download("wordnet")
nltk.download("omw-1.4")  # Erforderlich für Spanisch

if __name__ == "__main__":
    synsets = wn.synsets("zorro", lang="spa")

    if len(synsets) == 0:
        print("Keine Synsets für 'zorro' gefunden.")

    for syn in synsets:
        print(f"ID: {syn.name()}")
        print(f"Definition (en): {syn.definition()}")

        # Oberbegriffe (Hypernyme) finden, um das Thema zu bestimmen
        hypernyms = syn.hypernyms()
        print(f"Gehört zur Kategorie: {[h.name() for h in hypernyms]}")
        print("-" * 20)
