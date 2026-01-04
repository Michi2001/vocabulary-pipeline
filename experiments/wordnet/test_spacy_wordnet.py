"""Test script for spacy-wordnet integration with Spanish."""

import spacy

# Installation falls nötig:
# pip install spacy-wordnet
# python -m spacy download es_core_news_md
from spacy_wordnet.wordnet_annotator import WordnetAnnotator


def test_spacy_wordnet():
    """Test spacy-wordnet with Spanish words."""
    # Lade spanisches Modell (md oder lg für Word Vectors)
    nlp = spacy.load("es_core_news_lg")

    # Füge WordNet-Annotator zur Pipeline hinzu
    # Sprache wird automatisch vom nlp-Objekt übernommen
    # Spanische Modelle nutzen 'morphologizer' statt 'tagger'
    if "spacy_wordnet" not in nlp.pipe_names:
        nlp.add_pipe("spacy_wordnet", after="lemmatizer")

    # Testwörter
    test_words = ["medio", "perro", "comer", "grande", "laberinto", "zorro"]

    for word in test_words:
        print(f"\n{'='*50}")
        print(f"Wort: {word}")
        print("=" * 50)

        doc = nlp(word)
        token = doc[0]

        # Zugriff auf WordNet-Daten über Token
        wordnet = token._.wordnet

        # Alle Synsets für dieses Token
        synsets = wordnet.synsets()

        if not synsets:
            print("Keine Synsets gefunden")
            continue

        print(f"Anzahl Synsets: {len(synsets)}")

        for syn in synsets[:5]:  # Nur erste 5 anzeigen
            print(f"\n  ID: {syn.name()}")
            print(f"  Definition (en): {syn.definition()}")
            print(f"  POS: {syn.pos()}")

            # Lemmas in verschiedenen Sprachen
            spa_lemmas = syn.lemma_names("spa")
            if spa_lemmas:
                print(f"  Spanische Lemmas: {spa_lemmas}")

            # Hypernyme (Oberbegriffe)
            hypernyms = syn.hypernyms()
            if hypernyms:
                print(f"  Hypernyme: {[h.name() for h in hypernyms[:3]]}")

            # Hyponyme (Unterbegriffe)
            hyponyms = syn.hyponyms()
            if hyponyms:
                print(f"  Hyponyme: {[h.name() for h in hyponyms[:3]]}")

            print("-" * 40)


def test_domains():
    """Test WordNet domains for topic categorization."""
    nlp = spacy.load("es_core_news_lg")

    if "spacy_wordnet" not in nlp.pipe_names:
        nlp.add_pipe("spacy_wordnet", after="lemmatizer")

    print("\n" + "=" * 50)
    print("Domain-Test (Themengebiete)")
    print("=" * 50)

    # Wörter aus verschiedenen Domänen
    test_sentences = [
        "El médico examina al paciente",  # Medizin
        "El programador escribe código",  # IT
        "El cocinero prepara la comida",  # Kochen
    ]

    for sentence in test_sentences:
        print(f"\nSatz: {sentence}")
        doc = nlp(sentence)

        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                wordnet = token._.wordnet
                domains = wordnet.wordnet_domains()

                if domains:
                    print(f"  {token.text} ({token.pos_}): {list(domains)[:5]}")


def test_similarity_via_wordnet():
    """Test semantic similarity using WordNet path similarity."""
    from nltk.corpus import wordnet as wn

    print("\n" + "=" * 50)
    print("Ähnlichkeitstest via WordNet Path Similarity")
    print("=" * 50)

    # Wortpaare zum Vergleichen
    word_pairs = [
        ("perro", "gato"),  # Beide Tiere
        ("perro", "coche"),  # Tier vs Objekt
        ("comer", "beber"),  # Beide Verben der Nahrungsaufnahme
        ("zorro", "rojo"),
    ]

    for word1, word2 in word_pairs:
        synsets1 = wn.synsets(word1, lang="spa")
        synsets2 = wn.synsets(word2, lang="spa")

        if synsets1 and synsets2:
            # Nimm jeweils erstes Synset
            syn1, syn2 = synsets1[0], synsets2[0]
            similarity = syn1.path_similarity(syn2)
            sim_str = f"{similarity:.3f}" if similarity else "N/A"
            print(f"  {word1} <-> {word2}: {sim_str}")


if __name__ == "__main__":
    print("SpaCy-WordNet Integration Test")
    print("=" * 50)

    test_spacy_wordnet()
    test_domains()
    test_similarity_via_wordnet()
