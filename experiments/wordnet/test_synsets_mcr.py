import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

# 1. Vorbereitung: NLTK Ressourcen sicherstellen
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def load_mcr_mapping(file_path):
    """
    Lädt die MCR variant-Datei in einen Pandas DataFrame.
    Struktur basierend auf dem hochgeladenen Screenshot.
    """
    # Spaltennamen basierend auf der MCR-Struktur
    columns = ["lemma", "rank", "synset_id", "pos", "score", "none_col", "dash_col"]

    # Einlesen der Tab-separierten Datei
    df = pd.read_csv(file_path, sep="\t", names=columns, header=None, encoding="utf-8")
    return df


def get_word_senses(lemma, df):
    """
    Sucht alle Bedeutungen (Senses) für ein Lemma und löst sie über WordNet auf.
    """
    results = []

    # Filtere alle Einträge für das gesuchte Wort
    word_entries = df[df["lemma"] == lemma]

    if word_entries.empty:
        return None

    for _, row in word_entries.iterrows():
        try:
            # Extraktion: 'spa-30-02118333-n' -> offset: 02118333, pos: n
            # Der Teil '30' im String bestätigt WordNet 3.0 Kompatibilität
            parts = row["synset_id"].split("-")
            offset = int(parts[2])
            pos = parts[3]

            # Hole das Synset direkt über den Offset aus WordNet
            synset = wn.synset_from_pos_and_offset(pos, offset)

            # Alle Hypernyme rekursiv sammeln
            def get_all_hypernyms(syn, depth=0, max_depth=10):
                """Rekursiv alle Hypernyme bis zur Wurzel sammeln."""
                if depth >= max_depth:
                    return []
                hypernyms = syn.hypernyms()
                result = []
                for h in hypernyms:
                    result.append((h, depth))
                    result.extend(get_all_hypernyms(h, depth + 1, max_depth))
                return result

            all_hypernyms = get_all_hypernyms(synset)
            hypernym_names = [h.name() for h, _ in all_hypernyms]

            results.append(
                {
                    "id": synset.name(),
                    "definition": synset.definition(),
                    "pos": pos,
                    "hypernyms_all": all_hypernyms,
                    "hypernym_names": hypernym_names,
                    "synonyms_en": synset.lemma_names(),
                    "examples": synset.examples()[0] if synset.examples() else [],
                }
            )
        except Exception as e:
            # Falls ein Offset in der lokalen NLTK-Version fehlt
            continue

    return results


# --- ANWENDUNGSBEISPIEL ---
if __name__ == "__main__":
    # Pfad zu deiner MCR-Datei (bitte anpassen)
    MCR_FILE = "data/wordnet/mcr30-2016/spaWN/wei_spa-30_variant.tsv"

    # 1. Matrix/Daten laden
    print("Lade MCR Datenbank...")
    mcr_df = load_mcr_mapping(MCR_FILE)

    # 2. Test-Vokabeln abfragen
    test_words = ["zorro", "laberinto", "creer", "silla", "mesa"]

    for word in test_words:
        print(f"\n{'='*50}")
        print(f"BEDEUTUNGEN FÜR: {word.upper()}")
        print(f"{'='*50}")

        senses = get_word_senses(word, mcr_df)

        if senses:
            for i, sense in enumerate(senses, 1):
                print(f"\nSinn {i}: {sense['id']} ({sense['pos']})")
                print(f"Definition: {sense['definition']}")
                print(f"Synonyme (en): {', '.join(sense['synonyms_en'])}")
                print(f"Beispiel: {sense['examples']}")

                # Alle Hypernyme hierarchisch anzeigen
                if sense["hypernyms_all"]:
                    print(
                        f"\nHypernym-Hierarchie ({len(sense['hypernyms_all'])} Ebenen):"
                    )
                    for hyp, depth in sense["hypernyms_all"]:
                        indent = "  " * (depth + 1)
                        print(f"{indent}↑ {hyp.name()}")
                        print(f"{indent}  {hyp.definition()}")
                else:
                    print("Keine Hypernyme (Top-Level-Konzept)")
                print("-" * 50)
        else:
            print(f"Keine Einträge für '{word}' gefunden.")
