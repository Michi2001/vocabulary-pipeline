from PyMultiDictionary import MultiDictionary

# Erstelle eine Instanz des Wörterbuchs
dictionary = MultiDictionary()

# Wort festlegen
word = "laberinto"

if __name__ == "__main__":
    print(f"Wort: {word}\n")
    # 1. Bedeutungen (Meanings) abrufen
    # Rückgabe ist ein Tupel: ([Wortarten], "Definition", "Wikipedia-Auszug")
    meanings = dictionary.meaning("es", word)
    print(f"Bedeutung von {word}: {meanings}")

    # 2. Synonyme abrufen (Gibt eine Liste zurück)
    synonyms = dictionary.synonym("es", word)
    print(f"Synonyme: {synonyms}")

    # 3. Übersetzungen
    # Übersetzt das Wort standardmäßig in alle unterstützten Sprachen
    translations = dictionary.translate("es", word)
    print(f"Übersetzung (en): {dictionary.translate('es', word, to='en')}")
