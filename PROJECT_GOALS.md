# Projektziele: Vokabellern-App mit Story-Generierung

## Vision
Das Ziel dieses Projekts ist die Entwicklung einer Vokabellern-App, die das Prinzip der **Spaced Repetition** (verteilte Wiederholung) nutzt, ähnlich wie Anki. Die App soll jedoch über das bloße Abfragen von Karteikarten hinausgehen, indem sie die zu lernenden Vokabeln in generierte Kurzgeschichten einbettet. Dies ermöglicht das Lernen im Kontext.

## Datenbasis
*   Die Vokabeln liegen im JSON-Format unter `data/translations.json` vor.
*   Die Liste ist nach Worthäufigkeit (Frequenz) sortiert, sodass wichtigere Wörter zuerst gelernt werden.

## Funktionsweise

1.  **Spaced Repetition:** Die App berechnet basierend auf dem Nutzerfeedback (verstanden/nicht verstanden), wann ein Wort erneut zur Wiederholung fällig ist.
2.  **Kontextuelles Lernen:** Anstatt isolierter Abfragen werden **Kurzgeschichten** generiert, die die fälligen Wiederholungswörter enthalten.
3.  **Nutzerinteraktion:** Der Nutzer liest die Geschichte und gibt für jedes hervorgehobene Wiederholungswort an, ob die Bedeutung bekannt war.
4.  **Neue Wörter:** Neben den Wiederholungswörtern werden sukzessive neue, noch unbekannte Wörter (basierend auf ihrer Frequenz) in die Geschichten eingeführt.

## Technologie
*   **KI-Modell:** Apple Intelligence (lokal oder via API) wird für die Generierung der Geschichten verwendet.

## Pipeline zur Story-Generierung
Da die Kapazität und "Intelligenz" des verwendeten Modells begrenzt sein könnten (Schwierigkeit, *alle* fälligen Wörter in *eine* sinnvolle Geschichte zu packen), wird folgende Pipeline implementiert:

### Schritt 1: Gruppierung der Wiederholungswörter
Die aktuell fälligen Wiederholungswörter werden in sinnvolle, handhabbare Gruppen unterteilt.

### Schritt 2: Anreicherung mit neuen Wörtern
Für jede Gruppe werden passende neue Wörter ausgewählt. Diese stammen aus dem Pool der dem Nutzer noch unbekannten Wörter mit der höchsten Frequenz. Eine feste Anzahl dieser neuen Wörter wird der jeweiligen Gruppe hinzugefügt.

### Schritt 3: Generierung der Kurzgeschichten
Für jede so erstellte Gruppe (bestehend aus Wiederholungswörtern + neuen Wörtern) wird eine individuelle Kurzgeschichte generiert. Diese Geschichten dienen als Lernmaterial für die aktuelle Sitzung.

## Pipeline zur Erstellung der Vokabelliste und Wort-Vektoren

Die Vokabelliste in `data/translations.json` wird durch eine mehrstufige Pipeline erstellt, die große spanische Textkorpora analysiert und mit semantischen Informationen anreichert.

### Überblick der Pipeline-Stufen

**1. Frequenzanalyse** (spaCy-basiert, gecached)
- Verarbeitung großer Textkorpora (Leipzig, OPUS)
- Kontextbasierte Lemmatisierung mit spaCy
- Zählung der Worthäufigkeiten nach Lemma und Wortart
- Stream-basierte Verarbeitung (beliebig große Korpora)

**2. LLM-Korrektur**
- Validierung und Korrektur von Lemmatisierungsfehlern
- Batch-Verarbeitung mit intelligentem Caching
- Operationen: KEEP (korrekt), DELETE (ungültig), MERGE (Varianten zusammenführen)

**3. LLM-Übersetzung**
- Übersetzung spanischer Lemmas ins Deutsche
- Wortart-spezifische Prompts (VERB, NOUN, ADJ, ADV, ADP, CONJ)
- Generierung mehrerer Bedeutungen pro Lemma mit:
  - Deutsche Übersetzungen
  - Spanische und deutsche Definitionen
  - Geschätzter Häufigkeitsanteil pro Bedeutung

**4. Embedding-Generierung**
- Erzeugung semantischer Vektoren für spanische Definitionen
- Verwendung von Sentence-Transformers (LaBSE)
- Ermöglicht semantisches Clustering und Ähnlichkeitssuche
- Basis für intelligente Wortgruppierung in Stories

