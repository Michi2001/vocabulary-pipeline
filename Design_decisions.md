# Designentscheidung: Clustering & Story-Generierung

## Kontext
- Apple Foundation Models (AFM): 3B Parameter, 4096 Token Kontextlimit
- Erste Tests zeigten: AFM-Clustering unzuverlässig (Wörter vergessen/dupliziert), Story-Qualität inkonsistent

## Evaluierte Optionen

| Option | Beschreibung | Bewertung |
|--------|--------------|-----------|
| **1. Algorithmisch + AFM** | Clustering algorithmisch, nur Stories via AFM | ✓ Gewählt |
| 2. AFM + AFM | Beide Schritte mit AFM, separat finegetuned | ✗ |
| 3. Alles-in-Einem | Wortauswahl + Story in einem AFM-Prompt | ✗ |

## Entscheidung: Option 1

```
Algorithmisches Clustering (Co-Occurrence + Embeddings)
        ↓
Algorithmische Anreicherung mit neuen Wörtern
        ↓
Finetuned AFM (nur Story-Generierung)
```

## Begründung

**Gegen AFM für Clustering:**
- Strukturierte Outputs (Listen, Zuordnungen) sind Schwäche kleiner LLMs
- 4096 Token Limit macht große Wortlisten unmöglich
- Finetuning verbessert Reasoning-Fähigkeiten nicht fundamental
- Schwierigkeit bei Debugging

**Für algorithmisches Clustering:**
- Deterministisch und debugbar
- Skaliert unabhängig vom Kontextlimit
- Klare Trennung der Zuständigkeiten

## Offener Punkt: Narrative Kompatibilität

Semantische Embedding-Ähnlichkeit ≠ narrative Kompatibilität ("rot" und "Fuchs" haben unterschiedliche Embeddings, passen aber gut zusammen).

**Lösung:** Bedeutungssensitive Co-Occurrence-Matrix via Word Sense Disambiguation auf Korpus-Ebene.


# Designetnscheidung: Kein Batchweises Caching beim lesen der Sätze

## Begründung
- Zuerst werden zufällig Sätze gesampelt, dann gelesen -> Sampling Schritt müsste angepasst werden
- Sentence Store: gzip unterstützt kein APPEND


# Designentscheidung: WordVariant IDs nicht persistiert

## Kontext
WordVariant IDs dienen als direkte Indizes in die Co-Occurrence-Matrix. Die IDs werden bei jedem Laden/Erstellen des Vocabulary neu berechnet (`WordVariant.from_translations()` in `vocab.py`), nicht aus einer Datei gelesen.

## Entscheidung
IDs werden **nicht persistiert**, sondern deterministisch aus den Translations abgeleitet (sortiert nach `derived_freq` desc, dann `translation.id` und `meaning_idx` als Tiebreaker).

## Begründung
- Sortierung ist deterministisch → gleiche Translations = gleiche IDs
- Co-Occurrence-Matrix wird im selben Pipeline-Run erzeugt wie das Vocabulary
- Disambiguation-Cache-Key enthält Vocabulary-Hash → bei Vocabulary-Änderung wird Matrix neu berechnet
- Persistierte IDs würden ID-Verwaltung und Migrations-Logik erfordern, ohne Mehrwert

## Einschränkung
Setzt voraus, dass die Matrix immer zusammen mit dem Vocabulary invalidiert wird. Falls in Zukunft die Matrix unabhängig vom Vocabulary aktualisiert werden soll, müsste man IDs persistieren.


# Designentscheidung: POS-Tag-Auswahl für die Vokabel-Pipeline

## Kontext
spaCy verwendet das Universal Dependencies (UD) Tagset mit 18 POS-Kategorien. Nicht alle sind für eine Vokabellern-App relevant. Die Auswahl bestimmt, welche Wörter in die Frequenzliste und damit in die Story-Generierung einfließen.

## Akzeptierte POS

| POS | Beschreibung | Beispiele (Spanisch) | Anmerkung |
|-----|-------------|---------------------|-----------|
| **NOUN** | Nomen | *casa, tiempo, vida* | Kernwortschatz |
| **VERB** | Verb | *hablar, comer, vivir* | Kernwortschatz |
| **ADJ** | Adjektiv | *grande, bueno, nuevo* | Produktiver Wortschatz |
| **ADV** | Adverb | *siempre, muy, también, nunca* | Viele hochfrequente Wörter |
| **ADP** | Adposition (Präposition) | *de, en, con, por, para, sin* | Kleine Menge (~20), aber *por* vs. *para* etc. erfordern aktives Lernen |
| **AUX** | Hilfsverb | *ser, estar, haber, poder* | Wird auf VERB gemappt |
| **CCONJ** | Koordinierende Konjunktion | *y, o, pero, sino, ni* | Wird auf CONJ gemappt |
| **SCONJ** | Subordinierende Konjunktion | *que, porque, aunque, si, cuando* | Wird auf CONJ gemappt |
| **PRON** | Pronomen | *yo, me, le, se, algo, nada, alguien* | Objektpronomen/Klitika sind ein häufiges Lernproblem |
| **INTJ** | Interjektion | *oh, ay, eh, oye* | Kleine Menge; relevant für gesprochenes Spanisch. spaCy taggt viele Diskursmarker (*bueno, mira, vamos*) kontextabhängig als ADJ/VERB, daher landet hier nur eine kleine Teilmenge |
| **NUM** | Zahlwort | *uno, dos, mil, cien* | Ausgeschriebene Zahlwörter sind lernrelevant. spaCy taggt auch Ziffern-Strings (*"42", "2024"*) als NUM — diese müssen separat herausgefiltert werden (Tokens die Ziffern enthalten) |

### POS-Mappings

| Original | Ziel | Begründung |
|----------|------|-----------|
| AUX | VERB | *ser, estar, haber* sollten als Verben gelernt werden |
| CCONJ | CONJ | Vereinheitlichung der Konjunktionen |
| SCONJ | CONJ | Vereinheitlichung der Konjunktionen |

## Nicht akzeptierte POS

| POS | Beschreibung | Beispiele (Spanisch) | Begründung |
|-----|-------------|---------------------|-----------|
| **DET** | Determiner | *el, un, este, cada, otro* | Artikel (*el, la, un, una*) sind trivial und werden kontextuell in Stories gelernt. Interessante Determiner (*cada, cualquier, ambos, cierto, demás, mismo, sendos*) hätten Lernwert, aber spaCy unterscheidet nicht — alles ist DET. Aufnahme würde triviale Artikel mit reinziehen. Option: DET aufnehmen und triviale per LLM-Korrektur (DELETE) entfernen |
| **PROPN** | Eigenname | *España, Madrid, Juan* | Nur Ländernamen werden einbezogen (via `include_countries: true`) |
| **PART** | Partikel | *no* (selten als PART getaggt) | Kommt im Spanischen kaum vor |
| **PUNCT** | Interpunktion | *. , ! ?* | Wird im Lemmatizer herausgefiltert |
| **SYM** | Symbol | *€, %, +* | Keine Vokabel |
| **SPACE** | Whitespace | Leerzeichen, Zeilenumbrüche | Wird im Lemmatizer herausgefiltert |
| **X** | Sonstiges | Fremdwörter, unleserliche Tokens | Rauschen |

## Konfiguration

```yaml
filtering:
  accepted_pos:
    - NOUN
    - VERB
    - ADJ
    - ADV
    - ADP
    - CONJ
    - AUX      # → VERB
    - CCONJ    # → CONJ
    - SCONJ    # → CONJ
    - PRON
    - INTJ
    - NUM
  pos_mappings:
    AUX: VERB
    CCONJ: CONJ
    SCONJ: CONJ
```