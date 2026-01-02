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