## SentenceStore -> Aufteilen in 3 Klassen: SentenceStore, LemmaStore, DisambiguatedStore
- Speicherorte
    - SentenceStore: cache / corpus_name / cache_key (Aus Single Corpus Processor) / "SentenceStore.pkl"
    - Frequency Counter: cache / corpus_name / cache_key / "FrequencyCounter.pkl"
    - LemmaStore: cache / corpus_name / cache_key (Aus Single Corpus Processor)/ "LemmaStore.pkl"
    - DisambiguatedStore: cache / corpus_name / cache_key (Aus Single Corpus Processor) / Disambiguation_cache_key / DisambiguatedStore.pkl
        - Disambiguated Store bezieht sich immer auf einen Lemma / SentenceStore, daher macht es Sinn, diese direkt neben diesen zu speichern
        - Im Disambiguation cache key muss ebenfalls der Cache Key des Vocabularies einfließen, da eine Änderung am Vocabulary auch Einfluss auf die Disambiguierung hat
    - Co-occurenceMatrix: Cache / corpus_name / cache_key (Aus Single Corpus Processor) / Disambiguated_cache_key /  co-occurence_matrix.pkl (oder passenderes Format)

### BaseStore: 
- init(base_path)
- Alle Klasse speichern File mit json lines: sid, tokens / variants
- __iter__: Gibt tupel aus sid, tokens / variants zurück
- iter_batched(batch_size)

### LemmaStore
- Speichert das Correction Mapping nicht mehr unter einem Pfad, sondern hält es einfach in Memory, sobald übergeben
- iter_corrected, iter_corrected_batched

### SingleCorpusProcessor -> Umbenennung: ??

- process -> SentenceStore, LemmaStore, FrequencyCounter
- Speichern von Sentence und LemmaStore in den entsprechenden Dictionaries
- Caching: Weiterhin prüfen: Existiert bereits ein Sentence / LemmaStore / 

### main
- Erstellen von Dictionaries: dataset_name -> SentenceStore, LemmaStore, DisambiguatedStore
- Disambiguierung: Iteration über alle Sentence + LemmaStores, aufruf von Disambiguate für alle Stores
    - Speicherung in disambiguated_store dictionary


### BatchDisambiguator
- process: Übergabe von SentenceStore, LemmaStore
- Ausgabe: DisambiguatedStore für EINEN Datensatz


### Co-occurenceBuilder
- Übergabe: List[DisambiguatedStore]
- Ausgabe: SparseMatrix


### Caching
- SingleCorpusProcessor
    - Wie bisher: Prüfen, ob bereits ein SentenceStore / LemmaStore / Frequency Counter für diesen Cache Key exisitiert -> Wenn ja zurückgeben, sonst erneut berechnen
    - Translation / Correction: Wie bisher über Database
- Disambiguator:
    - Analog zu SingleCorpusProcessor: Prüfen, ob für den Cache Key bereits ein Disambiguated Store exisitiert, wenn ja -> Laden und zurückgeben, sonst: neu berechnen
- CoOccurenceBuilder
    - Caching hinzufügen
    - Prüfen, ob unter entsrprechenden Pfad bereits eine Co-occurence Matrix existiert
    - Falls ja, laden und zurückgeben, sonst neu erstellen


### Fragen
- macht es Sinn, den Frequency Counter im LemmaStore zu sepichern ??
- Problem: Correction Mapping muss für alle LemmaStore mehrmals gespeichert werden ??
- Macht es Sinn, den MultiCorpusProcessor zu entfernen und seine Logik direkt in Main auszuführen ??