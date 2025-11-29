# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spanish vocabulary frequency analyzer that processes large text corpora using spaCy for contextual lemmatization. The system generates frequency lists of Spanish lemmas sorted by occurrence, with support for multiple corpora and caching.

## Development Commands

### Running the Pipeline
```bash
# Main execution - processes corpora and generates frequency lists
python main.py
```

### Testing
```bash
# Test lemmatization functionality
python test_lemmatization.py
```

### Dependencies
The project requires spaCy with Spanish language models:
```bash
# Install Spanish language model (small/medium/large)
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg
```

## Architecture

### Processing Pipeline Flow

The pipeline follows a stream-based architecture for memory efficiency:

1. **CorpusReader** (src/corpus_readers/) → yields sentence batches
2. **Lemmatizer** (src/lemmatizer.py) → yields (lemma, pos, is_oov) tuples
3. **FrequencyCounter** (src/frequency_counter.py) → aggregates frequencies incrementally
4. **OutputWriter** (src/output_writer.py) → writes CSV results

### Key Components

**SingleCorpusProcessor** (src/single_corpus_processor.py)
- Orchestrates processing for one corpus
- Implements caching using pickle with hash-based keys (corpus filename, mtime, model, POS categories)
- Cache structure: `cache/{corpus_name}/{hash}.pkl`

**MultiCorpusProcessor** (src/multi_corpus_processor.py)
- Processes multiple corpora sequentially
- Merges frequency counts across all corpora using `FrequencyCounter.merge()`

**Lemmatizer** (src/lemmatizer.py)
- Uses spaCy for contextual lemmatization
- POS filtering: Only processes NOUN, VERB, ADJ, ADV, ADP, CONJ (normalizes AUX→VERB, CCONJ/SCONJ→CONJ)
- Ignores: DET, PROPN, NUM, INTJ, PRON, PUNCT, PART, SYM, X, SPACE
- Cleans multi-word lemmas by removing enclitic pronouns
- Processes in batches for efficiency

**FrequencyCounter** (src/frequency_counter.py)
- Tracks lemmas with main POS categories (NOUN, VERB, ADJ, ADV, CONJ) as separate entries
- Keys format: `lemma|POS` (e.g., "hablar|VERB", "hablar|NOUN")
- Separate tracking for OOV (out-of-vocabulary) words
- Supports serialization via `to_dict()`/`from_dict()` for caching

### Corpus Reader Pattern

Abstract base class pattern (src/corpus_readers/base_reader.py) enables support for different corpus formats:
- **LeipzigCorpusReader**: Tab-separated format (ID\tSentence)
- Extensible to XML, JSON, plain text via subclassing

### Configuration

**config.yaml** structure:
- `corpora`: List of corpus configs (name, file_path, type, encoding)
- `lemmatization`: spaCy model and batch_size
- `processing`: sentence_batch_size, min_frequency
- `output`: directory (timestamped subdirs created automatically)
- `cache`: enabled, directory

### Output Structure

Results are written to timestamped directories:
```
output/spanish_lemma_frequencies/YYYYMMDD_HHMMSS/
├── config.yaml           # Snapshot of configuration used
├── output.csv           # Known words (lemma, frequency, pos)
└── output_oov.csv       # Out-of-vocabulary words
```

### Memory Efficiency Design

The entire pipeline uses generators to process arbitrarily large corpora without loading full datasets:
- CorpusReader yields sentence batches
- Lemmatizer yields token tuples via `nlp.pipe()`
- FrequencyCounter updates incrementally (logs every 100k tokens)

### Cache Key Generation

Cache invalidation occurs when:
- Corpus file content changes (mtime check)
- spaCy model changes
- MAIN_POS_CATEGORIES constant changes

Batch sizes do NOT affect cache keys (performance-only parameters).

## Data Expectations

Place Leipzig Corpora files in `data/Leipzig_Wortschatz/{corpus_name}/{corpus_name}-sentences.txt` format.

