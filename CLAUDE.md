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

The pipeline follows a two-stage architecture with stream-based processing:

**Stage 1: Raw Counting (with caching)**
1. **CorpusReader** (src/corpus_readers/) → yields sentence batches
2. **Lemmatizer** (src/lemmatizer.py) → yields ALL tokens with original POS tags and lemmas
3. **FrequencyCounter** (src/frequency_counter.py) → aggregates raw frequencies (lemma|ORIGINAL_POS)
4. **Cache** → stores raw, unfiltered frequency data

**Stage 2: Post-Processing (after cache)**
5. **PostProcessor** (src/post_processor.py) → applies POS filtering, mapping, lemma cleaning
6. **OutputWriter** (src/output_writer.py) → writes filtered CSV results

This separation enables cache reusability across different filter configurations.

### Key Components

**SingleCorpusProcessor** (src/single_corpus_processor.py)
- Orchestrates processing for one corpus
- Implements caching using pickle with hash-based keys (corpus filename, mtime, model, max_sentences)
- Integrates PostProcessor for filtering after cache load
- Cache structure: `cache/{corpus_name}/{hash}.pkl`

**MultiCorpusProcessor** (src/multi_corpus_processor.py)
- Processes multiple corpora sequentially
- Merges frequency counts across all corpora using `FrequencyCounter.merge()`

**Lemmatizer** (src/lemmatizer.py)
- Uses spaCy for contextual lemmatization
- Yields ALL tokens with original POS tags and original lemmas (no filtering)
- Skips only whitespace and punctuation tokens
- Processes in batches for efficiency

**FrequencyCounter** (src/frequency_counter.py)
- Stores raw frequencies with original POS tags (no normalization)
- Keys format: `lemma|ORIGINAL_POS` (e.g., "ser|AUX", "hablar|VERB")
- Separate tracking for OOV (out-of-vocabulary) words
- Supports serialization via `to_dict()`/`from_dict()` for caching

**PostProcessor** (src/post_processor.py)
- Applies post-processing filters to raw frequency data
- POS filtering: Only include specified POS categories (configurable)
- POS mapping: Merge categories (e.g., AUX→VERB, CCONJ/SCONJ→CONJ)
- Lemma cleaning: Remove enclitic pronouns from multi-word lemmas
- POS normalization: Group by main categories or "OTHER"

### Corpus Reader Pattern

Abstract base class pattern (src/corpus_readers/base_reader.py) enables support for different corpus formats:
- **LeipzigCorpusReader**: Tab-separated format (ID\tSentence)
- **OpusCorpusReader**: One sentence per line (.txt format)
- Supports optional max_sentences parameter for random sampling
- Extensible to other formats via subclassing

### Configuration

**config.yaml** structure:
- `corpora`: List of corpus configs (name, file_path, type, encoding, max_sentences)
- `lemmatization`: spaCy model and batch_size
- `processing`: sentence_batch_size, min_frequency
- `output`: directory (timestamped subdirs created automatically)
- `cache`: enabled, directory
- `filtering`: accepted_pos, pos_mappings, main_pos_categories, clean_lemmas

### Output Structure

Results are written to timestamped directories:
```
output/spanish_lemma_frequencies/YYYYMMDD_HHMMSS/
├── config.yaml           # Snapshot of configuration used
├── pipeline.log          # Complete terminal output log
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
- max_sentences parameter changes

Cache key does NOT include:
- Batch sizes (performance-only parameters)
- Filtering configuration (applied during post-processing)

This design enables cache reuse across different filter configurations.

## Data Expectations

**Leipzig Corpora:** Place files in `data/Leipzig_Wortschatz/{corpus_name}/{corpus_name}-sentences.txt` format.
**OPUS Corpora:** Place .txt files with one sentence per line anywhere (path specified in config.yaml).

