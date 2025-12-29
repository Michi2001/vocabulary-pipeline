# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spanish vocabulary frequency analyzer that processes large text corpora using spaCy for contextual lemmatization. The system generates frequency lists of Spanish lemmas sorted by occurrence, with support for multiple corpora and caching. The pipeline extends beyond frequency analysis to include LLM-based correction, translation to German, and semantic embedding generation for vocabulary learning applications.

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

The pipeline follows a multi-stage architecture with stream-based processing:

**Stage 1: Frequency Analysis (with caching)**
1. **CorpusReader** (src/corpus_readers/) → yields sentence batches
2. **Lemmatizer** (src/lemmatizer.py) → yields ALL tokens with original POS tags and lemmas
3. **FrequencyCounter** (src/frequency_counter.py) → aggregates raw frequencies (lemma|ORIGINAL_POS)
4. **Cache** → stores raw, unfiltered frequency data

**Stage 2: Post-Processing (after cache)**
5. **PostProcessor** (src/post_processor.py) → applies POS filtering, mapping, lemma cleaning
6. **OutputWriter** (src/output_writer.py) → writes filtered CSV results

**Stage 3: LLM Correction (optional, with caching)**
7. **LemmaCorrector** (src/llm_corrector.py) → sends lemmas to LLM in batches for validation
   - Identifies lemmatization errors (e.g., incorrect conjugations)
   - Supports three operations: KEEP, DELETE, MERGE
   - Cache key includes model config + system prompt hash
   - Outputs: `original_with_ids.csv`, `llm_corrections.csv`

**Stage 4: Translation (optional, with caching)**
8. **LemmaTranslator** (src/llm_translator.py) → translates Spanish lemmas to German
   - Processes lemmas by POS category (VERB, NOUN, ADJ, ADV, ADP, CONJ)
   - Uses POS-specific prompts for accurate translations
   - Generates multiple meanings with definitions and estimated frequency share
   - Cache key includes model config + combined prompt hash (general + POS-specific)
   - Outputs: `translations.json` with enriched frequency data

**Stage 5: Embedding Generation (optional)**
9. **EmbeddingGenerator** (src/embedding_generator.py) → generates semantic embeddings
   - Creates vector embeddings for Spanish definitions (`definition_es`)
   - Uses sentence-transformers models (e.g., LaBSE)
   - Enables semantic clustering and similarity search
   - Embeddings stored in `translations.json` under each meaning

This separation enables cache reusability across different configurations and progressive enhancement of the data.

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
- Supports serialization via `to_dict()`/`from_dict()` for caching

**PostProcessor** (src/post_processor.py)
- Applies post-processing filters to raw frequency data
- POS filtering: Only include specified POS categories (configurable)
- POS mapping: Merge categories (e.g., AUX→VERB, CCONJ/SCONJ→CONJ)
- Lemma cleaning: Remove enclitic pronouns from multi-word lemmas
- POS normalization: Group by main categories or "OTHER"

**LemmaCorrector** (src/llm_corrector.py)
- Sends lemmas to LLM for quality validation and correction
- Batch processing with intelligent caching (prompt hash-based)
- Three operations: KEEP (correct), DELETE (invalid), MERGE (combine variants)
- Pre-loads cache to minimize API calls
- Applies corrections to FrequencyCounter (removes deleted, merges variants)
- Configurable slicing (only correct top N lemmas)

**LemmaTranslator** (src/llm_translator.py)
- Translates Spanish lemmas to German with POS-aware prompts
- Sequential processing by POS category (VERB, NOUN, ADJ, ADV, ADP, CONJ)
- Each POS uses specific prompt (general + POS-specific combined)
- Generates multiple meanings with:
  - German translations (infinitive form)
  - Spanish definitions (`definition_es`)
  - German definitions (`definition_de`)
  - Estimated frequency share per meaning
- Cache key combines model config + general prompt hash + POS prompt hash
- Normalizes `estimated_share` values to sum to 1.0
- Tracks token usage statistics (prompt/completion tokens, API calls, cache hits)

**LemmaCache** (src/lemma_cache.py)
- Unified cache for both correction and translation results
- Two-level directory structure: `cache/lemma_cache/{correction|translation}/`
- Cache key format: `{lemma}|{pos}_{model_hash}_{prompt_hash}.json`
- Model hash includes: provider type, model name, temperature, max_tokens
- Separate prompt hashes for correction vs translation
- In-memory buffer with periodic disk writes (`save_all()`)

**EmbeddingGenerator** (src/embedding_generator.py)
- Generates semantic embeddings for Spanish definitions (`definition_es`)
- Uses sentence-transformers models (default: LaBSE for multilingual support)
- Batch encoding with progress tracking
- Adds `embedding` field (vector as list) to each meaning in translations
- Enables downstream clustering and semantic similarity search

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
- `cache`: enabled, directory (for spaCy frequency cache)
- `filtering`: accepted_pos, pos_mappings, main_pos_categories, clean_lemmas
- `llm_cache`: enabled, directory (for LLM correction/translation cache)
- `llm_correction`: enabled, batch_size, system_prompt_file, slice, llm provider config
- `llm_translation`: enabled, batch_size, prompt_directory, llm provider config
- `embeddings`: enabled, model, batch_size

### Output Structure

Results are written to timestamped directories:
```
output/spanish_lemma_frequencies/YYYYMMDD_HHMMSS/
├── config.yaml              # Snapshot of configuration used
├── pipeline.log             # Complete terminal output log
├── output.csv               # Lemma frequency list (lemma, frequency, pos)
├── original_with_ids.csv    # Original lemmas with IDs (if correction enabled)
├── llm_corrections.csv      # LLM correction operations (if correction enabled)
└── translations.json        # Translations with embeddings (if translation/embeddings enabled)
```

The `translations.json` structure (if enabled):
```json
[
  {
    "id": "000001",
    "lemma": "ser",
    "pos": "VERB",
    "frequency": 123456,
    "meanings": [
      {
        "translation": "sein",
        "definition_es": "Existir, tener realidad",
        "definition_de": "Existieren, Realität haben",
        "estimated_share": 0.7,
        "embedding": [0.123, -0.456, ...]  // 768-dim vector
      },
      {
        "translation": "sein",
        "definition_es": "Pertenecer, formar parte de",
        "definition_de": "Gehören, Teil sein von",
        "estimated_share": 0.3,
        "embedding": [0.234, -0.567, ...]
      }
    ]
  }
]
```

### Memory Efficiency Design

The entire pipeline uses generators to process arbitrarily large corpora without loading full datasets:
- CorpusReader yields sentence batches
- Lemmatizer yields token tuples via `nlp.pipe()`
- FrequencyCounter updates incrementally (logs every 100k tokens)

### Cache Key Generation

**Frequency Cache** (spaCy-based):
Cache invalidation occurs when:
- Corpus file content changes (mtime check)
- spaCy model changes
- max_sentences parameter changes

Cache key does NOT include:
- Batch sizes (performance-only parameters)
- Filtering configuration (applied during post-processing)

**LLM Cache** (correction/translation):
Cache key format: `{lemma}|{pos}_{model_hash}_{prompt_hash}.json`

Cache invalidation occurs when:
- Model configuration changes (type, name, temperature, max_tokens)
- System prompt changes (hash-based detection)

Cache key does NOT include:
- Batch sizes (performance-only parameters)
- Frequency data (only lemma and POS are key)

This design enables cache reuse across different configurations and runs.

## Data Expectations

**Leipzig Corpora:** Place files in `data/Leipzig_Wortschatz/{corpus_name}/{corpus_name}-sentences.txt` format.
**OPUS Corpora:** Place .txt files with one sentence per line anywhere (path specified in config.yaml).



# Claude Code Rules

* Keep the code as simple as possible.
* Inline comments: Add comments only for non-obvious lines or logic. Avoid unnecessary inline comments.
* Docstrings:
    * Use a short one-line docstring for most functions.
    + Only explain parameters or behavior when they are not immediately obvious from the code itself.