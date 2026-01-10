# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spanish vocabulary frequency analyzer and word sense disambiguation system that processes large text corpora using spaCy for contextual lemmatization. The pipeline transforms raw Spanish text into a disambiguated, semantically-enriched vocabulary dataset through seven stages:

1. **Frequency Analysis**: Contextual lemmatization and frequency counting with caching
2. **Post-Processing**: POS filtering, mapping, and lemma cleaning
3. **LLM Correction**: Optional LLM-based lemmatization error correction
4. **Translation**: Optional LLM-based translation to German with multiple meanings
5. **Meaning Embeddings**: BERT embeddings for each meaning variant
6. **Word Sense Disambiguation**: Corpus-wide disambiguation using embedding similarity
7. **Co-Occurrence Matrix**: Semantic co-occurrence matrix for vocabulary relationships

The system supports multiple corpora, aggressive caching at each stage, and streaming processing for memory efficiency. Primary use case: vocabulary learning applications requiring disambiguated word meanings and semantic relationships.

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

For disambiguation and meaning embeddings, the pipeline uses BERT models from Hugging Face:
```bash
# Default model (automatically downloaded on first use)
# dccuchile/bert-base-spanish-wwm-cased
```

Other dependencies are managed via `requirements.txt`.

## Architecture

### Processing Pipeline Flow

The pipeline follows a multi-stage architecture with stream-based processing:

**Stage 1: Frequency Analysis (with caching)**
1. **CorpusReader** (src/corpus_readers/) → yields sentence batches
2. **SentenceStore** (src/sentence_store.py) → stores original tokens in `sentences.jsonl.gz` (if enabled)
3. **Lemmatizer** (src/lemmatizer.py) → yields ALL tokens with original POS tags and lemmas
4. **SentenceStore** → stores lemmatized tokens in `lemmatized.jsonl.gz` (if enabled)
5. **FrequencyCounter** (src/frequency_counter.py) → aggregates raw frequencies (lemma|ORIGINAL_POS)
6. **Cache** → stores raw, unfiltered frequency data in pickle format

**Stage 2: Post-Processing (after cache)**
7. **PostProcessor** (src/post_processor.py) → applies POS filtering, mapping, lemma cleaning
8. **OutputWriter** (src/output_writer.py) → writes filtered CSV results

**Stage 3: LLM Correction (optional, with caching)**
9. **LemmaCorrector** (src/llm_corrector.py) → sends lemmas to LLM in batches for validation
   - Identifies lemmatization errors (e.g., incorrect conjugations)
   - Supports three operations: KEEP, DELETE, MERGE
   - Applies corrections to FrequencyCounter (removes deleted, merges variants)
   - Saves correction mapping to SentenceStore → `correction_mapping.pkl`
   - Corrected tokens generated dynamically via `iter_corrected()`
   - Cache key includes model config + system prompt hash
   - Outputs: `original_with_ids.csv`, `llm_corrections.csv`

**Stage 4: Translation (optional, with caching)**
10. **LemmaTranslator** (src/llm_translator.py) → translates Spanish lemmas to German
    - Processes lemmas by POS category (VERB, NOUN, ADJ, ADV, ADP, CONJ)
    - Uses POS-specific prompts for accurate translations
    - Generates multiple meanings with definitions and estimated frequency share
    - Cache key includes model config + combined prompt hash (general + POS-specific)
    - Outputs: `translations.json` with enriched frequency data

**Stage 5: Meaning Embeddings (optional, requires translation)**
11. **MeaningEmbeddingGenerator** (src/meaning_embedding_generator.py) → generates BERT embeddings
    - Creates embeddings for each meaning's Spanish example sentence
    - Uses target word masking to extract context embeddings
    - Adds `meaning_embedding` field to each meaning in `translations.json`

**Stage 6: Word Sense Disambiguation (optional, requires embeddings + sentence_store)**
12. **Vocabulary** (src/vocab.py) → builds from translations
    - Creates WordVariant objects (one per meaning)
    - Provides lookup structures for disambiguation
13. **BatchDisambiguator** (src/batch_disambiguator.py) → disambiguates corpus tokens
    - Compares runtime token embeddings with pre-computed meaning embeddings
    - Uses cosine similarity to select best meaning variant
    - Processes corrected lemmas if available, otherwise uses original lemmas
    - Writes variant IDs to SentenceStore → `disambiguated.jsonl.gz`

**Stage 7: Co-Occurrence Matrix (optional, requires disambiguation)**
14. **CoOccurrenceBuilder** (src/cooccurrence_builder.py) → builds co-occurrence matrix
    - Uses disambiguated variant IDs from SentenceStore
    - Creates sparse co-occurrence matrix across corpus
    - Outputs: `cooccurrence_matrix.npz`

This separation enables cache reusability across different configurations and progressive enhancement of the data through the pipeline stages.

### Key Components

**SingleCorpusProcessor** (src/single_corpus_processor.py)
- Orchestrates processing for one corpus
- Implements caching using pickle with hash-based keys (corpus filename, mtime, model, max_sentences)
- Integrates PostProcessor for filtering after cache load
- Creates and manages SentenceStore (if `sentence_store.enabled`)
- Cache structure: `cache/{corpus_name}/{hash}.pkl` (frequency data)
- SentenceStore structure: `cache/{corpus_name}/sentence_store/` (sentence data)

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

**SentenceStore** (src/sentence_store.py)
- Stores sentence data in separate compressed JSONL files
- Index-aligned architecture: all token lists have same length/order per sentence
- Four file types:
  - `sentences.jsonl.gz`: Original sentence tokens
  - `lemmatized.jsonl.gz`: Lemmatized tokens (lemma, pos) tuples
  - `correction_mapping.pkl`: Correction mapping and deleted tokens (compact)
  - `disambiguated.jsonl.gz`: Variant IDs after disambiguation
- `iter_corrected()` applies mapping dynamically to lemmatized tokens (no separate file)
- Streaming readers for memory efficiency
- Parallel iteration methods (e.g., `iter_sentences_with_lemmatized()`)

**Vocabulary** (src/vocab.py)
- Central storage for translation data
- Three main classes:
  - `Translation`: Lemma with all its meanings (from LLM)
  - `Meaning`: Single sense with definitions, translations, embeddings
  - `WordVariant`: Flattened meaning variant (one per meaning)
- Provides lookup structures:
  - By lemma token: `(lemma, pos)` → Translation
  - By variant ID: `variant_id` → WordVariant
- Supports sampling and grouping for testing

**MeaningEmbeddingGenerator** (src/meaning_embedding_generator.py)
- Generates BERT embeddings for each meaning's example sentence
- Uses target word masking to extract context embeddings
- Batch processing for efficiency
- Embeddings stored in `meaning_embedding` field

**BatchDisambiguator** (src/batch_disambiguator.py)
- Disambiguates polysemous words using BERT embeddings
- Compares runtime token embeddings with pre-computed meaning embeddings
- Uses cosine similarity for meaning selection
- Handles mono-semantic and polysemous lemmas separately
- Writes results directly to SentenceStore

**CoOccurrenceBuilder** (src/cooccurrence_builder.py)
- Builds co-occurrence matrix from disambiguated corpus
- Uses sparse matrix format for memory efficiency
- Counts variant co-occurrences within sentences
- Outputs: `cooccurrence_matrix.npz`


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
- `filtering`: accepted_pos, pos_mappings, main_pos_categories, clean_lemmas, include_countries
- `llm_cache`: enabled, directory (for LLM correction/translation cache)
- `llm_correction`: enabled, batch_size, system_prompt_file, slice, llm provider config
- `llm_translation`: enabled, batch_size, prompt_directory, llm provider config
- `sentence_store`: enabled (required for disambiguation and co-occurrence)
- `disambiguation`: enabled, model (BERT model name), batch_size
- `cooccurrence`: enabled (requires disambiguation)

### Output Structure

Results are written to timestamped directories:
```
output/spanish_lemma_frequencies/YYYYMMDD_HHMMSS/
├── config.yaml              # Snapshot of configuration used
├── pipeline.log             # Complete terminal output log
├── output.csv               # Lemma frequency list (lemma, frequency, pos)
├── original_with_ids.csv    # Original lemmas with IDs (if correction enabled)
├── llm_corrections.csv      # LLM correction operations (if correction enabled)
├── translations.json        # Translations with embeddings (if translation enabled)
└── cooccurrence_matrix.npz  # Co-occurrence matrix (if cooccurrence enabled)
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
        "definition_es": "Existir, tener realidad",
        "estimated_share": 0.7,
        "translations": [
          {"text": "sein", "hint": "existieren"}
        ],
        "structure": "verb conjugation pattern",
        "example_es": "El ser humano piensa",
        "flags": ["common"],
        "meaning_embedding": [0.123, -0.456, ...],  // 768-dim BERT vector
        "_target_word": "ser",
        "_target_start": 3,
        "_target_end": 6
      },
      {
        "definition_es": "Pertenecer, formar parte de",
        "estimated_share": 0.3,
        "translations": [
          {"text": "sein", "hint": "zugehören"}
        ],
        "meaning_embedding": [0.234, -0.567, ...]
      }
    ]
  }
]
```

### SentenceStore Structure (if enabled)

The SentenceStore creates a cache directory with compressed files:
```
cache/{corpus_name}/
└── sentence_store/
    ├── sentences.jsonl.gz         # Original tokens: {"sid": 0, "tokens": ["El", "gato", ...]}
    ├── lemmatized.jsonl.gz        # Lemmas: {"sid": 0, "tokens": [["el","DET"], ["gato","NOUN"], ...]}
    ├── correction_mapping.pkl     # Correction mapping (if llm_correction enabled)
    └── disambiguated.jsonl.gz     # Variant IDs (if disambiguation enabled)
```

**Efficiency Design:**
- `correction_mapping.pkl` stores only the mapping, not full corrected sentences
- `iter_corrected()` applies mapping dynamically when reading
- Saves disk space: O(unique corrections) vs O(total sentences)
- All files are index-aligned: same sentence ID, same token count, same order

### Memory Efficiency Design

The entire pipeline uses generators and streaming to process arbitrarily large corpora without loading full datasets:
- CorpusReader yields sentence batches
- Lemmatizer yields token tuples via `nlp.pipe()`
- FrequencyCounter updates incrementally (logs every 100k tokens)
- SentenceStore uses streaming writers (compressed JSONL)
- SentenceStore readers are generators (no full load into memory)
- BatchDisambiguator processes sentences in batches

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

### Pipeline Stage Dependencies

The pipeline stages have the following dependencies:

**Stage 1-2: Frequency Analysis + Post-Processing**
- Always runs (required)
- No dependencies
- Outputs: `output.csv`, frequency cache

**Stage 3: LLM Correction**
- Optional (enabled via `llm_correction.enabled`)
- Requires: Frequency data from Stage 1-2
- Outputs: `original_with_ids.csv`, `llm_corrections.csv`, corrected FrequencyCounter
- If `sentence_store.enabled`: saves `correction_mapping.pkl` (compact)

**Stage 4: Translation**
- Optional (enabled via `llm_translation.enabled`)
- Requires: Frequency data (corrected or original)
- Outputs: `translations.json`

**Stage 5: Meaning Embeddings**
- Optional (enabled via `disambiguation.enabled`)
- Requires: `translations.json` from Stage 4
- Modifies: Adds `meaning_embedding` field to each meaning in `translations.json`

**Stage 6: Word Sense Disambiguation**
- Optional (enabled via `disambiguation.enabled`)
- Requires:
  - `sentence_store.enabled` (must be true)
  - `translations.json` with embeddings from Stage 5
  - `sentences.jsonl.gz` and `lemmatized.jsonl.gz` (or `correction_mapping.pkl`) from SentenceStore
- Outputs: `disambiguated.jsonl.gz` in SentenceStore

**Stage 7: Co-Occurrence Matrix**
- Optional (enabled via `cooccurrence.enabled`)
- Requires:
  - `disambiguation.enabled` (must be true)
  - `disambiguated.jsonl.gz` from Stage 6
- Outputs: `cooccurrence_matrix.npz`

**Dependency Chain Summary:**
```
Stage 1-2 (always) → Stage 3 (optional) → Stage 4 (optional) → Stage 5 (optional, auto-enabled with Stage 6)
                                                                        ↓
                                            sentence_store.enabled → Stage 6 (optional) → Stage 7 (optional)
```

## Data Expectations

**Leipzig Corpora:** Place files in `data/Leipzig_Wortschatz/{corpus_name}/{corpus_name}-sentences.txt` format.
**OPUS Corpora:** Place .txt files with one sentence per line anywhere (path specified in config.yaml).



# Claude Code Rules

* Keep the code as simple as possible.
* Inline comments: Add comments only for non-obvious lines or logic. Avoid unnecessary inline comments.
* Docstrings:
    * Use a short one-line docstring for most functions.
    + Only explain parameters or behavior when they are not immediately obvious from the code itself.