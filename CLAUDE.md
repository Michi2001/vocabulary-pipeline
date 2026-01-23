# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## IMPORTANT: Code Rules (MUST follow)

### Code Style
- **Imports:** ALWAYS at the TOP of the file. NEVER inside functions/methods.
- **Keep it Simple:** Avoid over-engineering. Minimal code for the task.
- **Comments:** Only for non-obvious logic.
- **Docstrings:** One-line for most functions. Only elaborate when truly needed.
- **Dicts:** Comment key→value types if not obvious, e.g. `# lemma -> frequency`

### Error Handling
- **Fail Fast:** Let errors crash. NO unnecessary try/except.
- **Warnings over Catches:** Prefer logging warnings over catching errors.

### Logging
- **Sparse:** Only log what's essential for understanding the flow.

---

## Project Overview

Spanish vocabulary frequency analyzer and word sense disambiguation system that processes large text corpora using spaCy for contextual lemmatization. The pipeline transforms raw Spanish text into a disambiguated, semantically-enriched vocabulary dataset through eight stages:

1. **Frequency Analysis**: Contextual lemmatization and frequency counting with caching
2. **Post-Processing**: POS filtering, mapping, and lemma cleaning
3. **LLM Correction**: Optional LLM-based lemmatization error correction
4. **Translation**: Optional LLM-based translation to German with multiple meanings (without examples)
5. **Example Generation**: LLM-based diverse example sentences per meaning (when disambiguation enabled)
6. **Meaning Embeddings**: BERT embeddings for each meaning variant
7. **Word Sense Disambiguation**: Corpus-wide disambiguation using embedding similarity
8. **Co-Occurrence Matrix**: Semantic co-occurrence matrix for vocabulary relationships

The system supports multiple corpora, aggressive caching at each stage, and streaming processing for memory efficiency. Primary use case: vocabulary learning applications requiring disambiguated word meanings and semantic relationships.

## Development Commands

### Conda Environment
The project uses a conda environment called `LanguageApp-new`. Run commands with:
```bash
# Run a Python script
~/anaconda3/bin/conda run -n LanguageApp-new python3 script.py

# Run inline Python code (set PYTHONPATH for imports from project root)
~/anaconda3/bin/conda run -n LanguageApp-new python3 -c "
import sys
sys.path.insert(0, '.')
from src.vocab import Vocabulary
# ... your code
"
```

### Running the Pipeline
```bash
# Main execution - processes corpora and generates frequency lists
~/anaconda3/bin/conda run -n LanguageApp-new python3 main.py
```

### Testing
```bash
# Test lemmatization functionality
~/anaconda3/bin/conda run -n LanguageApp-new python3 test_lemmatization.py
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
2. **SentenceStore** (src/stores/sentence_store.py) → stores original tokens in `SentenceStore.jsonl.gz`
3. **Lemmatizer** (src/lemmatizer.py) → yields ALL tokens with original POS tags and lemmas
4. **LemmatizedStore** (src/stores/lemmatized_store.py) → stores lemmatized tokens in `LemmatizedStore.jsonl.gz`
5. **FrequencyCounter** (src/frequency_counter.py) → aggregates raw frequencies (lemma|ORIGINAL_POS)
6. **Cache** → stores raw, unfiltered frequency data in `FrequencyCounter.pkl`

**Stage 2: Post-Processing (after cache)**
7. **PostProcessor** (src/post_processor.py) → applies POS filtering, mapping, lemma cleaning
8. **OutputWriter** (src/output_writer.py) → writes filtered CSV results

**Stage 3: LLM Correction (optional, with caching)**
9. **LemmaCorrector** (src/llm_corrector.py) → sends lemmas to LLM in batches for validation
   - Identifies lemmatization errors (e.g., incorrect conjugations)
   - Supports three operations: KEEP, DELETE, MERGE
   - Applies corrections to FrequencyCounter (removes deleted, merges variants)
   - Saves global correction mapping to output directory → `correction_mapping.pkl`
   - Mapping linked to all LemmatizedStores (no duplication)
   - Corrected tokens generated dynamically via `iter_corrected()`
   - Cache key includes model config + system prompt hash
   - Outputs: `original_with_ids.csv`, `llm_corrections.csv`, `correction_mapping.pkl`

**Stage 4: Translation (optional, with caching)**
10. **LemmaTranslator** (src/llm_translator.py) → translates Spanish lemmas to German
    - Processes lemmas by POS category (VERB, NOUN, ADJ, ADV, ADP, CONJ)
    - Uses POS-specific prompts for accurate translations
    - Generates multiple meanings with definitions and estimated frequency share
    - Does NOT generate example sentences (handled by Stage 5)
    - Cache key includes model config + combined prompt hash (general + POS-specific)
    - Outputs: `translations.json` with enriched frequency data (no examples yet)

**Stage 5: Example Generation (optional, requires translation, runs when disambiguation enabled)**
11. **ExampleGenerator** (src/example_generator.py) → generates diverse example sentences
    - Generates 1 example for monosemantic words (1 meaning)
    - Generates `num_examples_poly` (default: 10) examples for polysemantic words (2+ meanings)
    - Uses separate LLM provider (can be cheaper model than translation)
    - Cache key includes model config + prompt hash + definition_hash
    - Extracts target word positions for BERT embedding
    - Outputs: Updated `translations.json` with `examples_es` per meaning

**Stage 6: Meaning Embeddings (optional, requires examples)**
12. **MeaningEmbeddingGenerator** (src/meaning_embedding_generator.py) → generates BERT embeddings
    - Creates embeddings for each meaning's multiple example sentences
    - Uses target word masking to extract context embeddings
    - Averages embeddings across all examples for robust meaning representation
    - Adds `meaning_embedding` field (averaged) to each meaning in `translations.json`

**Stage 7: Word Sense Disambiguation (optional, requires embeddings + stores)**
13. **Vocabulary** (src/vocab.py) → builds from translations
    - Creates WordVariant objects (one per meaning)
    - Provides lookup structures for disambiguation
14. **BatchDisambiguator** (src/batch_disambiguator.py) → disambiguates corpus tokens
    - Processes ALL corpora (multi-corpus support)
    - Compares runtime token embeddings with pre-computed meaning embeddings
    - Uses cosine similarity to select best meaning variant
    - Processes corrected lemmas if available, otherwise uses original lemmas
    - Writes variant IDs to DisambiguatedStore → `DisambiguatedStore.jsonl.gz` (per corpus)
    - Includes caching: checks if disambiguation already exists for vocabulary + BERT model

**Stage 8: Co-Occurrence Matrix (optional, requires disambiguation)**
15. **CoOccurrenceBuilder** (src/cooccurrence_builder.py) → builds co-occurrence matrix
    - Processes ALL disambiguated stores from all corpora
    - Creates sparse co-occurrence matrix across all corpora
    - Outputs: `cooccurrence_matrix.npz`

This separation enables cache reusability across different configurations and progressive enhancement of the data through the pipeline stages.

### Key Components

**CorpusProcessor** (src/corpus_processor.py)
- Orchestrates processing for one corpus
- Implements caching using pickle with hash-based keys (corpus filename, mtime, model, max_sentences)
- Returns tuple: (SentenceStore, LemmatizedStore, FrequencyCounter)
- Cache structure: `cache/{corpus_name}/{cache_key}/` with:
  - `SentenceStore.jsonl.gz` (sentence tokens)
  - `LemmatizedStore.jsonl.gz` (lemmatized tokens)
  - `FrequencyCounter.pkl` (frequency data)

**Main Pipeline** (main.py)
- Processes multiple corpora sequentially
- Merges frequency counts across all corpora using `FrequencyCounter.merge()`
- Manages global correction mapping (stored once in output directory)
- Orchestrates disambiguation across ALL corpora

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
  - Estimated frequency share per meaning
- Does NOT generate example sentences (handled by ExampleGenerator)
- Cache key combines model config + general prompt hash + POS prompt hash
- Normalizes `estimated_share` values to sum to 1.0
- Tracks token usage statistics (prompt/completion tokens, API calls, cache hits)

**ExampleGenerator** (src/example_generator.py)
- Generates diverse example sentences for translations
- Monosemantic words (1 meaning): 1 example
- Polysemantic words (2+ meanings): `num_examples_poly` examples per meaning
- Uses separate LLM provider (can be cheaper/faster model)
- Cache key: `{lemma}|{pos}|{definition_hash}_{model_hash}_{prompt_hash}`
- Extracts bracketed target words and computes character positions
- Adds `examples_es`, `_target_word`, `_target_starts`, `_target_ends` to meanings

**LemmaCache** (src/lemma_cache.py)
- Unified cache for correction, translation, and example results
- SQLite database: `cache/lemma_cache.db`
- Tables: `corrections`, `translations`, `examples`
- Cache key format: `{lemma}|{pos}_{model_hash}_{prompt_hash}.json`
- Model hash includes: provider type, model name, temperature, max_tokens
- Separate prompt hashes for correction vs translation
- In-memory buffer with periodic disk writes (`save_all()`)

**BaseStore** (src/stores/base_store.py)
- Abstract base class for all store types
- Provides common interface: `__iter__()`, `iter_batched()`, `open_writer()`, `exists()`, `count()`
- Uses generic type parameter for type safety
- All stores use compressed JSONL format (.jsonl.gz)

**SentenceStore** (src/stores/sentence_store.py)
- Stores original sentence tokens
- Format: `{"sid": int, "tokens": ["El", "gato", ...]}`

**LemmatizedStore** (src/stores/lemmatized_store.py)
- Stores lemmatized tokens (lemma, pos) tuples
- Format: `{"sid": int, "tokens": [["el", "DET"], ["gato", "NOUN"], ...]}`
- Supports correction mapping (loaded from global file in output directory)
- `iter_corrected()` applies mapping dynamically (no separate file needed)
- Parallel iteration methods: `iter_with_sentences()`, `iter_corrected_with_sentences()`

**DisambiguatedStore** (src/stores/disambiguated_store.py)
- Stores word variant IDs after disambiguation
- Format: `{"sid": int, "variants": ["000123_0", null, "000456_1", ...]}`

**Index-Aligned Architecture:**
- All stores maintain same sentence ID order
- Token lists have identical length and order per sentence
- Enables efficient parallel iteration without re-indexing

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
- Takes SentenceStore and LemmatizedStore as input
- Returns DisambiguatedStore
- Supports caching: checks if disambiguation already exists

**CoOccurrenceBuilder** (src/cooccurrence_builder.py)
- Builds co-occurrence matrix from multiple disambiguated stores
- Takes List[DisambiguatedStore] (supports multi-corpus)
- Uses sparse matrix format for memory efficiency
- Counts variant co-occurrences within sentences across all corpora
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
- `cache`: directory, llm_directory
- `filtering`: accepted_pos, pos_mappings, include_countries
- `llm_correction`: enabled, batch_size, prompt_file, slice, llm provider config
- `llm_translation`: enabled, batch_size, prompt_directory, num_examples, llm provider config
- `disambiguation`: enabled, model (BERT model name), batch_size

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
└── vocabulary/              # Vocabulary directory (if disambiguation enabled)
    ├── translations.json    # Vocabulary translations
    └── cooccurrence_matrix.npz  # Co-occurrence matrix
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
        "examples_es": [
          "El ser humano piensa.",
          "¿Quién eres tú?",
          "Somos estudiantes.",
          "Era una noche oscura.",
          "Serán las cinco de la tarde."
        ],
        "flags": ["common"],
        "meaning_embedding": [0.123, -0.456, ...],  // 768-dim averaged BERT vector
        "_target_word": "ser",
        "_target_starts": [3, 7, 0, 0, 0],
        "_target_ends": [6, 11, 5, 3, 5]
      },
      {
        "definition_es": "Pertenecer, formar parte de",
        "estimated_share": 0.3,
        "translations": [
          {"text": "sein", "hint": "zugehören"}
        ],
        "examples_es": ["...", "...", "...", "...", "..."],
        "meaning_embedding": [0.234, -0.567, ...]
      }
    ]
  }
]
```

### Store Structure

The pipeline creates separate store files per corpus:
```
cache/{corpus_name}/{cache_key}/
├── SentenceStore.jsonl.gz         # Original tokens: {"sid": 0, "tokens": ["El", "gato", ...]}
├── LemmatizedStore.jsonl.gz       # Lemmas: {"sid": 0, "tokens": [["el","DET"], ["gato","NOUN"], ...]}
├── FrequencyCounter.pkl           # Raw frequency data (pickled dict)
└── {disambiguation_cache_key}/
    └── DisambiguatedStore.jsonl.gz  # Variant IDs: {"sid": 0, "variants": ["000123_0", null, ...]}
```

**Global Correction Mapping** (if llm_correction enabled):
```
output/spanish_lemma_frequencies/YYYYMMDD_HHMMSS/
└── correction_mapping.pkl         # Stored once, referenced by all LemmatizedStores
```

**Efficiency Design:**
- `correction_mapping.pkl` stored globally (not per corpus)
- `iter_corrected()` applies mapping dynamically when reading (no separate file)
- Saves disk space: O(unique corrections) vs O(total sentences × num_corpora)
- All stores are index-aligned: same sentence ID, same token count, same order

### Memory Efficiency Design

The entire pipeline uses generators and streaming to process arbitrarily large corpora without loading full datasets:
- CorpusReader yields sentence batches
- Lemmatizer yields token tuples via `nlp.pipe()`
- FrequencyCounter updates incrementally (logs every 100k tokens)
- All stores use streaming writers (compressed JSONL via BaseStoreWriter)
- All store readers are generators (no full load into memory)
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

**Disambiguation Cache**:
Cache key format: `SHA256(vocabulary_hash | bert_model_name)[:12]`

Vocabulary hash includes:
- Lemma, POS, and number of meanings for each translation
- Sorted for deterministic ordering

Cache invalidation occurs when:
- Vocabulary changes (different translations or meanings)
- BERT model changes

This enables efficient caching: same vocabulary + BERT model = reuse disambiguation results

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
- Saves global `correction_mapping.pkl` in output directory (referenced by all LemmatizedStores)

**Stage 4: Translation**
- Optional (enabled via `llm_translation.enabled`)
- Requires: Frequency data (corrected or original)
- Outputs: `translations.json` (without examples)

**Stage 5: Example Generation**
- Runs automatically when `disambiguation.enabled=true`
- Requires: `translations.json` from Stage 4
- Uses separate LLM provider (configurable in `example_generation.llm`)
- Monosemantic words (1 meaning): 1 example
- Polysemantic words (2+ meanings): `num_examples_poly` examples per meaning
- Outputs: Updated `translations.json` with `examples_es`

**Stage 6: Meaning Embeddings**
- Runs automatically when `disambiguation.enabled=true`
- Requires: `translations.json` with examples from Stage 5
- Modifies: Adds `meaning_embedding` field to each meaning in `translations.json`

**Stage 7: Word Sense Disambiguation**
- Optional (enabled via `disambiguation.enabled`)
- Requires:
  - `translations.json` with embeddings from Stage 6
  - SentenceStore and LemmatizedStore from all corpora
- Processes ALL corpora (multi-corpus support)
- Outputs: `DisambiguatedStore.jsonl.gz` per corpus in `cache/{corpus_name}/{cache_key}/{disambiguation_cache_key}/`
- Includes caching based on vocabulary hash + BERT model

**Stage 8: Co-Occurrence Matrix**
- Runs automatically when disambiguation is enabled
- Requires DisambiguatedStore from ALL corpora
- Builds matrix across all corpora
- Outputs: `vocabulary/cooccurrence_matrix.npz` in output directory

**Dependency Chain Summary:**
```
Stage 1-2 (always) → Stage 3 (optional) → Stage 4 (optional) → Stage 5-6 (auto with Stage 7)
                                                                             ↓
                          SentenceStore + LemmatizedStore (always) → Stage 7 (optional) → Stage 8 (auto with Stage 7)
```

## Data Expectations

**Leipzig Corpora:** Place files in `data/Leipzig_Wortschatz/{corpus_name}/{corpus_name}-sentences.txt` format.
**OPUS Corpora:** Place .txt files with one sentence per line anywhere (path specified in config.yaml).