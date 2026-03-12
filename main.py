#!/usr/bin/env python3
"""
Lemma Frequency Analysis Pipeline

Analyzes text corpora, lemmatizes words in context, and generates frequency lists.
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from scipy import sparse

from src.batch_disambiguator import BatchDisambiguator
from src.cooccurrence_builder import CoOccurrenceBuilder
from src.corpus_processor import CorpusProcessor
from src.corpus_readers import LeipzigCorpusReader, OpusCorpusReader
from src.frequency_counter import FrequencyCounter
from src.lemma_cache import LemmaCache
from src.lemmatizer import Lemmatizer
from src.llm_corrector import LemmaCorrector
from src.llm_providers import create_provider
from src.llm_translator import LemmaTranslator
from src.example_generator import ExampleGenerator
from src.meaning_embedding_generator import MeaningEmbeddingGenerator
from src.lemma_output_writer import LemmaFrequencyOutputWriter
from src.post_processor import FrequencyPostProcessor
from src.stores.disambiguated_store import DisambiguatedStore
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.sentence_store import SentenceStore
from src.vocab import Translation, Vocabulary
from src.share_updater import ShareUpdater

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result container for pipeline execution."""

    output_dir: str
    vocabulary: Optional["Vocabulary"] = None
    translations: Optional[List["Translation"]] = None
    cooccurrence_matrix: Optional[sparse.csr_matrix] = None
    merged_counter: Optional["FrequencyCounter"] = None
    # corpus_name -> (SentenceStore, LemmatizedStore, cache_key)
    corpus_stores: Dict[str, Tuple["SentenceStore", "LemmatizedStore", str]] = field(
        default_factory=dict
    )
    # corpus_name -> DisambiguatedStore
    disambiguated_stores: Dict[str, "DisambiguatedStore"] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Shared state passed between pipeline stages."""

    config: dict
    output_dir: str
    cache_config: dict
    filter_config: dict
    disambiguation_config: dict
    cache_only: bool
    min_frequency: int

    # Populated during pipeline
    lemmatizer: Optional[Lemmatizer] = None
    merged_counter: Optional[FrequencyCounter] = None
    corpus_stores: Dict[str, Tuple[SentenceStore, LemmatizedStore, str]] = field(
        default_factory=dict
    )
    translations: Optional[List[Translation]] = None
    vocabulary: Optional[Vocabulary] = None
    disambiguated_stores: Dict[str, DisambiguatedStore] = field(default_factory=dict)
    cooccurrence_matrix: Optional[sparse.csr_matrix] = None
    correction_mapping_path: Optional[Path] = None
    llm_cache: Optional[LemmaCache] = None


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_vocabulary_hash(translations: List[Translation]) -> str:
    """Generate hash from vocabulary for disambiguation cache key."""
    hash_data = []
    for t in sorted(translations, key=lambda x: (x.lemma, x.pos)):
        hash_data.append(f"{t.lemma}|{t.pos}|{len(t.meanings)}")
    content = "|".join(hash_data)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def generate_disambiguation_cache_key(vocab_hash: str, bert_model: str) -> str:
    """Generate cache key for disambiguation."""
    content = f"{vocab_hash}|{bert_model}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _ensure_llm_cache(ctx: PipelineContext) -> LemmaCache:
    """Lazy-initialize LLM cache on context."""
    if ctx.llm_cache is None:
        llm_cache_dir = ctx.cache_config.get("llm_directory", "cache/llm_results")
        ctx.llm_cache = LemmaCache({"directory": llm_cache_dir})
    return ctx.llm_cache


def _setup_pipeline(
    config: dict,
    config_path: Optional[str],
    translations_override: Optional[List[Translation]],
) -> PipelineContext:
    """Create output dir, configure logging, initialize context."""
    output_config = config["output"]
    lemma_config = config["lemmatization"]
    cache_config = config.get("cache", {})
    cache_only = config.get("cache_only", False)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = output_config["directory"]
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging to file and console
    log_file_path = os.path.join(output_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    console_handler = logging.StreamHandler()

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Copy config file to output directory (if path provided)
    if config_path is not None:
        config_backup_path = os.path.join(output_dir, "config.yaml")
        shutil.copy2(config_path, config_backup_path)
        logger.info(f"Config copied to: {config_backup_path}")
    logger.info(f"Log file: {log_file_path}")

    logger.info("Starting pipeline...")
    if cache_only:
        logger.info("Cache-only mode: LLM API calls will be skipped")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {lemma_config['model']}")

    # Initialize lemmatizer
    lemmatizer = Lemmatizer(
        language_model=lemma_config["model"],
        batch_size=lemma_config.get("batch_size", 1000),
    )

    ctx = PipelineContext(
        config=config,
        output_dir=output_dir,
        cache_config=cache_config,
        filter_config=config.get("filtering", {}),
        disambiguation_config=config.get("disambiguation", {"enabled": False}),
        cache_only=cache_only,
        min_frequency=output_config.get("min_frequency", 1),
        lemmatizer=lemmatizer,
        merged_counter=FrequencyCounter(),
        correction_mapping_path=Path(output_dir) / "correction_mapping.pkl",
    )

    # Handle translations override
    if translations_override is not None:
        logger.info("=" * 60)
        logger.info("STAGE 4-6: Using provided translations (skipping LLM stages)")
        logger.info("=" * 60)
        ctx.translations = translations_override

    return ctx


def _stage_1_2_process_corpora(ctx: PipelineContext) -> None:
    """Process all corpora, collect stores, apply post-processing."""
    logger.info("=" * 60)
    logger.info("STAGE 1-2: Processing corpora")
    logger.info("=" * 60)

    corpora_configs = ctx.config["corpora"]
    processing_config = ctx.config["processing"]
    logger.info(f"Processing {len(corpora_configs)} corpora")

    for corpus_config in corpora_configs:
        corpus_type = corpus_config.get("type", "leipzig").lower()
        max_sentences = corpus_config.get("max_sentences", None)
        encoding = corpus_config.get("encoding", "utf-8")

        if corpus_type == "leipzig":
            reader = LeipzigCorpusReader(
                file_path=corpus_config["file_path"],
                encoding=encoding,
                max_sentences=max_sentences,
            )
        elif corpus_type == "opus":
            reader = OpusCorpusReader(
                file_path=corpus_config["file_path"],
                encoding=encoding,
                max_sentences=max_sentences,
            )
        else:
            raise ValueError(
                f"Unknown corpus type: {corpus_type}. Supported types: 'leipzig', 'opus'"
            )

        processor = CorpusProcessor(
            corpus_config=corpus_config,
            reader=reader,
            lemmatizer=ctx.lemmatizer,
            processing_config=processing_config,
            cache_config=ctx.cache_config,
        )

        sentence_store, lemmatized_store, counter = processor.process()
        cache_key = processor.get_cache_key()

        corpus_name = corpus_config.get("name", Path(corpus_config["file_path"]).stem)
        ctx.corpus_stores[corpus_name] = (sentence_store, lemmatized_store, cache_key)

        ctx.merged_counter.merge(counter)
        logger.info(f"Processed {corpus_name}: {counter.total_tokens:,} tokens")

    # Apply post-processing to merged counter
    if ctx.filter_config:
        logger.info("Applying post-processing filters to merged data...")
        post_processor = FrequencyPostProcessor(ctx.filter_config)
        ctx.merged_counter = post_processor.process(ctx.merged_counter)

    # Set POS mapping on all LemmatizedStores
    pos_mappings = ctx.filter_config.get("pos_mappings", {})
    if pos_mappings:
        for _, (_, lemma_store, _) in ctx.corpus_stores.items():
            lemma_store.set_pos_mapping(pos_mappings)
        logger.info(f"POS mapping set on {len(ctx.corpus_stores)} LemmatizedStores")

    # Output raw frequencies
    raw_frequencies = ctx.merged_counter.get_sorted_frequencies(
        min_frequency=ctx.min_frequency
    )
    raw_output_path = os.path.join(ctx.output_dir, "frequencies_raw.csv")
    LemmaFrequencyOutputWriter.write_csv(raw_frequencies, raw_output_path)
    logger.info(f"Raw frequencies saved to: {raw_output_path}")


def _stage_3_correction(ctx: PipelineContext) -> None:
    """Apply LLM-based lemma correction."""
    logger.info("=" * 60)
    logger.info("STAGE 3: LLM Correction")
    logger.info("=" * 60)

    correction_config = ctx.config["llm_correction"]
    llm_cache = _ensure_llm_cache(ctx)

    # Load system prompt
    prompt_file = correction_config.get(
        "prompt_file", "prompts/correction/general_prompt.txt"
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    if not system_prompt or not system_prompt.strip():
        raise ValueError(
            f"System prompt is empty or could not be loaded from: {prompt_file}"
        )

    # Create provider and corrector
    provider_type = correction_config["llm"]["type"]
    provider_config = correction_config["llm"].get(provider_type, {})
    correction_provider = create_provider(provider_type, provider_config)

    corrector_config = {
        "batch_size": correction_config.get("batch_size", 50),
        "system_prompt": system_prompt,
        "output_dir": ctx.output_dir,
        "target_output_size": correction_config.get("target_output_size"),
        "max_requests_per_minute": correction_config.get("max_requests_per_minute", 0),
        "max_concurrent_requests": correction_config.get("max_concurrent_requests", 5),
        "cache_only": ctx.cache_only,
    }
    corrector = LemmaCorrector(correction_provider, llm_cache, corrector_config)

    # Apply corrections
    ctx.merged_counter, correction_mapping, deleted_tokens = corrector.correct(
        ctx.merged_counter, min_frequency=ctx.min_frequency
    )

    # Save global correction mapping
    mapping_data = {
        "correction_mapping": correction_mapping,
        "deleted_tokens": deleted_tokens,
    }
    with open(ctx.correction_mapping_path, "wb") as f:
        pickle.dump(mapping_data, f)
    logger.info(f"Global correction mapping saved to: {ctx.correction_mapping_path}")

    # Link correction mapping to ALL LemmatizedStores
    for _, (_, lemma_store, _) in ctx.corpus_stores.items():
        lemma_store.set_correction_mapping(
            correction_mapping, deleted_tokens, ctx.correction_mapping_path
        )
    logger.info(
        f"Correction mapping linked to {len(ctx.corpus_stores)} LemmatizedStores"
    )

    logger.info("LLM correction completed successfully")

    # Output corrected frequencies to CSV
    stats = ctx.merged_counter.get_statistics()
    logger.info(
        f"\nFinal statistics:\n"
        f"  Total tokens: {stats['total_tokens']:,}\n"
        f"  Unique lemmas: {stats['unique_lemmas']:,}"
    )
    frequencies = ctx.merged_counter.get_sorted_frequencies(
        min_frequency=ctx.min_frequency
    )
    corrected_output_path = os.path.join(ctx.output_dir, "frequencies_corrected.csv")
    LemmaFrequencyOutputWriter.write_csv(frequencies, corrected_output_path)
    logger.info(f"Corrected frequencies saved to: {corrected_output_path}")


def _stage_4_translation(ctx: PipelineContext) -> None:
    """Translate lemmas using LLM."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Translation")
    logger.info("=" * 60)

    translation_config = ctx.config["llm_translation"]
    llm_cache = _ensure_llm_cache(ctx)

    provider_type = translation_config["llm"]["type"]
    provider_config = translation_config["llm"].get(provider_type, {})
    translation_provider = create_provider(provider_type, provider_config)

    translator_config = {
        "batch_size": translation_config.get("batch_size", 25),
        "prompts_dir": translation_config.get(
            "prompt_directory", "prompts/translation"
        ),
        "output_dir": ctx.output_dir,
        "max_requests_per_minute": translation_config.get("max_requests_per_minute", 0),
        "max_concurrent_requests": translation_config.get("max_concurrent_requests", 5),
        "cache_only": ctx.cache_only,
    }

    translator = LemmaTranslator(translation_provider, llm_cache, translator_config)
    ctx.translations, _ = translator.translate(
        ctx.merged_counter, min_frequency=ctx.min_frequency
    )

    logger.info("LLM translation completed")


def _stage_5_examples(ctx: PipelineContext) -> None:
    """Generate example sentences for translations."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Example Generation")
    logger.info("=" * 60)

    example_config = ctx.config.get("example_generation", {})
    llm_cache = _ensure_llm_cache(ctx)

    example_provider_type = example_config.get("llm", {}).get("type", "google")
    example_provider_config = example_config.get("llm", {}).get(
        example_provider_type, {}
    )
    example_provider = create_provider(example_provider_type, example_provider_config)

    example_gen_config = {
        "batch_size": example_config.get("batch_size", 50),
        "prompts_dir": example_config.get("prompt_directory", "prompts/examples"),
        "num_examples_poly": example_config.get("num_examples_poly", 10),
        "max_requests_per_minute": example_config.get("max_requests_per_minute", 0),
        "max_concurrent_requests": example_config.get("max_concurrent_requests", 5),
        "cache_only": ctx.cache_only,
    }

    disambiguated_pos = set(
        ctx.disambiguation_config.get("pos_types", ["NOUN", "VERB", "ADJ", "ADV"])
    )
    example_generator = ExampleGenerator(
        example_provider, llm_cache, example_gen_config, disambiguated_pos
    )
    ctx.translations, _ = example_generator.generate(ctx.translations)
    logger.info("Example generation completed")


def _stage_6_embeddings(ctx: PipelineContext) -> None:
    """Generate BERT meaning embeddings."""
    logger.info("=" * 60)
    logger.info("STAGE 6: Meaning Embeddings")
    logger.info("=" * 60)

    model_name = ctx.disambiguation_config.get(
        "model", "dccuchile/bert-base-spanish-wwm-cased"
    )
    meaning_emb_generator = MeaningEmbeddingGenerator(
        model_name=model_name,
        bert_batch_tokens=ctx.disambiguation_config.get("bert_batch_tokens", 4096),
    )
    ctx.translations = meaning_emb_generator.generate(ctx.translations)
    logger.info("Meaning embeddings generated")


def _save_translations(
    ctx: PipelineContext, filename: str = "translations.json"
) -> None:
    """Save translations to JSON."""
    if ctx.translations is None:
        return

    translations_path = os.path.join(ctx.output_dir, filename)
    translations_dicts = [t.to_dict() for t in ctx.translations]
    with open(translations_path, "w", encoding="utf-8") as f:
        json.dump(translations_dicts, f, ensure_ascii=False, indent=2)
    logger.info(f"Translations saved to: {translations_path}")


def _stage_7_disambiguation(ctx: PipelineContext) -> None:
    """Disambiguate word senses across all corpora."""
    logger.info("=" * 60)
    logger.info("STAGE 7: Disambiguation")
    logger.info("=" * 60)

    # Build vocabulary
    disambiguated_pos = set(
        ctx.disambiguation_config.get("pos_types", ["NOUN", "VERB", "ADJ", "ADV"])
    )
    ctx.vocabulary = Vocabulary.from_translations_list(
        ctx.translations, disambiguated_pos=disambiguated_pos
    )
    vocab_hash = generate_vocabulary_hash(ctx.translations)
    bert_model = ctx.disambiguation_config.get(
        "model", "dccuchile/bert-base-spanish-wwm-cased"
    )
    disambiguation_cache_key = generate_disambiguation_cache_key(vocab_hash, bert_model)

    disambiguator = BatchDisambiguator(
        vocabulary=ctx.vocabulary,
        model_name=bert_model,
        bert_batch_tokens=ctx.disambiguation_config.get("bert_batch_tokens", 4096),
    )

    for corpus_name, (sent_store, lemma_store, cache_key) in ctx.corpus_stores.items():
        logger.info(f"Disambiguating: {corpus_name}")

        # Cache path for this corpus's disambiguation
        cache_base = Path(ctx.cache_config.get("directory", "cache"))
        corpus_cache_dir = cache_base / corpus_name / cache_key
        output_path = (
            corpus_cache_dir / disambiguation_cache_key / "DisambiguatedStore.jsonl.gz"
        )

        # Check if already cached
        if output_path.exists():
            logger.info(f"  Loading from cache: {output_path}")
            dis_store = DisambiguatedStore(output_path)
        else:
            logger.info(f"  Processing...")
            dis_store = disambiguator.process(
                sent_store,
                lemma_store,
                output_path,
                use_corrections=ctx.correction_mapping_path.exists(),
            )

        ctx.disambiguated_stores[corpus_name] = dis_store

    logger.info(f"Disambiguation complete for {len(ctx.disambiguated_stores)} corpora")


def _stage_7_5_share_update(ctx: PipelineContext) -> None:
    """Update estimated_share values from actual corpus usage."""
    logger.info("=" * 60)
    logger.info("STAGE 7.5: Updating estimated shares from corpus usage")
    logger.info("=" * 60)

    share_updater = ShareUpdater()
    ctx.translations = share_updater.update(
        list(ctx.disambiguated_stores.values()), ctx.translations, ctx.vocabulary
    )
    # Note: Do NOT recreate vocabulary here - IDs must stay stable for
    # DisambiguatedStores and Co-Occurrence Matrix consistency
    logger.info("Estimated shares updated based on corpus usage")


def _stage_8_cooccurrence(ctx: PipelineContext) -> None:
    """Build co-occurrence matrix from disambiguated stores."""
    logger.info("=" * 60)
    logger.info("STAGE 8: Co-Occurrence Matrix")
    logger.info("=" * 60)

    builder = CoOccurrenceBuilder(ctx.vocabulary)
    ctx.cooccurrence_matrix = builder.build(list(ctx.disambiguated_stores.values()))
    logger.info("Co-occurrence matrix built successfully")


def _save_vocabulary(ctx: PipelineContext) -> None:
    """Save vocabulary in light and full versions."""
    if ctx.cooccurrence_matrix is not None:
        ctx.vocabulary.cooccurrence_matrix = ctx.cooccurrence_matrix

    # Save lightweight version (for quick inspection)
    vocab_dir_light = Path(ctx.output_dir) / "vocabulary"
    ctx.vocabulary.save(
        str(vocab_dir_light),
        include_embeddings=False,
        include_private=False,
        save_cooccurrence=False,
    )
    logger.info(f"Vocabulary (light) saved to: {vocab_dir_light}")

    # Save full version (with embeddings and cooccurrence matrix)
    vocab_dir_full = Path(ctx.output_dir) / "vocabulary_full"
    ctx.vocabulary.save(
        str(vocab_dir_full),
        include_embeddings=True,
        include_private=True,
        save_cooccurrence=True,
    )
    logger.info(f"Vocabulary (full) saved to: {vocab_dir_full}")


def run_pipeline(
    config: dict,
    config_path: Optional[str] = None,
    translations_override: Optional[List[Translation]] = None,
) -> PipelineResult:
    """
    Run the full lemma frequency analysis pipeline.

    Args:
        config: Configuration dictionary
        config_path: Optional path to config file (for backup copy to output dir)
        translations_override: Optional pre-built translations (skips LLM stages 3-5)

    Returns:
        PipelineResult with all outputs
    """
    ctx = _setup_pipeline(config, config_path, translations_override)

    _stage_1_2_process_corpora(ctx)

    if ctx.config.get("llm_correction", {}).get("enabled", False):
        _stage_3_correction(ctx)

    if ctx.translations is None:
        if ctx.config.get("llm_translation", {}).get("enabled", False):
            _stage_4_translation(ctx)
            _save_translations(ctx, "translations.json")

    disambiguation_enabled = ctx.disambiguation_config.get("enabled", False)

    if ctx.translations is not None and disambiguation_enabled:
        _stage_5_examples(ctx)
        _stage_6_embeddings(ctx)

    _save_translations(ctx, filename="translations_with_embeddings.json")

    if ctx.translations is not None and disambiguation_enabled:
        _stage_7_disambiguation(ctx)
        _stage_7_5_share_update(ctx)
        _save_translations(ctx, filename="translations_disambiguated.json")
        _stage_8_cooccurrence(ctx)

    if ctx.vocabulary is not None:  # Vocabulary is built during stage 7 disambiguation
        _save_vocabulary(ctx)

    logger.info("=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)

    return PipelineResult(
        output_dir=ctx.output_dir,
        vocabulary=ctx.vocabulary,
        translations=ctx.translations,
        cooccurrence_matrix=ctx.cooccurrence_matrix,
        merged_counter=ctx.merged_counter,
        corpus_stores=ctx.corpus_stores,
        disambiguated_stores=ctx.disambiguated_stores,
    )


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    run_pipeline(config, config_path=config_path)
