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
    lemma_config = config["lemmatization"]
    processing_config = config["processing"]
    output_config = config["output"]
    cache_config = config.get("cache", {})
    filter_config = config.get("filtering", {})
    disambiguation_config = config.get("disambiguation", {"enabled": False})
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

    # =========================================================================
    # STAGE 1-2: Process all corpora, collect stores
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1-2: Processing corpora")
    logger.info("=" * 60)

    corpora_configs = config["corpora"]
    logger.info(f"Processing {len(corpora_configs)} corpora")

    corpus_stores: Dict[str, Tuple[SentenceStore, LemmatizedStore, str]] = {}
    merged_counter = FrequencyCounter()

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
            lemmatizer=lemmatizer,
            processing_config=processing_config,
            cache_config=cache_config,
        )

        sentence_store, lemmatized_store, counter = processor.process()
        cache_key = processor.get_cache_key()

        corpus_name = corpus_config.get("name", Path(corpus_config["file_path"]).stem)
        corpus_stores[corpus_name] = (sentence_store, lemmatized_store, cache_key)

        merged_counter.merge(counter)
        logger.info(f"Processed {corpus_name}: {counter.total_tokens:,} tokens")

    # Apply post-processing to merged counter
    if filter_config:
        logger.info("Applying post-processing filters to merged data...")
        processor = FrequencyPostProcessor(filter_config)
        merged_counter = processor.process(merged_counter)

    # Set POS mapping on all LemmatizedStores (needed for correction lookup and disambiguation)
    pos_mappings = filter_config.get("pos_mappings", {})
    if pos_mappings:
        for corpus_name, (sent_store, lemma_store, cache_key) in corpus_stores.items():
            lemma_store.set_pos_mapping(pos_mappings)
        logger.info(f"POS mapping set on {len(corpus_stores)} LemmatizedStores")

    # Output raw frequencies (before LLM correction)
    min_freq = output_config.get("min_frequency", 1)
    raw_frequencies = merged_counter.get_sorted_frequencies(min_frequency=min_freq)
    raw_output_path = os.path.join(output_dir, "frequencies_raw.csv")
    LemmaFrequencyOutputWriter.write_csv(raw_frequencies, raw_output_path)
    logger.info(f"Raw frequencies saved to: {raw_output_path}")

    # =========================================================================
    # STAGE 3: LLM Correction (global mapping)
    # =========================================================================
    correction_mapping_path = Path(output_dir) / "correction_mapping.pkl"
    correction_mapping = None
    deleted_tokens = None

    correction_config = config.get("llm_correction", {})
    if correction_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 3: LLM Correction")
        logger.info("=" * 60)

        # Initialize LLM cache
        llm_cache_dir = cache_config.get("llm_directory", "cache/llm_results")
        llm_cache = LemmaCache({"directory": llm_cache_dir})

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
            "output_dir": output_dir,
            "target_output_size": correction_config.get("target_output_size"),
            "max_requests_per_minute": correction_config.get(
                "max_requests_per_minute", 0
            ),
            "max_concurrent_requests": correction_config.get(
                "max_concurrent_requests", 5
            ),
            "cache_only": cache_only,
        }
        corrector = LemmaCorrector(correction_provider, llm_cache, corrector_config)

        # Apply corrections
        min_freq = output_config.get("min_frequency", 1)
        merged_counter, correction_mapping, deleted_tokens = corrector.correct(
            merged_counter, min_frequency=min_freq
        )

        # Save global correction mapping
        mapping_data = {
            "correction_mapping": correction_mapping,
            "deleted_tokens": deleted_tokens,
        }
        with open(correction_mapping_path, "wb") as f:
            pickle.dump(mapping_data, f)
        logger.info(f"Global correction mapping saved to: {correction_mapping_path}")

        # Link correction mapping to ALL LemmatizedStores
        for corpus_name, (
            sent_store,
            lemma_store,
            cache_key,
        ) in corpus_stores.items():
            lemma_store.set_correction_mapping(
                correction_mapping, deleted_tokens, correction_mapping_path
            )
        logger.info(
            f"Correction mapping linked to {len(corpus_stores)} LemmatizedStores"
        )

        logger.info("LLM correction completed successfully")

    # Output corrected frequencies to CSV
    stats = merged_counter.get_statistics()
    logger.info(
        f"\nFinal statistics:\n"
        f"  Total tokens: {stats['total_tokens']:,}\n"
        f"  Unique lemmas: {stats['unique_lemmas']:,}"
    )
    min_freq = output_config.get("min_frequency", 1)
    frequencies = merged_counter.get_sorted_frequencies(min_frequency=min_freq)
    corrected_output_path = os.path.join(output_dir, "frequencies_corrected.csv")
    LemmaFrequencyOutputWriter.write_csv(frequencies, corrected_output_path)
    logger.info(f"Corrected frequencies saved to: {corrected_output_path}")

    # =========================================================================
    # STAGE 4-6: Translation, Example Generation, and Meaning Embeddings
    # (Skip if translations_override provided)
    # =========================================================================
    translations: Optional[List[Translation]] = None

    if translations_override is not None:
        logger.info("=" * 60)
        logger.info("STAGE 4-6: Using provided translations (skipping LLM stages)")
        logger.info("=" * 60)
        translations = translations_override
    else:
        translation_config = config.get("llm_translation", {})
        if translation_config.get("enabled", False):
            logger.info("=" * 60)
            logger.info("STAGE 4: Translation")
            logger.info("=" * 60)

            # Initialize LLM cache if not already done
            if "llm_cache" not in locals():
                llm_cache_dir = cache_config.get("llm_directory", "cache/llm_results")
                llm_cache = LemmaCache({"directory": llm_cache_dir})

            provider_type = translation_config["llm"]["type"]
            provider_config = translation_config["llm"].get(provider_type, {})
            translation_provider = create_provider(provider_type, provider_config)

            translator_config = {
                "batch_size": translation_config.get("batch_size", 25),
                "prompts_dir": translation_config.get(
                    "prompt_directory", "prompts/translation"
                ),
                "output_dir": output_dir,
                "max_requests_per_minute": translation_config.get(
                    "max_requests_per_minute", 0
                ),
                "max_concurrent_requests": translation_config.get(
                    "max_concurrent_requests", 5
                ),
                "cache_only": cache_only,
            }

            translator = LemmaTranslator(
                translation_provider, llm_cache, translator_config
            )
            min_freq = output_config.get("min_frequency", 1)
            translations, _ = translator.translate(
                merged_counter, min_frequency=min_freq
            )

            logger.info("LLM translation completed")

        # =========================================================================
        # STAGE 5: Example Generation (only when disambiguation enabled)
        # =========================================================================
        if translations is not None and disambiguation_config.get("enabled", False):
            logger.info("=" * 60)
            logger.info("STAGE 5: Example Generation")
            logger.info("=" * 60)

            example_config = config.get("example_generation", {})

            # Initialize LLM cache if not already done
            if "llm_cache" not in locals():
                llm_cache_dir = cache_config.get("llm_directory", "cache/llm_results")
                llm_cache = LemmaCache({"directory": llm_cache_dir})

            example_provider_type = example_config.get("llm", {}).get("type", "google")
            example_provider_config = example_config.get("llm", {}).get(
                example_provider_type, {}
            )
            example_provider = create_provider(
                example_provider_type, example_provider_config
            )

            example_gen_config = {
                "batch_size": example_config.get("batch_size", 50),
                "prompts_dir": example_config.get(
                    "prompt_directory", "prompts/examples"
                ),
                "num_examples_poly": example_config.get("num_examples_poly", 10),
                "max_requests_per_minute": example_config.get(
                    "max_requests_per_minute", 0
                ),
                "max_concurrent_requests": example_config.get(
                    "max_concurrent_requests", 5
                ),
                "cache_only": cache_only,
            }

            disambiguated_pos = set(
                disambiguation_config.get("pos_types", ["NOUN", "VERB", "ADJ", "ADV"])
            )
            example_generator = ExampleGenerator(
                example_provider, llm_cache, example_gen_config, disambiguated_pos
            )
            translations, _ = example_generator.generate(translations)
            logger.info("Example generation completed")

        # =========================================================================
        # STAGE 6: Meaning Embeddings
        # =========================================================================
        if translations is not None and disambiguation_config.get("enabled", False):
            logger.info("=" * 60)
            logger.info("STAGE 6: Meaning Embeddings")
            logger.info("=" * 60)

            model_name = disambiguation_config.get(
                "model", "dccuchile/bert-base-spanish-wwm-cased"
            )
            meaning_emb_generator = MeaningEmbeddingGenerator(
                model_name=model_name,
                bert_batch_tokens=disambiguation_config.get("bert_batch_tokens", 4096),
            )
            translations = meaning_emb_generator.generate(translations)
            logger.info("Meaning embeddings generated")

    # Save translations to JSON (if available)
    if translations is not None:
        translations_path = os.path.join(output_dir, "translations.json")
        translations_dicts = [t.to_dict() for t in translations]
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(translations_dicts, f, ensure_ascii=False, indent=2)
        logger.info(f"Translations saved to: {translations_path}")

    # =========================================================================
    # STAGE 7: Disambiguation (ALL corpora)
    # =========================================================================
    disambiguated_stores: Dict[str, DisambiguatedStore] = {}  # corpus_name -> store
    vocabulary: Optional[Vocabulary] = None

    if translations is not None and disambiguation_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 7: Disambiguation")
        logger.info("=" * 60)

        # Get disambiguated POS types from config
        disambiguated_pos = set(
            disambiguation_config.get("pos_types", ["NOUN", "VERB", "ADJ", "ADV"])
        )
        vocabulary = Vocabulary.from_translations_list(
            translations, disambiguated_pos=disambiguated_pos
        )
        vocab_hash = generate_vocabulary_hash(translations)
        bert_model = disambiguation_config.get(
            "model", "dccuchile/bert-base-spanish-wwm-cased"
        )
        disambiguation_cache_key = generate_disambiguation_cache_key(
            vocab_hash, bert_model
        )

        disambiguator = BatchDisambiguator(
            vocabulary=vocabulary,
            model_name=bert_model,
            bert_batch_tokens=disambiguation_config.get("bert_batch_tokens", 4096),
        )

        for corpus_name, (sent_store, lemma_store, cache_key) in corpus_stores.items():
            logger.info(f"Disambiguating: {corpus_name}")

            # Cache path for this corpus's disambiguation
            cache_base = Path(cache_config.get("directory", "cache"))
            corpus_cache_dir = cache_base / corpus_name / cache_key
            output_path = (
                corpus_cache_dir
                / disambiguation_cache_key
                / "DisambiguatedStore.jsonl.gz"
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
                    use_corrections=correction_mapping_path.exists(),
                )

            disambiguated_stores[corpus_name] = dis_store

        logger.info(f"Disambiguation complete for {len(disambiguated_stores)} corpora")

    # =========================================================================
    # STAGE 7.5: Update estimated_share from corpus usage
    # =========================================================================
    if disambiguated_stores and vocabulary is not None and translations is not None:
        logger.info("=" * 60)
        logger.info("STAGE 7.5: Updating estimated shares from corpus usage")
        logger.info("=" * 60)

        share_updater = ShareUpdater()
        translations = share_updater.update(
            list(disambiguated_stores.values()), translations, vocabulary
        )
        # Note: Do NOT recreate vocabulary here - IDs must stay stable for
        # DisambiguatedStores and Co-Occurrence Matrix consistency
        logger.info("Estimated shares updated based on corpus usage")

        # Re-save translations.json with updated estimated_share values
        translations_path = os.path.join(output_dir, "translations.json")
        translations_dicts = [t.to_dict() for t in translations]
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(translations_dicts, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated translations saved to: {translations_path}")

    # =========================================================================
    # STAGE 8: Co-Occurrence Matrix (ALL corpora)
    # =========================================================================
    cooccurrence_matrix = None
    if disambiguated_stores and vocabulary is not None:
        logger.info("=" * 60)
        logger.info("STAGE 8: Co-Occurrence Matrix")
        logger.info("=" * 60)

        builder = CoOccurrenceBuilder(vocabulary)
        cooccurrence_matrix = builder.build(list(disambiguated_stores.values()))
        logger.info("Co-occurrence matrix built successfully")

    # =========================================================================
    # Save Vocabulary
    # =========================================================================
    if vocabulary is not None:
        if cooccurrence_matrix is not None:
            vocabulary.cooccurrence_matrix = cooccurrence_matrix

        # Save lightweight version (for quick inspection)
        vocab_dir_light = Path(output_dir) / "vocabulary"
        vocabulary.save(
            str(vocab_dir_light),
            include_embeddings=False,
            include_private=False,
            save_cooccurrence=False,
        )
        logger.info(f"Vocabulary (light) saved to: {vocab_dir_light}")

        # Save full version (with embeddings and cooccurrence matrix)
        vocab_dir_full = Path(output_dir) / "vocabulary_full"
        vocabulary.save(
            str(vocab_dir_full),
            include_embeddings=True,
            include_private=True,
            save_cooccurrence=True,
        )
        logger.info(f"Vocabulary (full) saved to: {vocab_dir_full}")

    logger.info("=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)

    return PipelineResult(
        output_dir=output_dir,
        vocabulary=vocabulary,
        translations=translations,
        cooccurrence_matrix=cooccurrence_matrix,
        merged_counter=merged_counter,
        corpus_stores=corpus_stores,
        disambiguated_stores=disambiguated_stores,
    )


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    run_pipeline(config, config_path=config_path)
