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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

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
from src.meaning_embedding_generator import MeaningEmbeddingGenerator
from src.output_writer import OutputWriter
from src.post_processor import FrequencyPostProcessor
from src.stores.disambiguated_store import DisambiguatedStore
from src.stores.lemmatized_store import LemmatizedStore
from src.stores.sentence_store import SentenceStore
from src.vocab import Translation, Vocabulary

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    """Main pipeline execution."""

    config_path = "config.yaml"
    config = load_config(config_path)

    lemma_config = config["lemmatization"]
    processing_config = config["processing"]
    output_config = config["output"]
    cache_config = config.get("cache", {"enabled": False})
    filter_config = config.get("filtering", {})
    disambiguation_config = config.get("disambiguation", {"enabled": False})
    cooccurrence_config = config.get("cooccurrence", {"enabled": False})

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

    # Copy config file to output directory
    config_backup_path = os.path.join(output_dir, "config.yaml")
    shutil.copy2(config_path, config_backup_path)
    logger.info(f"Config copied to: {config_backup_path}")
    logger.info(f"Log file: {log_file_path}")

    logger.info("Starting pipeline...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {lemma_config['model']}")
    logger.info(f"Cache: {'enabled' if cache_config.get('enabled') else 'disabled'}")
    logger.info(f"Filtering: {'enabled' if filter_config else 'disabled'}")

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

        # Initialize shared LLM cache
        llm_cache_config = config.get(
            "llm_cache", {"enabled": True, "directory": "cache"}
        )
        llm_cache = (
            LemmaCache(llm_cache_config)
            if llm_cache_config.get("enabled", True)
            else None
        )

        # Load system prompt
        prompt_file = correction_config.get(
            "system_prompt_file", "config/llm_system_prompt.txt"
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
            "slice": correction_config.get("slice", None),
        }
        corrector = LemmaCorrector(correction_provider, llm_cache, corrector_config)

        # Apply corrections
        min_freq = processing_config.get("min_frequency", 1)
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

    # Output lemma frequencies to CSV
    stats = merged_counter.get_statistics()
    logger.info(
        f"\nFinal statistics:\n"
        f"  Total tokens: {stats['total_tokens']:,}\n"
        f"  Unique lemmas: {stats['unique_lemmas']:,}"
    )
    min_freq = processing_config.get("min_frequency", 1)
    frequencies = merged_counter.get_sorted_frequencies(min_frequency=min_freq)
    output_path = os.path.join(output_dir, "output.csv")
    OutputWriter.write_csv(frequencies, output_path)

    # =========================================================================
    # STAGE 4: Translation
    # =========================================================================
    translations: Optional[List[Translation]] = None
    translation_config = config.get("llm_translation", {})
    if translation_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 4: Translation")
        logger.info("=" * 60)

        # Initialize shared LLM cache if not already done
        if "llm_cache" not in locals():
            llm_cache_config = config.get(
                "llm_cache", {"enabled": True, "directory": "cache"}
            )
            llm_cache = (
                LemmaCache(llm_cache_config)
                if llm_cache_config.get("enabled", True)
                else None
            )

        provider_type = translation_config["llm"]["type"]
        provider_config = translation_config["llm"].get(provider_type, {})
        translation_provider = create_provider(provider_type, provider_config)

        translator_config = {
            "batch_size": translation_config.get("batch_size", 25),
            "prompts_dir": translation_config.get(
                "prompt_directory", "prompts/translation"
            ),
            "output_dir": output_dir,
        }

        translator = LemmaTranslator(translation_provider, llm_cache, translator_config)
        min_freq = processing_config.get("min_frequency", 1)
        translations, usage_stats = translator.translate(
            merged_counter, min_frequency=min_freq
        )

        logger.info("LLM translation completed")

    # =========================================================================
    # STAGE 5: Meaning Embeddings
    # =========================================================================
    if translations is not None and disambiguation_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 5: Meaning Embeddings")
        logger.info("=" * 60)

        model_name = disambiguation_config.get(
            "model", "dccuchile/bert-base-spanish-wwm-cased"
        )
        meaning_emb_generator = MeaningEmbeddingGenerator(
            model_name=model_name,
            batch_size=disambiguation_config.get("batch_size", 32),
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
    # STAGE 6: Disambiguation (ALL corpora)
    # =========================================================================
    disambiguated_stores: List[DisambiguatedStore] = []
    vocabulary: Optional[Vocabulary] = None

    if translations is not None and disambiguation_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 6: Disambiguation")
        logger.info("=" * 60)

        vocabulary = Vocabulary.from_translations_list(translations)
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

            disambiguated_stores.append(dis_store)

        logger.info(f"Disambiguation complete for {len(disambiguated_stores)} corpora")

    # =========================================================================
    # STAGE 7: Co-Occurrence Matrix (ALL corpora)
    # =========================================================================
    cooccurrence_matrix = None
    if disambiguated_stores and cooccurrence_config.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STAGE 7: Co-Occurrence Matrix")
        logger.info("=" * 60)

        if vocabulary is None:
            raise RuntimeError("Vocabulary required for co-occurrence matrix")

        builder = CoOccurrenceBuilder(vocabulary)
        cooccurrence_matrix = builder.build(disambiguated_stores)
        logger.info("Co-occurrence matrix built successfully")

    # =========================================================================
    # Save Vocabulary (with co-occurrence matrix if available)
    # =========================================================================
    if vocabulary is not None:
        # Attach co-occurrence matrix if available
        if cooccurrence_matrix is not None:
            vocabulary.cooccurrence_matrix = cooccurrence_matrix

        vocab_path = Path(output_dir) / "vocabulary.pkl"
        vocabulary.to_file(str(vocab_path))
        logger.info(f"Vocabulary saved to: {vocab_path}")

    logger.info("=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)
