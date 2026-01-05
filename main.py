#!/usr/bin/env python3
"""
Lemma Frequency Analysis Pipeline

Analyzes text corpora, lemmatizes words in context, and generates frequency lists.
"""

import yaml
import json
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from src.lemmatizer import Lemmatizer
from src.output_writer import OutputWriter
from src.single_corpus_processor import SingleCorpusProcessor
from src.multi_corpus_processor import MultiCorpusProcessor
from src.corpus_readers import LeipzigCorpusReader, OpusCorpusReader
from src.llm_providers import create_provider
from src.llm_corrector import LemmaCorrector
from src.llm_translator import LemmaTranslator
from src.lemma_cache import LemmaCache
from src.meaning_embedding_generator import MeaningEmbeddingGenerator
from src.batch_disambiguator import BatchDisambiguator
from src.cooccurrence_builder import CoOccurrenceBuilder
from src.vocab import Translation, Vocabulary


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    # Clear any existing handlers to prevent duplicate logs
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

    # Get corpora configurations
    corpora_configs = config["corpora"]
    logger.info(f"Processing {len(corpora_configs)} corpora")

    # Create SingleCorpusProcessor for each corpus
    processors = []
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

        processor = SingleCorpusProcessor(
            corpus_config=corpus_config,
            reader=reader,
            lemmatizer=lemmatizer,
            processing_config=processing_config,
            cache_config=cache_config,
            filter_config=filter_config,
        )
        processors.append(processor)

    # Process all corpora
    multi_processor = MultiCorpusProcessor(processors=processors)
    counter = multi_processor.process()

    # Initialize shared LLM cache once (used by both correction and translation)
    llm_cache_config = config.get("llm_cache", {"enabled": True, "directory": "cache"})
    llm_cache = (
        LemmaCache(llm_cache_config) if llm_cache_config.get("enabled", True) else None
    )

    # Optional: LLM-based correction
    correction_config = config.get("llm_correction", {})
    correction_mapping = None
    deleted_tokens = None
    if correction_config.get("enabled", False):
        logger.info("LLM correction enabled - applying corrections...")

        try:
            # Load system prompt from file
            prompt_file = correction_config.get(
                "system_prompt_file", "config/llm_system_prompt.txt"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()

            if system_prompt:
                # Get provider config
                provider_type = correction_config["llm"]["type"]
                provider_config = correction_config["llm"].get(provider_type, {})

                # Create provider
                correction_provider = create_provider(provider_type, provider_config)

                # Create corrector
                corrector_config = {
                    "batch_size": correction_config.get("batch_size", 50),
                    "system_prompt": system_prompt,
                    "output_dir": output_dir,
                    "slice": correction_config.get("slice", None),
                }
                corrector = LemmaCorrector(
                    correction_provider, llm_cache, corrector_config
                )

                # Apply corrections - returns (counter, mapping, deleted)
                min_freq = processing_config.get("min_frequency", 1)
                counter, correction_mapping, deleted_tokens = corrector.correct(
                    counter, min_frequency=min_freq
                )

                # Apply corrections to SentenceStore if available
                sentence_store = processors[0].get_sentence_store()
                if sentence_store is not None and sentence_store.has_lemmatized():
                    logger.info("Applying corrections to SentenceStore...")
                    LemmaCorrector.apply_corrections_to_store(
                        sentence_store, correction_mapping, deleted_tokens
                    )
                    logger.info("SentenceStore corrections applied")

                logger.info("LLM correction completed successfully")

        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            logger.warning("Continuing with uncorrected data")

    # Output lemma frequencies to CSV
    stats = counter.get_statistics()
    logger.info(
        f"\nFinal statistics:\n"
        f"  Total tokens: {stats['total_tokens']:,}\n"
        f"  Unique lemmas: {stats['unique_lemmas']:,}"
    )
    min_freq = processing_config.get("min_frequency", 1)
    frequencies = counter.get_sorted_frequencies(min_frequency=min_freq)
    output_path = os.path.join(output_dir, "output.csv")
    OutputWriter.write_csv(frequencies, output_path)

    # Optional: LLM-based translation
    translations: Optional[List[Translation]] = None
    translation_config = config.get("llm_translation", {})
    if translation_config.get("enabled", False):
        logger.info("LLM translation enabled - translating lemmas...")

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
            counter, min_frequency=min_freq
        )

        logger.info("LLM translation completed")

    # Save translations to JSON (if available)
    if translations is not None:
        # Generate meaning embeddings for disambiguation (if enabled)
        if disambiguation_config.get("enabled", False):
            logger.info("Generating meaning embeddings for disambiguation...")
            model_name = disambiguation_config.get(
                "model", "dccuchile/bert-base-spanish-wwm-cased"
            )
            meaning_emb_generator = MeaningEmbeddingGenerator(
                model_name=model_name,
                batch_size=disambiguation_config.get("batch_size", 32),
            )
            translations = meaning_emb_generator.generate(translations)
            logger.info("Meaning embeddings generated")

        # Save translations as JSON (convert Translation objects to dicts)
        translations_path = os.path.join(output_dir, "translations.json")
        translations_dicts = [t.to_dict() for t in translations]
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(translations_dicts, f, ensure_ascii=False, indent=2)
        logger.info(f"Translations saved to: {translations_path}")

    # Disambiguation and Co-Occurrence Matrix (if enabled)
    if translations is not None and disambiguation_config.get("enabled", False):
        logger.info("Starting batch disambiguation...")

        # Get sentence store from first processor
        # (TODO: support multi-corpus by combining stores)
        sentence_store = processors[0].get_sentence_store()
        if sentence_store is not None and sentence_store.has_sentences():
            # Create Vocabulary from translations
            vocabulary = Vocabulary.from_translations_list(translations)

            # Run disambiguation
            model_name = disambiguation_config.get(
                "model", "dccuchile/bert-base-spanish-wwm-cased"
            )
            disambiguator = BatchDisambiguator(
                vocabulary=vocabulary,
                model_name=model_name,
                batch_size=disambiguation_config.get("batch_size", 32),
            )

            # Disambiguation writes directly to SentenceStore
            disambiguator.process(sentence_store)
            logger.info("Disambiguation completed and saved to SentenceStore")

            # Build co-occurrence matrix (if enabled)
            if cooccurrence_config.get("enabled", False):
                logger.info("Building co-occurrence matrix...")
                builder = CoOccurrenceBuilder(vocabulary)
                matrix = builder.build(sentence_store)

                matrix_path = Path(output_dir) / "cooccurrence_matrix.npz"
                builder.save(matrix, matrix_path)
                logger.info(f"Co-occurrence matrix saved to: {matrix_path}")
        else:
            logger.warning(
                "SentenceStore not available - skipping disambiguation. "
                "Clear cache to regenerate."
            )

    logger.info("Pipeline completed!")
