#!/usr/bin/env python3
"""
Lemma Frequency Analysis Pipeline

Analyzes text corpora, lemmatizes words in context, and generates frequency lists.
"""

import yaml
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from src.lemmatizer import Lemmatizer
from src.output_writer import OutputWriter
from src.single_corpus_processor import SingleCorpusProcessor
from src.multi_corpus_processor import MultiCorpusProcessor
from src.corpus_readers import LeipzigCorpusReader, OpusCorpusReader
from src.llm_providers import create_provider
from src.llm_corrector import LemmaCorrector
from src.llm_translator import LemmaTranslator


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

    # Optional: LLM-based correction
    llm_config = config.get("llm_correction", {})
    if llm_config.get("enabled", False):
        logger.info("LLM correction enabled - applying corrections...")

        # Load API key from environment
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        try:
            # Load system prompt from file
            prompt_file = llm_config.get(
                "system_prompt_file", "config/llm_system_prompt.txt"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()

            if system_prompt:
                # Get provider config
                provider_type = llm_config["llm"]["type"]
                provider_config = llm_config["llm"].get(provider_type, {})
                provider_config["api_key"] = api_key

                # Create provider
                provider = create_provider(provider_type, provider_config)

                # Create corrector
                corrector_config = {
                    "batch_size": llm_config.get("batch_size", 50),
                    "system_prompt": system_prompt,
                    "output_dir": output_dir,
                    "cache_config": llm_config.get("cache", {}),
                    "llm_model": provider_config.get("model", "gpt-5-mini"),
                }
                corrector = LemmaCorrector(provider, corrector_config)

                # Apply corrections
                min_freq = processing_config.get("min_frequency", 1)
                counter = corrector.correct(counter, min_frequency=min_freq)

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
    translation_config = config.get("llm_translation", {})
    if translation_config.get("enabled", False):
        logger.info("LLM translation enabled - translating lemmas...")

        try:
            # Reuse provider if available, otherwise create new one
            if not llm_config.get("enabled", False) or "provider" not in locals():
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
                provider_type = translation_config["llm"]["type"]
                provider_config = translation_config["llm"].get(provider_type, {})
                provider_config["api_key"] = api_key
                provider = create_provider(provider_type, provider_config)

            translator_config = {
                "batch_size": translation_config.get("batch_size", 25),
                "prompts_dir": translation_config.get(
                    "prompt_directory", "prompts/translation"
                ),
                "output_dir": output_dir,
                "cache_config": translation_config.get("cache", {}),
                "llm_model": provider_config.get("model", "gpt-5-mini"),
            }

            translator = LemmaTranslator(provider, translator_config)
            min_freq = processing_config.get("min_frequency", 1)
            translator.translate(counter, min_frequency=min_freq)

            logger.info("LLM translation completed - output: translations.json")

        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            logger.warning("Continuing without translation")

    logger.info("Pipeline completed!")
