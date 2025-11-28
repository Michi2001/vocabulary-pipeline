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

from src.lemmatizer import Lemmatizer
from src.output_writer import OutputWriter
from src.single_corpus_processor import SingleCorpusProcessor
from src.multi_corpus_processor import MultiCorpusProcessor
from src.corpus_readers.leipzig_reader import LeipzigCorpusReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = output_config["directory"]
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Copy config file to output directory
    config_backup_path = os.path.join(output_dir, "config.yaml")
    shutil.copy2(config_path, config_backup_path)
    logger.info(f"Config copied to: {config_backup_path}")

    logger.info("Starting pipeline...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {lemma_config['model']}")
    logger.info(f"Cache: {'enabled' if cache_config.get('enabled') else 'disabled'}")

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
        reader = LeipzigCorpusReader(
            file_path=corpus_config["file_path"],
            encoding=corpus_config.get("encoding", "utf-8"),
        )
        processor = SingleCorpusProcessor(
            corpus_config=corpus_config,
            reader=reader,
            lemmatizer=lemmatizer,
            processing_config=processing_config,
            cache_config=cache_config,
        )
        processors.append(processor)

    # Process all corpora
    multi_processor = MultiCorpusProcessor(processors=processors)
    counter = multi_processor.process()

    # Output results
    stats = counter.get_statistics()
    logger.info(
        f"\nFinal statistics:\n"
        f"  Total tokens: {stats['total_tokens']:,}\n"
        f"  Known lemmas: {stats['unique_lemmas']:,}\n"
        f"  OOV tokens: {stats['total_oov_tokens']:,} ({stats['oov_percentage']:.2f}%)\n"
        f"  Unique OOV: {stats['unique_oov']:,}"
    )

    min_freq = processing_config.get("min_frequency", 1)

    # Write known words
    frequencies = counter.get_sorted_frequencies(min_frequency=min_freq, oov=False)
    output_path = os.path.join(output_dir, "output.csv")
    OutputWriter.write_csv(frequencies, output_path, oov=False)

    # Write OOV words
    oov_frequencies = counter.get_sorted_frequencies(min_frequency=min_freq, oov=True)
    oov_output_path = os.path.join(output_dir, "output_oov.csv")
    OutputWriter.write_csv(oov_frequencies, oov_output_path, oov=True)

    logger.info("Pipeline completed!")
