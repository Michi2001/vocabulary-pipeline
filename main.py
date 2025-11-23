#!/usr/bin/env python3
"""
Lemma Frequency Analysis Pipeline

Analyzes text corpora, lemmatizes words in context, and generates frequency lists.
"""

import yaml
import logging
import os
from pathlib import Path

from src.corpus_readers import LeipzigCorpusReader
from src.lemmatizer import Lemmatizer
from src.frequency_counter import FrequencyCounter
from src.output_writer import OutputWriter

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
    logger.info("=" * 80)
    logger.info("Starting Lemma Frequency Analysis Pipeline")
    logger.info("=" * 80)

    config = load_config()

    corpus_config = config["corpus"]
    lemma_config = config["lemmatization"]
    processing_config = config["processing"]
    output_config = config["output"]

    logger.info(f"Corpus file: {corpus_config['file_path']}")
    logger.info(f"Language model: {lemma_config['model']}")
    logger.info(f"Output base path: {output_config['base_path']}")

    os.makedirs(output_config["base_path"], exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("Step 1/4: Loading corpus reader")
    logger.info("=" * 80)
    reader = LeipzigCorpusReader(
        file_path=corpus_config["file_path"],
        encoding=corpus_config.get("encoding", "utf-8"),
    )

    metadata = reader.get_metadata()
    logger.info(f"Corpus metadata: {metadata}")

    logger.info("\n" + "=" * 80)
    logger.info("Step 2/4: Initializing lemmatizer")
    logger.info("=" * 80)
    lemmatizer = Lemmatizer(
        language_model=lemma_config["model"],
        batch_size=lemma_config.get("batch_size", 1000),
    )

    logger.info("\n" + "=" * 80)
    logger.info("Step 3/4: Processing corpus and counting frequencies")
    logger.info("=" * 80)
    counter = FrequencyCounter()

    sentence_batches = reader.read_sentences(
        batch_size=processing_config.get("sentence_batch_size", 10000)
    )

    lemma_pos_stream = lemmatizer.process_corpus_batches(sentence_batches)

    counter.update(lemma_pos_stream)

    stats = counter.get_statistics()
    logger.info(f"Processing complete!")
    logger.info(f"Statistics: {stats}")

    logger.info("\n" + "=" * 80)
    logger.info("Step 4/4: Writing output files")
    logger.info("=" * 80)
    min_freq = processing_config.get("min_frequency", 1)
    frequencies = counter.get_sorted_frequencies(min_frequency=min_freq)

    logger.info(f"Top 10 most frequent lemmas:")
    for i, item in enumerate(frequencies[:10], 1):
        logger.info(f"  {i}. {item['lemma']} ({item['pos']}): {item['frequency']:,}")

    OutputWriter.write_csv(
        frequencies, os.path.join(output_config["base_path"], "output.csv")
    )

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Output files:")
    logger.info(f"  - {output_config['base_path']}.csv")
    logger.info(f"  - {output_config['base_path']}.tsv")
    logger.info(f"  - {output_config['base_path']}.json")
