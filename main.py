#!/usr/bin/env python3
"""
Lemma Frequency Analysis Pipeline

Analyzes text corpora, lemmatizes words in context, and generates frequency lists.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import *

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
    config = load_config()

    corpus_config = config["corpus"]
    lemma_config = config["lemmatization"]
    processing_config = config["processing"]
    output_config = config["output"]

    logger.info("Starting pipeline...")
    logger.info(f"Corpus: {corpus_config['file_path']}")
    logger.info(f"Model: {lemma_config['model']}")

    os.makedirs(output_config["base_path"], exist_ok=True)

    reader = LeipzigCorpusReader(
        file_path=corpus_config["file_path"],
        encoding=corpus_config.get("encoding", "utf-8"),
    )

    lemmatizer = Lemmatizer(
        language_model=lemma_config["model"],
        batch_size=lemma_config.get("batch_size", 1000),
    )

    counter = FrequencyCounter()

    sentence_batches: Generator[List[str], None, None] = reader.read_sentences(
        batch_size=processing_config.get("sentence_batch_size", 10000)
    )

    lemma_pos_stream: Generator[Tuple[str, str], None, None] = (
        lemmatizer.process_corpus_batches(sentence_batches)
    )

    counter.update(lemma_pos_stream)

    stats = counter.get_statistics()
    logger.info(
        f"Processed {stats['total_tokens']:,} tokens, {stats['unique_lemmas']:,} unique lemmas"
    )

    min_freq = processing_config.get("min_frequency", 1)
    frequencies = counter.get_sorted_frequencies(min_frequency=min_freq)

    OutputWriter.write_csv(
        frequencies, os.path.join(output_config["base_path"], "output.csv")
    )

    logger.info("Pipeline completed!")
