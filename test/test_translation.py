"""
Unit tests for LLM translation using test cases from CSV files.

This test validates that LLM providers correctly translate Spanish lemmas to German.
The test data is loaded from CSV files.

Run with: python -m unittest test.test_translation
Requires: OPENAI_API_KEY environment variable
"""

import unittest
import os
import csv
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from src.llm_providers import create_provider
from src.llm_translator import LemmaTranslator
from src.frequency_counter import FrequencyCounter
from datetime import datetime


def load_translation_input(csv_path):
    """Load translation input from CSV file and create FrequencyCounter."""
    counter = FrequencyCounter()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row["lemma"]
            frequency = int(row["frequency"])
            pos = row["pos"]

            # Create key and add to counter
            key = f"{lemma}|{pos}"
            counter.lemma_counts[key] = frequency
            counter.lemma_pos_details[key].add(pos)
            counter.total_tokens += frequency

    return counter


# Configuration
INPUT_CSV = Path("test/data/translation_input.csv")
PROMPT_FOLDER = Path("prompts/translation")
PROVIDER_TYPE = "openai"
PROVIDER_CONFIG = {"api_key": os.getenv("OPENAI_API_KEY"), "model": "gpt-5-mini"}
BATCH_SIZE = 10  # Number of items to process per batch
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"test/outputs/test_translation/{TIMESTAMP}")
OUTPUT_LOG_PATH = OUTPUT_DIR / "translation.log"


class TestLemmaTranslation(unittest.TestCase):
    """Test LLM lemma translation with real API calls."""

    @classmethod
    def setUpClass(cls):
        """Initialize provider and load input data."""
        # Create provider
        cls.provider = create_provider(PROVIDER_TYPE, PROVIDER_CONFIG)

        # Load translation input
        cls.counter = load_translation_input(INPUT_CSV)

        print(f"\nLoaded {len(cls.counter.lemma_counts)} lemmas for translation")
        print(f"Total tokens: {cls.counter.total_tokens}")

    def test_translate_all_lemmas(self):
        """Test translation of all lemmas from input CSV."""
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Configure translator
        translator_config = {
            "batch_size": BATCH_SIZE,
            "prompts_dir": str(PROMPT_FOLDER),
            "output_dir": str(OUTPUT_DIR),
            "cache_config": {
                "enabled": True,
                "directory": "cache/test_llm_translations",
            },
            "llm_model": PROVIDER_CONFIG["model"],
        }

        # Create translator
        translator = LemmaTranslator(self.provider, translator_config)

        # Run translation
        print(f"\nStarting translation with batch size {BATCH_SIZE}...")
        translator.translate(self.counter, min_frequency=1)

        # Check that output file was created
        output_file = OUTPUT_DIR / "translations.json"
        self.assertTrue(
            output_file.exists(), f"Translation output file not created: {output_file}"
        )

        # Load and validate output
        with open(output_file, "r", encoding="utf-8") as f:
            translations = json.load(f)

        print(f"\nTranslation complete!")
        print(f"Total translations: {len(translations)}")
        print(f"Output saved to: {output_file}")

        # Basic validation
        self.assertIsInstance(translations, list, "Translations must be a list")
        self.assertGreater(
            len(translations), 0, "Translations list should not be empty"
        )

        # Validate structure of first translation
        if translations:
            first = translations[0]
            required_fields = ["id", "lemma", "pos", "meanings", "frequency"]
            for field in required_fields:
                self.assertIn(
                    field, first, f"Translation missing required field: {field}"
                )


if __name__ == "__main__":
    # Create output directory
    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging to show INFO messages in both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_LOG_PATH, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    unittest.main(verbosity=2)
