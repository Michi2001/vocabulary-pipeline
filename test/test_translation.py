"""
Unit tests for LLM translation using test cases from CSV files.

This test validates that LLM providers correctly translate Spanish lemmas to German.
The test data is loaded from CSV files.

Run with: python -m unittest test.test_translation
Requires: GOOGLE_API_KEY or OPENAI_API_KEY environment variable
"""

import csv
import json
import logging
import sys
import unittest
from datetime import datetime
from pathlib import Path

from src.frequency_counter import FrequencyCounter
from src.llm_providers import create_provider
from src.llm_translator import LemmaTranslator
from src.vocab import Vocabulary


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
PROVIDER_TYPE = "google"
PROVIDER_CONFIG = {  # best settings for Google Gemini
    "model": "gemini-3-flash-preview",
    "thinking_level": "minimal",
    "temperature": 1.0,
}
# PROVIDER_CONFIG = {
#     "model": "gemini-3-pro-preview",
#     "thinking_level": "low",
#     "temperature": 1.0,
# }
BATCH_SIZE = 20  # Number of items to process per batch
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
        }

        # Create translator (cache=None disables caching)
        translator = LemmaTranslator(
            self.provider, cache=None, config=translator_config
        )

        # Run translation
        print(f"\nStarting translation with batch size {BATCH_SIZE}...")
        translations, usage_stats = translator.translate(self.counter, min_frequency=1)

        print(f"\nTranslation complete!")
        print(f"Total translations: {len(translations)}")
        print(
            f"API calls: {usage_stats['api_calls']}, Cache hits: {usage_stats['cache_hits']}"
        )

        # Basic validation
        self.assertIsInstance(translations, list, "Translations must be a list")
        self.assertGreater(
            len(translations), 0, "Translations list should not be empty"
        )

        # Validate structure of first translation (Translation objects have these attributes)
        if translations:
            first = translations[0]
            required_attrs = ["id", "lemma", "pos", "meanings", "frequency"]
            for attr in required_attrs:
                self.assertTrue(
                    hasattr(first, attr),
                    f"Translation missing required attribute: {attr}",
                )

        # Create Vocabulary and save to output directory
        vocab = Vocabulary.from_translations_list(translations)
        vocab_dir = OUTPUT_DIR / "vocabulary"
        vocab.save(str(vocab_dir))

        print(f"\nVocabulary saved to: {vocab_dir}")
        print(f"  Translations: {len(vocab.translations)}")
        print(f"  Word variants: {len(vocab.word_variants)}")

        # Verify vocabulary file was created
        translations_file = vocab_dir / "translations.json"
        self.assertTrue(
            translations_file.exists(),
            f"Vocabulary translations file not created: {translations_file}",
        )


class Tee:
    """Write to multiple output streams simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    # Create output directory
    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open log file and redirect stdout to both console and file
    log_file = open(OUTPUT_LOG_PATH, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    # Configure logging to show INFO messages in both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        unittest.main(verbosity=2)
    finally:
        sys.stdout = original_stdout
        log_file.close()
