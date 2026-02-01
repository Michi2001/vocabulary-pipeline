"""Unit tests for FrequencyPostProcessor numeric filtering."""

import sys
import unittest

sys.path.insert(0, ".")

from src.frequency_counter import FrequencyCounter
from src.post_processor import FrequencyPostProcessor


FILTER_CONFIG = {
    "accepted_pos": ["NOUN", "VERB", "ADJ", "NUM"],
    "pos_mappings": {},
    "include_countries": False,
}


def _make_counter(entries: dict) -> FrequencyCounter:
    """Create a FrequencyCounter from a dict of {\"lemma|POS\": count}."""
    counter = FrequencyCounter()
    for key, count in entries.items():
        counter.lemma_counts[key] = count
        counter.total_tokens += count
    return counter


class TestNumericFiltering(unittest.TestCase):
    """Test that NUM lemmas containing digits are filtered out."""

    def setUp(self):
        self.processor = FrequencyPostProcessor(FILTER_CONFIG)

    def test_pure_digit_num_filtered(self):
        """Pure digit NUM lemmas like '1', '23', '23492340' are removed."""
        counter = _make_counter({
            "1|NUM": 50,
            "23|NUM": 30,
            "23492340|NUM": 5,
        })
        result = self.processor.process(counter)
        self.assertEqual(len(result.lemma_counts), 0)

    def test_mixed_alphanumeric_num_filtered(self):
        """Mixed alphanumeric NUM lemmas like '3er', '2do' are removed."""
        counter = _make_counter({
            "3er|NUM": 10,
            "2do|NUM": 8,
            "100km|NUM": 3,
        })
        result = self.processor.process(counter)
        self.assertEqual(len(result.lemma_counts), 0)

    def test_written_number_num_kept(self):
        """Written-out number words like 'tres', 'cien', 'mil' are kept."""
        counter = _make_counter({
            "tres|NUM": 100,
            "cien|NUM": 40,
            "mil|NUM": 60,
        })
        result = self.processor.process(counter)
        self.assertEqual(len(result.lemma_counts), 3)
        self.assertEqual(result.lemma_counts["tres|NUM"], 100)
        self.assertEqual(result.lemma_counts["cien|NUM"], 40)
        self.assertEqual(result.lemma_counts["mil|NUM"], 60)

    def test_other_pos_with_digits_unaffected(self):
        """Digits in non-NUM POS lemmas are not filtered."""
        counter = _make_counter({
            "mp3|NOUN": 20,
            "3d|ADJ": 5,
        })
        result = self.processor.process(counter)
        self.assertEqual(len(result.lemma_counts), 2)

    def test_mixed_entries(self):
        """Combined scenario: only NUM lemmas with digits are removed."""
        counter = _make_counter({
            "42|NUM": 10,
            "tres|NUM": 50,
            "gato|NOUN": 200,
            "3er|NUM": 7,
            "mil|NUM": 30,
        })
        result = self.processor.process(counter)
        self.assertNotIn("42|NUM", result.lemma_counts)
        self.assertNotIn("3er|NUM", result.lemma_counts)
        self.assertIn("tres|NUM", result.lemma_counts)
        self.assertIn("mil|NUM", result.lemma_counts)
        self.assertIn("gato|NOUN", result.lemma_counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
