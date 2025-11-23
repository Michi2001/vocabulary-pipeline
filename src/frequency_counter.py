from collections import defaultdict
from typing import Tuple, Generator, Dict, List
import logging

logger = logging.getLogger(__name__)


class FrequencyCounter:
    """
    Incremental frequency counter for lemmas.

    Efficiently counts lemma frequencies from a stream of (lemma, pos) tuples
    without loading all data into memory at once.

    Treats lemmas with different main POS categories (NOUN, VERB, ADJ, ADV)
    as separate entries.
    """

    # Main POS categories that should be treated as separate lemmas
    MAIN_POS_CATEGORIES = {"NOUN", "VERB", "ADJ", "ADV"}

    def __init__(self):
        """Initialize the frequency counter."""
        self.lemma_counts = defaultdict(int)
        self.lemma_pos_details = defaultdict(set)  # Tracks actual POS tags for each key
        self.total_tokens = 0

    def _normalize_pos(self, pos: str) -> str:
        """
        Normalize POS tag to main category if applicable.

        Args:
            pos: Original POS tag

        Returns:
            Main category if pos is in MAIN_POS_CATEGORIES, otherwise original pos
        """
        return pos if pos in self.MAIN_POS_CATEGORIES else "OTHER"

    def _create_lemma_key(self, lemma: str, pos: str) -> str:
        """
        Create a unique key for lemma+pos combination.

        For main POS categories (NOUN, VERB, ADJ, ADV), creates separate keys.
        For other POS tags, groups them together.

        Args:
            lemma: The lemma
            pos: The POS tag

        Returns:
            Key string in format "lemma|POS" for main categories, "lemma|OTHER" for others
        """
        normalized_pos = self._normalize_pos(pos)
        return f"{lemma}|{normalized_pos}"

    def update(
        self, lemma_pos_generator: Generator[Tuple[str, str], None, None]
    ) -> None:
        """
        Update frequency counts from a generator of (lemma, pos) tuples.

        Args:
            lemma_pos_generator: Generator yielding (lemma, pos) tuples
        """
        for lemma, pos in lemma_pos_generator:
            key = self._create_lemma_key(lemma, pos)
            self.lemma_counts[key] += 1
            self.lemma_pos_details[key].add(pos)  # Track all actual POS tags
            self.total_tokens += 1

            if self.total_tokens % 100000 == 0:
                logger.info(
                    f"Processed {self.total_tokens:,} tokens, {len(self.lemma_counts):,} unique lemmas"
                )

    def get_sorted_frequencies(self, min_frequency: int = 1) -> List[Dict[str, any]]:
        """
        Get lemma frequencies sorted by count (descending).

        Args:
            min_frequency: Minimum frequency threshold (default: 1)

        Returns:
            List of dictionaries with 'lemma', 'frequency', and 'pos' keys
        """
        results = []
        for key, count in self.lemma_counts.items():
            if count >= min_frequency:
                lemma, _ = key.split("|", 1)
                pos_details = sorted(self.lemma_pos_details[key])

                results.append(
                    {
                        "lemma": lemma,
                        "frequency": count,
                        "pos": (
                            "/".join(pos_details)
                            if len(pos_details) > 1
                            else pos_details[0]
                        ),
                    }
                )

        results.sort(key=lambda x: x["frequency"], reverse=True)
        return results

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the frequency data.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "unique_lemmas": len(self.lemma_counts),
            "most_common": sorted(
                self.lemma_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
