from collections import defaultdict
from typing import Tuple, Generator, Dict, List
import logging

logger = logging.getLogger(__name__)


class FrequencyCounter:
    """
    Incremental frequency counter for lemmas.

    Efficiently counts lemma frequencies from a stream of (lemma, pos) tuples
    without loading all data into memory at once.

    Stores raw frequencies with original POS tags without normalization.
    """

    def __init__(self):
        """Initialize the frequency counter."""
        self.lemma_counts = defaultdict(int)

        # More for debugging
        self.lemma_pos_details = defaultdict(set)  # Tracks actual POS tags for each key
        self.total_tokens = 0

    def _create_lemma_key(self, lemma: str, pos: str) -> str:
        """
        Create a unique key for lemma+pos combination.

        Stores original POS tags without normalization.

        Args:
            lemma: The lemma
            pos: The original POS tag from spaCy

        Returns:
            Key string in format "lemma|ORIGINAL_POS"
        """
        return f"{lemma}|{pos}"

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
            self.lemma_pos_details[key].add(pos)
            self.total_tokens += 1

            if self.total_tokens % 100000 == 0:
                logger.info(
                    f"Processed {self.total_tokens:,} tokens "
                    f"({len(self.lemma_counts):,} unique lemmas)"
                )

    def get_sorted_frequencies(
        self, min_frequency: int = 1
    ) -> List[Dict[str, any]]:
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
                lemma, normalized_pos = key.split("|", 1)
                pos_details = sorted(self.lemma_pos_details[key])

                # Build POS string: normalized POS first, then additional POS (excluding normalized)
                additional_pos = [pos for pos in pos_details if pos != normalized_pos]
                if additional_pos:
                    pos_string = f"{normalized_pos}/{'/'.join(additional_pos)}"
                else:
                    pos_string = normalized_pos

                results.append(
                    {
                        "lemma": lemma,
                        "frequency": count,
                        "pos": pos_string,
                    }
                )

        results.sort(key=lambda x: x["frequency"], reverse=True)
        return results

    def merge(self, other: "FrequencyCounter") -> None:
        """
        Merge frequencies from another FrequencyCounter into this one.

        Args:
            other: Another FrequencyCounter to merge from
        """
        for key, count in other.lemma_counts.items():
            self.lemma_counts[key] += count
            if key in other.lemma_pos_details:
                self.lemma_pos_details[key].update(other.lemma_pos_details[key])

        self.total_tokens += other.total_tokens

    def to_dict(self) -> Dict[str, any]:
        """
        Export frequency data to dictionary for serialization.

        Returns:
            Dictionary with all counter data
        """
        return {
            "lemma_counts": dict(self.lemma_counts),
            "lemma_pos_details": {
                k: list(v) for k, v in self.lemma_pos_details.items()
            },
            "total_tokens": self.total_tokens,
        }

    @staticmethod
    def from_dict(data: Dict[str, any]) -> "FrequencyCounter":
        """
        Create FrequencyCounter from dictionary.

        Args:
            data: Dictionary created by to_dict()

        Returns:
            New FrequencyCounter instance with loaded data
        """
        counter = FrequencyCounter()
        counter.lemma_counts = defaultdict(int, data["lemma_counts"])
        counter.lemma_pos_details = defaultdict(
            set, {k: set(v) for k, v in data["lemma_pos_details"].items()}
        )
        counter.total_tokens = data["total_tokens"]
        return counter

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
