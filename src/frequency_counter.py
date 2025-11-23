from collections import defaultdict
from typing import Tuple, Generator, Dict, List
import logging

logger = logging.getLogger(__name__)


class FrequencyCounter:
    """
    Incremental frequency counter for lemmas.

    Efficiently counts lemma frequencies from a stream of (lemma, pos) tuples
    without loading all data into memory at once.
    """

    def __init__(self):
        """Initialize the frequency counter."""
        self.lemma_counts = defaultdict(int)
        self.lemma_pos = {}
        self.total_tokens = 0

    def update(self, lemma_pos_generator: Generator[Tuple[str, str], None, None]) -> None:
        """
        Update frequency counts from a generator of (lemma, pos) tuples.

        Args:
            lemma_pos_generator: Generator yielding (lemma, pos) tuples
        """
        for lemma, pos in lemma_pos_generator:
            self.lemma_counts[lemma] += 1
            self.total_tokens += 1

            if lemma not in self.lemma_pos:
                self.lemma_pos[lemma] = pos
            elif self.lemma_counts[lemma] % 1000 == 0:
                if self.lemma_pos[lemma] != pos:
                    self.lemma_pos[lemma] = self._merge_pos(self.lemma_pos[lemma], pos)

            if self.total_tokens % 500000 == 0:
                logger.info(f"Progress: {self.total_tokens:,} tokens")

    def _merge_pos(self, pos1: str, pos2: str) -> str:
        """
        Merge POS tags when a lemma has multiple tags.

        Args:
            pos1: First POS tag
            pos2: Second POS tag

        Returns:
            Combined POS tag (e.g., "NOUN/VERB")
        """
        if pos1 == pos2:
            return pos1

        tags = sorted(set([pos1, pos2]))
        return '/'.join(tags)

    def get_sorted_frequencies(self, min_frequency: int = 1) -> List[Dict[str, any]]:
        """
        Get lemma frequencies sorted by count (descending).

        Args:
            min_frequency: Minimum frequency threshold (default: 1)

        Returns:
            List of dictionaries with 'lemma', 'frequency', and 'pos' keys
        """
        results = []
        for lemma, count in self.lemma_counts.items():
            if count >= min_frequency:
                results.append({
                    'lemma': lemma,
                    'frequency': count,
                    'pos': self.lemma_pos.get(lemma, 'UNKNOWN')
                })

        results.sort(key=lambda x: x['frequency'], reverse=True)
        return results

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the frequency data.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_tokens': self.total_tokens,
            'unique_lemmas': len(self.lemma_counts),
            'most_common': sorted(self.lemma_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
