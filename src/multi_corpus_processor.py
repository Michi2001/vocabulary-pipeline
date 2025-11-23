import logging
from typing import List

from src.frequency_counter import FrequencyCounter
from src.single_corpus_processor import SingleCorpusProcessor

logger = logging.getLogger(__name__)


class MultiCorpusProcessor:
    """
    Processes multiple corpora sequentially and merges their frequencies.
    """

    def __init__(self, processors: List[SingleCorpusProcessor]):
        """
        Initialize the multi-corpus processor.

        Args:
            processors: List of SingleCorpusProcessor instances
        """
        self.processors = processors

    def process(self) -> FrequencyCounter:
        """
        Process all corpora sequentially and merge results.

        Returns:
            FrequencyCounter with merged frequencies from all corpora
        """
        logger.info(f"Processing {len(self.processors)} corpora sequentially...")

        merged_counter = FrequencyCounter()

        for idx, processor in enumerate(self.processors, 1):
            corpus_name = processor.corpus_name
            logger.info(
                f"\n{'='*60}\n"
                f"Processing corpus {idx}/{len(self.processors)}: {corpus_name}\n"
                f"{'='*60}"
            )

            # Process single corpus
            counter = processor.process()

            # Merge into combined counter
            merged_counter.merge(counter)

            logger.info(
                f"After merging {corpus_name}: "
                f"{merged_counter.total_tokens:,} total tokens, "
                f"{len(merged_counter.lemma_counts):,} unique lemmas"
            )

        logger.info(
            f"\n{'='*60}\n"
            f"Completed all {len(self.processors)} corpora\n"
            f"{'='*60}"
        )

        final_stats = merged_counter.get_statistics()
        logger.info(
            f"Final totals: {final_stats['total_tokens']:,} tokens, "
            f"{final_stats['unique_lemmas']:,} unique lemmas"
        )

        return merged_counter
