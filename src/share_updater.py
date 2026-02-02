"""Updates estimated_share values based on actual corpus usage after disambiguation."""

import logging
from collections import defaultdict
from typing import Dict, List, Set

from src.stores.disambiguated_store import DisambiguatedStore
from src.vocab import Translation, Vocabulary

logger = logging.getLogger(__name__)


class ShareUpdater:
    """
    Updates estimated_share values based on actual corpus usage.

    Only sets estimated_share for disambiguated POS types (NOUN, VERB, ADJ, ADV).
    Non-disambiguated POS types do not get estimated_share values.
    """

    def update(
        self,
        disambiguated_stores: List[DisambiguatedStore],
        translations: List[Translation],
        vocabulary: Vocabulary,
    ) -> List[Translation]:
        """Count entry occurrences and set estimated_share for disambiguated entries."""
        disambiguated_pos = vocabulary.disambiguated_pos
        entry_counts: Dict[int, int] = defaultdict(int)  # entry_id -> count

        for store in disambiguated_stores:
            for _, entries in store:
                for entry_id in entries:
                    if entry_id is not None:
                        entry_counts[entry_id] += 1

        # Group counts by translation (only for disambiguated entries)
        translation_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # translation_id -> meaning_idx -> count
        for entry_id, count in entry_counts.items():
            entry = vocabulary.vocab_entries[entry_id]
            if entry.meaning_idx is not None:  # Only disambiguated entries
                translation_counts[entry.source_translation_id][
                    entry.meaning_idx
                ] += count

        updated_count = 0
        for translation in translations:
            # Skip non-disambiguated POS types
            if translation.pos not in disambiguated_pos:
                continue

            counts = translation_counts.get(translation.id, {})  # meaning_idx -> count
            total = sum(counts.values())

            if total == 0:
                continue

            # Set estimated_share for each meaning
            for idx, meaning in enumerate(translation.meanings):
                meaning.estimated_share = counts.get(idx, 0) / total
            updated_count += 1

        logger.info(
            f"Set estimated_share for {updated_count} disambiguated translations"
        )

        return translations
