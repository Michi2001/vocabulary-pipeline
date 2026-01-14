"""Updates estimated_share values based on actual corpus usage after disambiguation."""

import logging
from collections import defaultdict
from typing import Dict, List

from src.stores.disambiguated_store import DisambiguatedStore
from src.vocab import Translation, Vocabulary

logger = logging.getLogger(__name__)


class ShareUpdater:
    """Updates estimated_share values based on actual corpus usage."""

    def update(
        self,
        disambiguated_stores: List[DisambiguatedStore],
        translations: List[Translation],
        vocabulary: Vocabulary,
    ) -> List[Translation]:
        """Count variant occurrences and update estimated_share values in-place."""
        variant_counts: Dict[int, int] = defaultdict(int)  # variant_id -> count

        for store in disambiguated_stores:
            for _, variants in store:
                for variant_id in variants:
                    if variant_id is not None:
                        variant_counts[variant_id] += 1

        # Group counts by translation
        translation_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # translation_id -> meaning_idx -> count
        for variant_id, count in variant_counts.items():
            variant = vocabulary.word_variants[variant_id]
            translation_counts[variant.source_translation_id][
                variant.meaning_idx
            ] += count

        updated_count = 0
        for translation in translations:
            counts = translation_counts.get(translation.id, {})  # meaning_idx -> count
            total = sum(counts.values())

            if total == 0:
                continue

            if total != translation.frequency:
                logger.warning(
                    f"Lemma '{translation.lemma}' (ID: {translation.id}): "
                    f"disambiguated count {total} != frequency {translation.frequency}"
                )

            for idx, meaning in enumerate(translation.meanings):
                meaning.estimated_share = counts.get(idx, 0) / total
            updated_count += 1

        logger.info(f"Updated estimated_share for {updated_count} translations")

        return translations
