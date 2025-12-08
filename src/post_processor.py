import re
import logging
from typing import Dict, Any

from babel import Locale

from src.frequency_counter import FrequencyCounter

logger = logging.getLogger(__name__)


class FrequencyPostProcessor:
    """
    Post-processes raw frequency data by applying filters and transformations.

    Transformations include:
    - POS filtering (which categories to include)
    - POS mapping (e.g., AUX→VERB, CCONJ/SCONJ→CONJ)
    - Lemma cleaning (removing enclitic pronouns)
    - Frequency merging (combining mapped POS categories)
    """

    def __init__(self, filter_config: Dict[str, Any]):
        """
        Initialize post-processor with filter configuration.

        Args:
            filter_config: {
                'accepted_pos': list of POS tags to include (None = accept all)
                'pos_mappings': dict mapping original POS to target POS
                'main_pos_categories': list for final grouping
                'clean_lemmas': bool to enable/disable lemma cleaning
                'include_countries': bool to automatically include all country names from babel
            }
        """
        self.accepted_pos = set(filter_config.get("accepted_pos", []))
        self.pos_mappings = filter_config.get("pos_mappings", {})
        self.main_pos_categories = set(filter_config.get("main_pos_categories", []))
        self.clean_lemmas = filter_config.get("clean_lemmas", True)

        self.allowed_propns = set()
        if filter_config.get("include_countries", False):
            country_names = self._load_country_names()
            self.allowed_propns = country_names
            logger.info(f"Loaded {len(country_names)} country names from babel")

        logger.info(
            f"PostProcessor initialized: "
            f"{len(self.accepted_pos)} accepted POS, "
            f"{len(self.pos_mappings)} mappings, "
            f"{len(self.main_pos_categories)} main categories, "
            f"cleaning={'enabled' if self.clean_lemmas else 'disabled'}, "
            f"{len(self.allowed_propns)} allowed country PROPNs"
        )

    def _load_country_names(self) -> set:
        """Load country names from babel for Spanish locale."""
        locale = Locale("es")
        country_names = [
            name for code, name in locale.territories.items() if code.isalpha()
        ]
        return set(country_names)

    def _clean_lemma(self, lemma: str) -> str:
        """
        Clean lemma by removing multi-word artifacts.

        Args:
            lemma: Raw lemma

        Returns:
            Cleaned single-word lemma
        """
        if not self.clean_lemmas:
            return lemma

        # Only clean if it's a multi-word lemma
        if " " in lemma:
            # Remove enclitic pronouns that spaCy sometimes adds
            lemma = re.sub(
                r"\s+(yo|tú|él|ella|nosotros|vosotros|ellos|ellas|lo|la|los|las|le|les|me|te|se)$",
                "",
                lemma,
            )

        return lemma

    def _normalize_pos(self, pos: str) -> str:
        """
        Normalize POS to main category or OTHER.

        Args:
            pos: POS tag

        Returns:
            Normalized POS (main category or "OTHER")
        """
        if pos in self.main_pos_categories:
            return pos
        else:
            return "OTHER"

    def process(self, raw_counter: FrequencyCounter) -> FrequencyCounter:
        """
        Apply post-processing filters to raw frequency data.

        Steps:
        1. Iterate through raw_counter.lemma_counts (all original POS)
        2. Apply POS filtering (skip if not in accepted_pos)
        3. Apply POS mapping (e.g., AUX→VERB)
        4. Apply lemma cleaning if enabled
        5. Create new key with cleaned lemma and mapped POS
        6. Normalize POS to main categories
        7. Aggregate frequencies in new FrequencyCounter

        Args:
            raw_counter: FrequencyCounter with raw, unfiltered data

        Returns:
            New FrequencyCounter with filtered/transformed data
        """
        logger.info("Starting post-processing of raw frequency data...")

        # Create new counter for filtered data
        filtered_counter = FrequencyCounter()

        # Process known words
        total_raw_lemmas = len(raw_counter.lemma_counts)
        filtered_count = 0
        skipped_count = 0

        for raw_key, count in raw_counter.lemma_counts.items():
            # Parse raw key: "lemma|POS"
            parts = raw_key.split("|")
            if len(parts) != 2:
                logger.warning(f"Invalid key format: {raw_key}")
                continue

            lemma, original_pos = parts

            # Step 1: Apply POS filtering
            if self.accepted_pos and original_pos not in self.accepted_pos:
                if original_pos == "PROPN" and lemma in self.allowed_propns:
                    pass
                else:
                    skipped_count += 1
                    continue

            # Step 2: Apply POS mapping
            mapped_pos = self.pos_mappings.get(original_pos, original_pos)

            # Step 3: Apply lemma cleaning
            cleaned_lemma = self._clean_lemma(lemma)

            # Step 4: Normalize POS
            normalized_pos = self._normalize_pos(mapped_pos)

            # Step 5: Create new key and add to filtered counter
            new_key = f"{cleaned_lemma}|{normalized_pos}"

            # Add to filtered counter
            filtered_counter.lemma_counts[new_key] += count
            filtered_counter.total_tokens += count

            # Track original POS details from raw counter
            if new_key not in filtered_counter.lemma_pos_details:
                filtered_counter.lemma_pos_details[new_key] = set()

            if raw_key in raw_counter.lemma_pos_details:
                filtered_counter.lemma_pos_details[new_key].update(
                    raw_counter.lemma_pos_details[raw_key]
                )

            filtered_count += 1

        # Process OOV words (same logic)
        for raw_key, count in raw_counter.oov_counts.items():
            parts = raw_key.split("|")
            if len(parts) != 2:
                continue

            lemma, original_pos = parts

            if self.accepted_pos and original_pos not in self.accepted_pos:
                # Special handling for PROPNs: check if it's in allowed list
                if original_pos == "PROPN" and lemma in self.allowed_propns:
                    # Allow this specific PROPN
                    pass
                else:
                    continue

            mapped_pos = self.pos_mappings.get(original_pos, original_pos)
            cleaned_lemma = self._clean_lemma(lemma)
            normalized_pos = self._normalize_pos(mapped_pos)
            new_key = f"{cleaned_lemma}|{normalized_pos}"

            filtered_counter.oov_counts[new_key] += count
            filtered_counter.total_oov_tokens += count

            if new_key not in filtered_counter.oov_pos_details:
                filtered_counter.oov_pos_details[new_key] = set()

            if raw_key in raw_counter.oov_pos_details:
                filtered_counter.oov_pos_details[new_key].update(
                    raw_counter.oov_pos_details[raw_key]
                )

        logger.info(
            f"Post-processing complete: "
            f"{total_raw_lemmas:,} raw lemmas → "
            f"{len(filtered_counter.lemma_counts):,} filtered lemmas "
            f"({skipped_count:,} skipped by POS filter)"
        )

        return filtered_counter
