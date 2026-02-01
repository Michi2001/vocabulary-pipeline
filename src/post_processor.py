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

    Config options (in config.yaml under 'filtering'):
        accepted_pos: List of POS tags to include (e.g., ["VERB", "NOUN", "ADJ"])
        pos_mappings: Dict mapping POS tags (e.g., {"AUX": "VERB", "SCONJ": "CONJ"})
        include_countries: If true, include country names as allowed PROPNs
    """

    def __init__(self, filter_config: Dict[str, Any]):
        """Initialize post-processor with filter configuration."""
        self.accepted_pos = set(filter_config.get("accepted_pos", []))
        self.pos_mappings = filter_config.get("pos_mappings", {})

        self.allowed_propns = set()
        if filter_config.get("include_countries", False):
            country_names = self._load_country_names()
            self.allowed_propns = country_names
            logger.info(f"Loaded {len(country_names)} country names from babel")

        logger.info(
            f"PostProcessor initialized: "
            f"{len(self.accepted_pos)} accepted POS, "
            f"{len(self.pos_mappings)} mappings, "
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
        """Clean lemma by removing enclitic pronouns from multi-word lemmas."""
        if " " in lemma:
            lemma = re.sub(
                r"\s+(yo|tú|él|ella|nosotros|vosotros|ellos|ellas|lo|la|los|las|le|les|me|te|se)$",
                "",
                lemma,
            )
        return lemma

    def process(self, raw_counter: FrequencyCounter) -> FrequencyCounter:
        """Apply post-processing filters to raw frequency data."""
        logger.info("Starting post-processing of raw frequency data...")

        filtered_counter = FrequencyCounter()
        total_raw_lemmas = len(raw_counter.lemma_counts)
        skipped_count = 0

        for raw_key, count in raw_counter.lemma_counts.items():
            parts = raw_key.rsplit("|", 1)
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

            # Step 1b: Skip NUM lemmas containing digits (keep only written-out numbers)
            if original_pos == "NUM" and re.search(r"\d", lemma):
                skipped_count += 1
                continue

            # Step 2: Apply POS mapping (e.g., AUX→VERB)
            mapped_pos = self.pos_mappings.get(original_pos, original_pos)

            # Step 3: Apply lemma cleaning
            cleaned_lemma = self._clean_lemma(lemma)

            # Step 4: Create new key and add to filtered counter
            new_key = f"{cleaned_lemma}|{mapped_pos}"

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

        logger.info(
            f"Post-processing complete: "
            f"{total_raw_lemmas:,} raw lemmas → "
            f"{len(filtered_counter.lemma_counts):,} filtered lemmas "
            f"({skipped_count:,} skipped by POS filter)"
        )

        return filtered_counter
