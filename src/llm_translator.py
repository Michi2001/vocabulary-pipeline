"""LLM-based lemma translator with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider
from src.lemma_cache import LemmaCache

logger = logging.getLogger(__name__)

POS_PROMPT_MAP = {
    "VERB": "verb_prompt.txt",
    "NOUN": "noun_prompt.txt",
    "ADJ": "adj_prompt.txt",
    "ADV": "adv_prompt.txt",
    "ADP": "adp_prompt.txt",
    "CONJ": "conj_prompt.txt",
}


class LemmaTranslator:
    """
    Orchestrates LLM-based lemma translation from Spanish to German.

    Processes lemmas by POS category sequentially, uses POS-specific prompts,
    outputs JSON file with full translation data enriched with frequencies.
    """

    def __init__(
        self, provider: BaseLLMProvider, cache: Optional[LemmaCache], config: Dict[str, Any]
    ):
        """
        Initialize translator.

        Args:
            provider: LLM provider instance
            cache: Shared LemmaCache instance (None to disable caching)
            config: {
                'batch_size': int,
                'prompts_dir': str,
                'output_dir': str
            }
        """
        self.provider = provider
        self.cache = cache
        self.model_config = provider.get_cache_key_components()
        self.batch_size = config.get("batch_size", 25)
        self.prompts_dir = Path(config.get("prompts_dir", "prompts/translation"))
        self.output_dir = Path(config["output_dir"])

    def translate(
        self, counter: FrequencyCounter, min_frequency: int = 1
    ) -> Dict[str, Any]:
        """
        Translate lemmas and write to JSON file.

        Args:
            counter: Input FrequencyCounter with Spanish lemmas (potentially already corrected)
            min_frequency: Minimum frequency threshold

        Returns:
            Dict with token usage statistics: {
                'total_prompt_tokens': int,
                'total_completion_tokens': int,
                'total_tokens': int,
                'api_calls': int,
                'cache_hits': int
            }
        """
        logger.info("Starting LLM-based lemma translation...")

        frequencies = counter.get_sorted_frequencies(min_frequency=min_frequency)
        logger.info(f"Processing {len(frequencies)} lemmas for translation")

        # Group by POS
        pos_groups = self._group_by_pos(frequencies)

        # Load general prompt once
        general_prompt = self._load_general_prompt()
        general_hash = hashlib.sha256(general_prompt.encode()).hexdigest()[:8]

        all_translations = []
        id_mapping = {}
        id_counter = 1

        # Token usage tracking
        total_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "cache_hits": 0,
        }

        # Process each POS sequentially
        for pos in ["VERB", "NOUN", "ADJ", "ADV", "ADP", "CONJ"]:
            if pos not in pos_groups:
                continue

            pos_items = pos_groups[pos]
            logger.info(f"Processing {len(pos_items)} {pos} lemmas")

            # Check if prompt file exists
            if pos not in POS_PROMPT_MAP:
                logger.warning(f"No prompt file for POS {pos}, skipping")
                continue

            pos_prompt_file = self.prompts_dir / POS_PROMPT_MAP[pos]
            if not pos_prompt_file.exists():
                logger.warning(
                    f"Prompt file not found: {pos_prompt_file}, skipping {pos}"
                )
                continue

            # Load POS-specific prompt
            pos_prompt = pos_prompt_file.read_text(encoding="utf-8")
            pos_hash = hashlib.sha256(pos_prompt.encode()).hexdigest()[:8]
            system_prompt = f"{general_prompt}\n\n{pos_prompt}"

            # Compute combined prompt hash for cache key
            combined_prompt_hash = hashlib.sha256(
                f"{general_hash}|{pos_hash}".encode()
            ).hexdigest()[:8]

            # Pre-load cache and create ID mapping for this POS
            pos_id_mapping, cached_translations, uncached_items = (
                self._preload_cache_for_pos(
                    pos_items, pos, id_counter, combined_prompt_hash
                )
            )

            # Track cache hits in total usage
            total_usage["cache_hits"] += len(cached_translations)

            # Update global id_mapping and counter
            id_mapping.update(pos_id_mapping)
            id_counter += len(pos_items)

            # Create batches only for uncached items
            batches = self._create_batches_from_items(uncached_items)
            if uncached_items:
                logger.info(
                    f"  Created {len(batches)} batches for {len(uncached_items)} uncached {pos} items"
                )
            else:
                logger.info(f"  All {pos} items loaded from cache - no batches needed")

            # Process uncached batches and combine with cached translations
            batch_translations = self._process_batches_for_pos(
                batches, pos, system_prompt, combined_prompt_hash, total_usage
            )
            all_translations.extend(batch_translations)
            all_translations.extend(cached_translations)

        # Enrich with frequencies
        enriched_translations = self._enrich_with_frequency(
            all_translations, id_mapping
        )

        # Sort by frequency descending (highest frequency first)
        enriched_translations.sort(key=lambda x: x.get("frequency", 0), reverse=True)

        # Save to JSON
        output_path = self.output_dir / "translations.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_translations, f, ensure_ascii=False, indent=2)

        # Save cache to disk (if cache is enabled)
        if self.cache is not None:
            self.cache.save_all()

        logger.info(
            f"Translation complete. Translated {len(enriched_translations)} lemmas"
        )
        logger.info(f"Output saved to: {output_path}")

        # Log token usage summary
        logger.info("\n" + "=" * 60)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"API Calls:          {total_usage['api_calls']}")
        logger.info(f"Cache Hits:         {total_usage['cache_hits']}")
        logger.info(f"Prompt Tokens:      {total_usage['total_prompt_tokens']:,}")
        logger.info(f"Completion Tokens:  {total_usage['total_completion_tokens']:,}")
        logger.info(f"Total Tokens:       {total_usage['total_tokens']:,}")
        logger.info("=" * 60 + "\n")

        return total_usage

    def _group_by_pos(
        self, frequencies: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group lemmas by normalized POS category."""
        groups = defaultdict(list)

        for item in frequencies:
            # Normalize POS (take first part before "/")
            pos_parts = item["pos"].split("/")
            normalized_pos = pos_parts[0]

            groups[normalized_pos].append(item)

        return dict(groups)

    def _load_general_prompt(self) -> str:
        """Load general translation prompt."""
        general_file = self.prompts_dir / "general_prompt.txt"
        if not general_file.exists():
            raise FileNotFoundError(f"General prompt not found: {general_file}")

        return general_file.read_text(encoding="utf-8")

    def _create_batches_from_items(
        self, items: List[Dict[str, Any]]
    ) -> List[List[Dict]]:
        """
        Create batches from items that already have IDs assigned.

        Args:
            items: List of items with 'id', 'lemma', 'frequency', 'pos' fields

        Returns:
            List of batches, where each batch is a list of items
        """
        batches = []
        batch_items = []

        for item in items:
            batch_items.append(item)

            # Create batch when size reached
            if len(batch_items) >= self.batch_size:
                batches.append(batch_items)
                batch_items = []

        # Add remaining items as final batch
        if batch_items:
            batches.append(batch_items)

        return batches

    def _preload_cache_for_pos(
        self,
        pos_items: List[Dict[str, Any]],
        pos: str,
        id_counter: int,
        combined_prompt_hash: str,
    ) -> tuple[Dict[str, Dict], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Pre-load cache and create ID mapping for one POS category.

        Args:
            pos_items: Items for this POS
            pos: POS category name
            id_counter: Starting ID counter
            combined_prompt_hash: Combined prompt hash for cache key

        Returns:
            Tuple of (pos_id_mapping, cached_translations, uncached_items)
        """
        pos_id_mapping = {}
        cached_translations = []
        uncached_items = []

        logger.info(f"  Pre-loading cache for {pos} lemmas...")

        for idx, item in enumerate(pos_items):
            # Generate ID (continuing from id_counter)
            item_id = f"{id_counter + idx:06d}"

            # Normalize POS
            pos_parts = item["pos"].split("/")
            normalized_pos = pos_parts[0]
            key = f"{item['lemma']}|{normalized_pos}"

            # Store in id_mapping
            pos_id_mapping[item_id] = {
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": item["pos"],
                "key": key,
            }

            # Try to load from cache (if cache is enabled)
            cached_translation = None
            if self.cache is not None:
                cached_translation = self.cache.get_translation(
                    item["lemma"], normalized_pos, self.model_config, combined_prompt_hash
                )

            if cached_translation is not None:
                cached_translations.append(cached_translation)
            else:
                # Add to uncached with ID and normalized POS
                uncached_items.append({
                    "id": item_id,
                    "lemma": item["lemma"],
                    "frequency": item["frequency"],
                    "pos": normalized_pos,
                })

        # Log statistics
        cache_hits = len(cached_translations)
        cache_misses = len(uncached_items)
        logger.info(
            f"  Pre-cache check for {pos}: {cache_hits} hits, {cache_misses} misses"
        )

        return pos_id_mapping, cached_translations, uncached_items

    def _process_batches_for_pos(
        self,
        batches: List[List[Dict[str, Any]]],
        pos: str,
        system_prompt: str,
        combined_prompt_hash: str,
        total_usage: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Process uncached batches for one POS category.

        Args:
            batches: Batches to process
            pos: POS category name
            system_prompt: System prompt for API call
            combined_prompt_hash: Combined prompt hash for cache key
            total_usage: Token usage tracking dict (modified in-place)

        Returns:
            List of translations from API calls
        """
        batch_translations_all = []

        for i, batch in enumerate(batches):
            logger.info(
                f"  Processing {pos} batch {i+1}/{len(batches)} ({len(batch)} uncached lemmas)"
            )

            # All items in batch are uncached by definition
            response = self.provider.translate_lemmas(batch, system_prompt)
            batch_translations = response["all_lemma_translations"]

            # Track token usage from API call
            if "_token_usage" in response:
                usage = response["_token_usage"]
                total_usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["total_completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_usage["api_calls"] += 1

            # Cache individual translations (if cache is enabled)
            if self.cache is not None:
                for translation in batch_translations:
                    # Find corresponding item to get lemma and pos
                    item_id = translation["id"]
                    matching_item = next(
                        (item for item in batch if item["id"] == item_id), None
                    )
                    if matching_item:
                        self.cache.set_translation(
                            matching_item["lemma"],
                            matching_item["pos"],
                            self.model_config,
                            combined_prompt_hash,
                            translation,
                        )

            batch_translations_all.extend(batch_translations)

        return batch_translations_all

    def _enrich_with_frequency(
        self,
        translations: List[Dict[str, Any]],
        id_mapping: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """Add frequency field to each translation object and normalize estimated_share values."""
        enriched = []

        for translation in translations:
            item_id = translation["id"]
            if item_id in id_mapping:
                # Add frequency from id_mapping
                translation["frequency"] = id_mapping[item_id]["frequency"]

            # Normalize estimated_share values
            self._normalize_estimated_shares(translation)

            enriched.append(translation)

        return enriched

    def _normalize_estimated_shares(self, translation: Dict[str, Any]) -> None:
        """
        Normalize estimated_share values so they sum to 1.0.

        Issues warning if normalization is needed.
        If sum is 0, distributes shares equally across all meanings.
        Modifies the translation dict in-place.
        """
        if "meanings" not in translation or not translation["meanings"]:
            return

        num_meanings = len(translation["meanings"])
        if num_meanings == 0:
            return

        # Calculate sum of estimated_share values
        total_share = sum(
            meaning.get("estimated_share", 0.0) for meaning in translation["meanings"]
        )

        # If sum is very close to 1.0, no normalization needed
        if abs(total_share - 1.0) < 0.001:
            return

        # If sum is 0, distribute equally
        if total_share == 0:
            logger.warning(
                f"Lemma {translation.get('lemma')} (ID: {translation.get('id')}): "
                f"All estimated_share values are 0 - distributing equally across {num_meanings} meanings"
            )
            equal_share = 1.0 / num_meanings
            for meaning in translation["meanings"]:
                meaning["estimated_share"] = equal_share
            return

        # Log warning and normalize proportionally
        logger.warning(
            f"Lemma {translation.get('lemma')} (ID: {translation.get('id')}): "
            f"estimated_share values sum to {total_share:.4f} instead of 1.0 - normalizing"
        )

        # Normalize each meaning's estimated_share proportionally
        for meaning in translation["meanings"]:
            if "estimated_share" in meaning:
                meaning["estimated_share"] = meaning["estimated_share"] / total_share
