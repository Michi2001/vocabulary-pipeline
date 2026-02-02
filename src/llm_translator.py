"""LLM-based lemma translator with caching support."""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.frequency_counter import FrequencyCounter
from src.lemma_cache import LemmaCache
from src.llm_providers.base_provider import BaseLLMProvider
from src.vocab import Translation

logger = logging.getLogger(__name__)

POS_PROMPT_MAP = {
    "VERB": "verb_prompt.txt",
    "NOUN": "noun_prompt.txt",
    "ADJ": "adj_prompt.txt",
    "ADV": "adv_prompt.txt",
    "ADP": "adp_prompt.txt",
    "CONJ": "conj_prompt.txt",
    "PRON": "pron_prompt.txt",
    "INTJ": "intj_prompt.txt",
    "NUM": "num_prompt.txt",
}

# Order in which POS categories are processed
POS_ORDER = ["VERB", "NOUN", "ADJ", "ADV", "ADP", "CONJ", "PRON", "INTJ", "NUM"]


class LemmaTranslator:
    """
    Orchestrates LLM-based lemma translation from Spanish to German.

    Processes lemmas by POS category sequentially, uses POS-specific prompts,
    outputs JSON file with full translation data enriched with frequencies.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache: Optional[LemmaCache],
        config: Dict[str, Any],
    ):
        self.provider = provider
        self.cache = cache
        self.model_config = provider.get_cache_key_components()
        self.batch_size = config.get("batch_size", 25)
        self.prompts_dir = Path(config.get("prompts_dir", "prompts/translation"))
        self.output_dir = Path(config["output_dir"])

    def translate(
        self, counter: FrequencyCounter, min_frequency: int = 1
    ) -> Tuple[List[Translation], Dict[str, int]]:
        """Translate lemmas and return translations."""
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
        for pos in POS_ORDER:
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

        for translation in enriched_translations:
            self._set_equal_shares(translation)

        # Sort by frequency descending
        enriched_translations.sort(key=lambda x: x.get("frequency", 0), reverse=True)

        # Save cache to disk
        if self.cache is not None:
            self.cache.save_all()

        # Convert dicts to Translation objects
        translation_objects = [Translation.from_dict(t) for t in enriched_translations]

        logger.info(
            f"Translation complete. Translated {len(translation_objects)} lemmas"
        )

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

        return translation_objects, total_usage

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
        """Create batches from items that already have IDs assigned."""
        batches = []
        batch_items = []

        for item in items:
            batch_items.append(item)
            if len(batch_items) >= self.batch_size:
                batches.append(batch_items)
                batch_items = []

        if batch_items:
            batches.append(batch_items)

        return batches

    def _preload_cache_for_pos(
        self,
        pos_items: List[Dict[str, Any]],
        pos: str,
        id_counter: int,
        combined_prompt_hash: str,
    ) -> Tuple[Dict[str, Dict], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pre-load cache and create ID mapping for one POS category."""
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

            # Try to load from cache
            cached_translation = None
            if self.cache is not None:
                cached_translation = self.cache.get_translation(
                    item["lemma"],
                    normalized_pos,
                    self.model_config,
                    combined_prompt_hash,
                )

            if cached_translation is not None:
                # Update ID to current run's ID (cache may have old ID)
                cached_translation["id"] = item_id
                cached_translations.append(cached_translation)
            else:
                # Add to uncached with ID and normalized POS
                uncached_items.append(
                    {
                        "id": item_id,
                        "lemma": item["lemma"],
                        "frequency": item["frequency"],
                        "pos": normalized_pos,
                    }
                )

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
        """Process uncached batches for one POS category."""
        batch_translations_all = []

        for i, batch in enumerate(batches):
            logger.info(
                f"  Processing {pos} batch {i+1}/{len(batches)} ({len(batch)} uncached lemmas)"
            )

            response = self.provider.translate_lemmas(batch, system_prompt)
            batch_translations = response["all_lemma_translations"]

            # Track token usage
            if "_token_usage" in response:
                usage = response["_token_usage"]
                total_usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["total_completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_usage["api_calls"] += 1

            # Cache individual translations
            if self.cache is not None:
                for translation in batch_translations:
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
                self.cache.save_all()

            batch_translations_all.extend(batch_translations)

        return batch_translations_all

    def _enrich_with_frequency(
        self,
        translations: List[Dict[str, Any]],
        id_mapping: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """Add frequency field to each translation object."""
        for translation in translations:
            item_id = translation.get("id", "")
            if item_id in id_mapping:
                translation["frequency"] = id_mapping[item_id]["frequency"]
            else:
                logger.warning(f"ID not found in mapping: {item_id}")
                translation["frequency"] = 0

        return translations

    def _set_equal_shares(self, translation: Dict[str, Any]) -> None:
        """Set equal estimated_share (1/n) as placeholder until disambiguation."""
        if "meanings" not in translation or not translation["meanings"]:
            return

        num_meanings = len(translation["meanings"])
        if num_meanings == 0:
            return

        equal_share = 1.0 / num_meanings
        for meaning in translation["meanings"]:
            meaning["estimated_share"] = equal_share
