"""LLM-based lemma translator with caching support."""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider
from src.lemma_cache import LemmaCache
from src.vocab import Translation

logger = logging.getLogger(__name__)


class LemmaTranslator:
    """
    Orchestrates LLM-based lemma translation from Spanish to German.

    Processes lemmas in batches using a single unified prompt,
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

        # Load prompt
        system_prompt = self._load_prompt()
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]

        # Token usage tracking
        total_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "cache_hits": 0,
        }

        # Assign IDs and normalize POS for all items
        items = []
        id_mapping = {}
        for idx, item in enumerate(frequencies):
            item_id = f"{idx + 1:06d}"
            pos_parts = item["pos"].split("/")
            normalized_pos = pos_parts[0]
            key = f"{item['lemma']}|{normalized_pos}"

            id_mapping[item_id] = {
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": item["pos"],
                "key": key,
            }
            items.append({
                "id": item_id,
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": normalized_pos,
            })

        # Pre-load cache
        cached_translations, uncached_items = self._preload_cache(
            items, prompt_hash
        )
        total_usage["cache_hits"] = len(cached_translations)

        # Create batches for uncached items
        batches = self._create_batches(uncached_items)
        if uncached_items:
            logger.info(
                f"Created {len(batches)} batches for {len(uncached_items)} uncached items"
            )
        else:
            logger.info("All items loaded from cache - no batches needed")

        # Process uncached batches
        batch_translations = self._process_batches(
            batches, system_prompt, prompt_hash, total_usage
        )

        all_translations = batch_translations + cached_translations

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

    def _load_prompt(self) -> str:
        """Load the translation prompt."""
        prompt_file = self.prompts_dir / "prompt.txt"
        if not prompt_file.exists():
            # Fallback to old general_prompt.txt for backwards compatibility
            prompt_file = self.prompts_dir / "general_prompt.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Translation prompt not found in {self.prompts_dir} "
                f"(tried prompt.txt and general_prompt.txt)"
            )
        return prompt_file.read_text(encoding="utf-8")

    def _create_batches(
        self, items: List[Dict[str, Any]]
    ) -> List[List[Dict]]:
        """Create batches from items."""
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

    def _preload_cache(
        self,
        items: List[Dict[str, Any]],
        prompt_hash: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pre-load cache and split items into cached and uncached."""
        cached_translations = []
        uncached_items = []

        logger.info("Pre-loading cache...")

        for item in items:
            cached_translation = None
            if self.cache is not None:
                cached_translation = self.cache.get_translation(
                    item["lemma"],
                    item["pos"],
                    self.model_config,
                    prompt_hash,
                )

            if cached_translation is not None:
                cached_translation["id"] = item["id"]
                cached_translations.append(cached_translation)
            else:
                uncached_items.append(item)

        logger.info(
            f"Cache: {len(cached_translations)} hits, {len(uncached_items)} misses"
        )

        return cached_translations, uncached_items

    def _process_batches(
        self,
        batches: List[List[Dict[str, Any]]],
        system_prompt: str,
        prompt_hash: str,
        total_usage: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Process uncached batches via LLM."""
        all_translations = []

        for i, batch in enumerate(batches):
            logger.info(
                f"Processing batch {i+1}/{len(batches)} ({len(batch)} lemmas)"
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
                            prompt_hash,
                            translation,
                        )
                self.cache.save_all()

            all_translations.extend(batch_translations)

        return all_translations

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
