"""LLM-based lemma translator with caching support."""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider
from src.lemma_cache import LemmaCache
from src.vocab import Translation, Meaning

logger = logging.getLogger(__name__)

# Regex for extracting bracketed target word from example sentences
BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")

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
        self,
        provider: BaseLLMProvider,
        cache: Optional[LemmaCache],
        config: Dict[str, Any],
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
    ) -> Tuple[List[Translation], Dict[str, int]]:
        """
        Translate lemmas and return translations.

        Args:
            counter: Input FrequencyCounter with Spanish lemmas (potentially already corrected)
            min_frequency: Minimum frequency threshold

        Returns:
            Tuple of (translations, usage_stats):
                - translations: List of Translation objects
                - usage_stats: Dict with token usage statistics
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

        # Validate and normalize translations
        self._extract_target_words_and_normalize_translations(enriched_translations)

        # Sort by frequency descending (highest frequency first)
        enriched_translations.sort(key=lambda x: x.get("frequency", 0), reverse=True)

        # Save cache to disk (if cache is enabled)
        if self.cache is not None:
            self.cache.save_all()

        # Convert dicts to Translation objects
        translation_objects = [
            Translation.from_dict(t) for t in enriched_translations
        ]

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
                    item["lemma"],
                    normalized_pos,
                    self.model_config,
                    combined_prompt_hash,
                )

            if cached_translation is not None:
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

                # Commit cache after each batch
                self.cache.save_all()

            batch_translations_all.extend(batch_translations)

        return batch_translations_all

    def _enrich_with_frequency(
        self,
        translations: List[Dict[str, Any]],
        id_mapping: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """
        Add frequency field to each translation object.

        Args:
            translations: List of translation dicts from LLM
            id_mapping: Mapping from translation ID to original lemma data

        Returns:
            Translation list with frequency added
        """
        for translation in translations:
            item_id = translation.get("id", "")

            # Add frequency from original data
            if item_id in id_mapping:
                translation["frequency"] = id_mapping[item_id]["frequency"]
            else:
                logger.warning(f"ID not found in mapping: {item_id}")
                translation["frequency"] = 0

        return translations

    def _extract_target_words_and_normalize_translations(
        self, translations: List[Dict[str, Any]]
    ) -> None:
        """
        Normalize estimated shares and extract target words from examples.

        Modifies translations in-place.

        Args:
            translations: List of translation dicts
        """
        for translation in translations:
            # Normalize estimated shares
            self._normalize_estimated_shares(translation)

            # Extract and process bracketed target words
            self._extract_target_words(translation)

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

    def _extract_target_words(self, translation: Dict[str, Any]) -> None:
        """
        Extract bracketed target words from example_es fields.

        For each meaning, checks that example_es contains a bracketed word [word].
        Extracts the bracketed word, removes brackets from example_es, and stores
        metadata (_target_word, _target_start, _target_end) for MeaningEmbeddingGenerator.
        Logs warnings for missing brackets but does not fail.
        """
        if "meanings" not in translation or not translation["meanings"]:
            return

        lemma = translation.get("lemma", "unknown")
        lemma_id = translation.get("id", "?")

        for idx, meaning in enumerate(translation["meanings"]):
            example_es = meaning.get("example_es", "")
            if not example_es:
                continue

            match = BRACKET_PATTERN.search(example_es)
            if not match:
                logger.warning(
                    f"Lemma '{lemma}' (ID: {lemma_id}), meaning {idx}: "
                    f"No bracketed word in example_es: '{example_es[:50]}...'"
                )
            else:
                # Extract bracketed word and position
                target_word = match.group(1)
                bracket_start = match.start()

                # Position in cleaned sentence (without brackets)
                char_start = bracket_start
                char_end = char_start + len(target_word)

                # Remove brackets from example_es
                cleaned_sentence = example_es.replace(
                    f"[{target_word}]", target_word, 1
                )
                meaning["example_es"] = cleaned_sentence

                # Store metadata for MeaningEmbeddingGenerator
                meaning["_target_word"] = target_word
                meaning["_target_start"] = char_start
                meaning["_target_end"] = char_end
