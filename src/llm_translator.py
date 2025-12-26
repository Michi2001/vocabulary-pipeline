"""LLM-based lemma translator with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider

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

    def __init__(self, provider: BaseLLMProvider, config: Dict[str, Any]):
        """
        Initialize translator.

        Args:
            config: {
                'batch_size': int,
                'prompts_dir': str,
                'output_dir': str,
                'cache_config': dict,
                'llm_model': str
            }
        """
        self.provider = provider
        self.batch_size = config.get("batch_size", 25)
        self.prompts_dir = Path(config.get("prompts_dir", "prompts/translation"))
        self.output_dir = Path(config["output_dir"])
        self.cache_config = config.get("cache_config", {})
        self.llm_model = config.get("llm_model", "gpt-5-mini")

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

            # Create batches for this POS
            pos_id_mapping, batches = self._create_batches_for_pos(
                pos_items, id_counter
            )
            id_mapping.update(pos_id_mapping)
            id_counter += len(pos_items)

            logger.info(f"Created {len(batches)} batches for {pos}")

            # Process batches with caching
            for i, batch in enumerate(batches):
                logger.info(
                    f"  Processing {pos} batch {i+1}/{len(batches)} ({len(batch)} lemmas)"
                )

                batch_cache_key = self._generate_batch_cache_key(
                    batch, pos, general_hash, pos_hash
                )

                # Try cache
                batch_translations = None
                if self.cache_config.get("enabled", True):
                    batch_translations = self._load_from_cache(batch_cache_key)

                if batch_translations is None:
                    response = self.provider.translate_lemmas(batch, system_prompt)
                    batch_translations = response["all_lemma_translations"]

                    # Track token usage from API call
                    if "_token_usage" in response:
                        usage = response["_token_usage"]
                        total_usage["total_prompt_tokens"] += usage.get(
                            "prompt_tokens", 0
                        )
                        total_usage["total_completion_tokens"] += usage.get(
                            "completion_tokens", 0
                        )
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                        total_usage["api_calls"] += 1

                    if self.cache_config.get("enabled", True):
                        self._save_to_cache(batch_cache_key, batch_translations)
                else:
                    logger.info(f"    Cache hit - loaded batch {i+1} from cache")
                    total_usage["cache_hits"] += 1

                all_translations.extend(batch_translations)

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

    def _create_batches_for_pos(
        self, pos_items: List[Dict[str, Any]], start_id: int
    ) -> Tuple[Dict[str, Dict], List[List[Dict]]]:
        """
        Create ID mapping and batches for one POS category.

        Args:
            pos_items: List of frequency items for this POS
            start_id: Starting ID number

        Returns:
            (id_mapping, batches)
            id_mapping: Dict mapping item IDs to their data
            batches: List of batches (each batch is a list of dicts)
        """
        id_mapping = {}
        batch_items = []
        batches = []

        for idx, item in enumerate(pos_items):
            item_id = f"{start_id + idx:06d}"

            # Normalize POS
            pos_parts = item["pos"].split("/")
            normalized_pos = pos_parts[0]
            key = f"{item['lemma']}|{normalized_pos}"

            id_mapping[item_id] = {
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": item["pos"],
                "key": key,
            }

            batch_items.append(
                {
                    "id": item_id,
                    "lemma": item["lemma"],
                    "frequency": item["frequency"],
                    "pos": normalized_pos,
                }
            )

            if len(batch_items) >= self.batch_size:
                batches.append(batch_items)
                batch_items = []

        if batch_items:
            batches.append(batch_items)

        return id_mapping, batches

    def _generate_batch_cache_key(
        self, batch: List[Dict[str, Any]], pos: str, general_hash: str, pos_hash: str
    ) -> str:
        """Generate cache key for batch."""
        batch_strings = [
            f"{item['id']}|{item['lemma']}|{item['frequency']}|{item['pos']}"
            for item in batch
        ]

        key_components = [
            "|".join(batch_strings),
            self.llm_model,
            general_hash,
            pos_hash,
            pos,
        ]

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> List[Dict[str, Any]]:
        """Load translations from cache."""
        cache_dir = Path(self.cache_config.get("directory", "cache/llm_translations"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                translations = json.load(f)
            return translations
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(
        self, cache_key: str, translations: List[Dict[str, Any]]
    ) -> None:
        """Save translations to cache."""
        cache_dir = Path(self.cache_config.get("directory", "cache/llm_translations"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

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
            meaning.get("estimated_share", 0.0)
            for meaning in translation["meanings"]
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
