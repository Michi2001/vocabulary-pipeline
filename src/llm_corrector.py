"""LLM-based lemma corrector with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import csv

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class LemmaCorrector:
    """
    Orchestrates LLM-based lemma correction.

    Takes FrequencyCounter, sends lemmas to LLM in batches,
    applies corrections, and generates audit outputs.
    """

    def __init__(self, provider: BaseLLMProvider, config: Dict[str, Any]):
        """
        Initialize corrector.

        Args:
            config: {
                'batch_size': int (default: 50),
                'system_prompt': str,
                'output_dir': str (for audit files),
                'cache_config': dict (enabled, directory),
                'llm_model': str (for cache key)
            }
        """
        self.provider = provider
        self.batch_size = config.get("batch_size", 50)
        self.system_prompt = config["system_prompt"]
        self.output_dir = Path(config["output_dir"])
        self.cache_config = config.get("cache_config", {})
        self.llm_model = config.get("llm_model", "gpt-4o-mini")

    def correct(
        self, counter: FrequencyCounter, min_frequency: int = 1
    ) -> FrequencyCounter:
        """
        Apply LLM corrections to FrequencyCounter.

        Args:
            counter: Input FrequencyCounter
            min_frequency: Minimum frequency threshold (same as output filter)

        Returns:
            Corrected FrequencyCounter
        """
        logger.info("Starting LLM-based lemma correction...")

        # Step 1: Get sorted frequencies (only known words)
        frequencies = counter.get_sorted_frequencies(
            min_frequency=min_frequency, oov=False
        )
        logger.info(f"Processing {len(frequencies)} lemmas for LLM correction")

        # Step 2: Assign IDs and create batches
        id_mapping, batches = self._create_batches(frequencies)
        logger.info(f"Created {len(batches)} batches of size {self.batch_size}")

        # Step 3: Process batches with per-batch caching
        all_corrections = []
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} lemmas)")

            batch_cache_key = self._generate_batch_cache_key(batch)

            # Try to load from cache
            batch_corrections = None
            if self.cache_config.get("enabled", True):
                batch_corrections = self._load_corrections_from_cache(batch_cache_key)

            if batch_corrections is None:
                response = self.provider.correct_lemmas(batch, self.system_prompt)
                batch_corrections = response["corrections"]

                if self.cache_config.get("enabled", True):
                    self._save_corrections_to_cache(batch_cache_key, batch_corrections)
            else:
                logger.info(f"  Cache hit - loaded batch {i+1} from cache")

            all_corrections.extend(batch_corrections)

        # Step 6: Save original list with IDs
        original_path = self.output_dir / "original_with_ids.csv"
        self._save_original(id_mapping, original_path)

        # Step 7: Save correction mapping
        mapping_path = self.output_dir / "llm_corrections.csv"
        self._save_corrections(all_corrections, mapping_path)

        # Step 8: Apply corrections to counter
        corrected_counter = self._apply_corrections(
            counter, id_mapping, all_corrections
        )

        logger.info(
            f"LLM correction complete. Applied {len(all_corrections)} operations."
        )

        return corrected_counter

    def _generate_batch_cache_key(self, batch: List[Dict[str, Any]]) -> str:
        """
        Generate SHA256 hash for a single batch cache key.

        Key components:
        - Batch items (id, lemma, frequency, pos)
        - LLM model name
        - System prompt hash

        Args:
            batch: Single batch with items containing 'id', 'lemma', 'frequency', 'pos'

        Returns:
            SHA256 hash string
        """
        # Serialize batch items (order matters since IDs are sequential)
        batch_strings = [
            f"{item['id']}|{item['lemma']}|{item['frequency']}|{item['pos']}"
            for item in batch
        ]

        # Hash system prompt to detect prompt changes
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]

        key_components = ["|".join(batch_strings), self.llm_model, prompt_hash]

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _load_corrections_from_cache(self, cache_key: str) -> List[Dict[str, Any]]:
        """
        Load cached LLM corrections.

        Args:
            cache_key: Cache key hash

        Returns:
            List of corrections or None if cache miss
        """
        cache_dir = Path(self.cache_config.get("directory", "cache/llm_corrections"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                corrections = json.load(f)

            logger.info(f"Loaded corrections from cache (key: {cache_key[:8]}...)")
            return corrections

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_corrections_to_cache(
        self, cache_key: str, corrections: List[Dict[str, Any]]
    ) -> None:
        """
        Save LLM corrections to cache.

        Args:
            cache_key: Cache key hash
            corrections: List of correction operations
        """
        cache_dir = Path(self.cache_config.get("directory", "cache/llm_corrections"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(corrections, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved corrections to cache (key: {cache_key[:8]}...)")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _create_batches(
        self, frequencies: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict], List[List[Dict]]]:
        """
        Assign IDs and create batches.

        Args:
            frequencies: List from get_sorted_frequencies()

        Returns:
            (id_mapping, batches)
            id_mapping: {"001": {"lemma": "...", "frequency": ..., "pos": "...", "key": "lemma|POS"}, ...}
            batches: [[{"id": "001", "lemma": "...", "frequency": ..., "pos": "..."}, ...], ...]
        """
        id_mapping = {}
        batch_items = []
        batches = []

        for idx, item in enumerate(frequencies):
            # Generate ID (zero-padded to 6 digits for up to 999,999 lemmas)
            item_id = f"{idx+1:06d}"

            # Create key for FrequencyCounter lookup
            # First POS is always the normalized main category (e.g., "VERB/AUX" -> "VERB")
            pos_parts = item["pos"].split("/")
            normalized_pos = pos_parts[0]
            key = f"{item['lemma']}|{normalized_pos}"

            # Store in mapping
            id_mapping[item_id] = {
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": item["pos"],
                "key": key,
            }

            # Add to batch
            batch_items.append(
                {
                    "id": item_id,
                    "lemma": item["lemma"],
                    "frequency": item["frequency"],
                    "pos": normalized_pos,  # Only main POS, not "VERB/AUX"
                }
            )

            # Create batch when size reached
            if len(batch_items) >= self.batch_size:
                batches.append(batch_items)
                batch_items = []

        # Add remaining items as final batch
        if batch_items:
            batches.append(batch_items)

        return id_mapping, batches

    def _apply_corrections(
        self,
        counter: FrequencyCounter,
        id_mapping: Dict[str, Dict],
        corrections: List[Dict[str, Any]],
    ) -> FrequencyCounter:
        """
        Apply corrections to create new FrequencyCounter.

        Three-pass algorithm:
        1. Build deletion and merge maps
        2. Process KEEP operations
        3. Process MERGE operations with frequency accumulation

        Args:
            counter: Original FrequencyCounter
            id_mapping: ID to lemma mapping
            corrections: List of correction operations from LLM

        Returns:
            New corrected FrequencyCounter
        """
        corrected_counter = FrequencyCounter()

        # Track which IDs are deleted or merged (source)
        deleted_ids = set()
        merge_sources = {}  # source_id -> target_key

        # First pass: build deletion and merge maps
        for correction in corrections:
            item_id = correction["id"]
            operation = correction["operation"]

            if operation == "DELETE":
                deleted_ids.add(item_id)
            elif operation == "MERGE":
                target = correction["target"]  # Expected format: "lemma|POS"
                merge_sources[item_id] = target

        # Second pass: process KEEP operations
        keep_count = 0
        for item_id, item_data in id_mapping.items():
            # Skip deleted and merged items
            if item_id in deleted_ids or item_id in merge_sources:
                logger.warning(
                    f"Skipping item ID {item_id} during KEEP pass (deleted or merged)"
                )
                continue

            original_key = item_data["key"]

            # KEEP: copy to corrected counter
            freq = counter.lemma_counts.get(original_key, 0)
            if freq > 0:
                corrected_counter.lemma_counts[original_key] = freq
                corrected_counter.total_tokens += freq
                keep_count += 1

                # Copy POS details
                if original_key in counter.lemma_pos_details:
                    corrected_counter.lemma_pos_details[original_key] = (
                        counter.lemma_pos_details[original_key].copy()
                    )

        # Third pass: process MERGE operations with accumulation
        merge_accumulator = {}  # target_key -> accumulated_frequency

        for item_id in merge_sources.keys():
            original_key = id_mapping[item_id]["key"]
            target_key = merge_sources[item_id]

            # Get frequency from original counter
            freq = counter.lemma_counts.get(original_key, 0)

            # Accumulate in merge map
            if target_key not in merge_accumulator:
                merge_accumulator[target_key] = 0
            merge_accumulator[target_key] += freq

        # Add merged items to corrected counter
        for target_key, accumulated_freq in merge_accumulator.items():
            # Check if target already exists (from KEEP operations)
            if target_key in corrected_counter.lemma_counts:
                corrected_counter.lemma_counts[target_key] += accumulated_freq
                corrected_counter.total_tokens += accumulated_freq
            else:
                # Create new entry
                corrected_counter.lemma_counts[target_key] = accumulated_freq
                corrected_counter.total_tokens += accumulated_freq

                # Extract POS from target_key for pos_details
                parts = target_key.split("|")
                if len(parts) == 2:
                    corrected_counter.lemma_pos_details[target_key] = {parts[1]}

        # Copy OOV counts (unchanged)
        corrected_counter.oov_counts = counter.oov_counts.copy()
        corrected_counter.oov_pos_details = {
            k: v.copy() for k, v in counter.oov_pos_details.items()
        }
        corrected_counter.total_oov_tokens = counter.total_oov_tokens

        logger.info(
            f"Applied corrections: "
            f"{len(deleted_ids)} deleted, "
            f"{len(merge_sources)} merged, "
            f"{keep_count} kept"
        )

        return corrected_counter

    def _save_original(self, id_mapping: Dict[str, Dict], output_path: Path) -> None:
        """
        Save original list with IDs to CSV.

        Args:
            id_mapping: ID to lemma mapping
            output_path: Path to output CSV
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "lemma", "frequency", "pos"])

            for item_id in sorted(id_mapping.keys()):
                item = id_mapping[item_id]
                writer.writerow(
                    [item_id, item["lemma"], item["frequency"], item["pos"]]
                )

        logger.info(f"Original list with IDs saved: {output_path}")

    def _save_corrections(
        self, corrections: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """
        Save LLM corrections to CSV.

        Args:
            corrections: List of correction operations
            output_path: Path to output CSV
        """
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "operation", "target"])

            for correction in corrections:
                target = correction.get("target", "")
                writer.writerow([correction["id"], correction["operation"], target])

        logger.info(f"LLM corrections saved: {output_path}")
