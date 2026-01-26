"""LLM-based lemma corrector with caching support."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import csv

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider
from src.lemma_cache import LemmaCache
from src.vocab import LemmaToken

logger = logging.getLogger(__name__)

# Type alias for correction mapping
CorrectionMapping = Dict[LemmaToken, LemmaToken]


class LemmaCorrector:
    """
    Orchestrates LLM-based lemma correction.

    Takes FrequencyCounter, sends lemmas to LLM in batches,
    applies corrections, and generates audit outputs.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache: Optional[LemmaCache],
        config: Dict[str, Any],
    ):
        """
        Initialize corrector.

        Args:
            provider: LLM provider instance
            cache: Shared LemmaCache instance (None to disable caching)
            config: {
                'batch_size': int (default: 50),
                'system_prompt': str,
                'output_dir': str (for audit files),
                'target_output_size': int (correct until output has >= this many lemmas)
            }
        """
        self.provider = provider
        self.cache = cache
        self.model_config = provider.get_cache_key_components()
        self.batch_size = config.get("batch_size", 50)
        self.system_prompt = config["system_prompt"]
        self.output_dir = Path(config["output_dir"])
        self.target_output_size = config.get("target_output_size")

    def correct(
        self, counter: FrequencyCounter, min_frequency: int = 1
    ) -> Tuple[FrequencyCounter, CorrectionMapping, Set[LemmaToken]]:
        """
        Apply LLM corrections to FrequencyCounter.

        Args:
            counter: Input FrequencyCounter
            min_frequency: Minimum frequency threshold (same as output filter)

        Returns:
            Tuple of:
                - Corrected FrequencyCounter
                - CorrectionMapping: {(orig_lemma, pos): (corrected_lemma, pos)} for MERGE operations
                - Set of deleted LemmaTokens
        """
        logger.info("Starting LLM-based lemma correction...")

        # Step 1: Get sorted frequencies (all lemmas above min_frequency)
        frequencies = counter.get_sorted_frequencies(min_frequency=min_frequency)
        logger.info(f"Total lemmas in counter: {len(frequencies)}")

        if self.target_output_size and self.target_output_size > len(frequencies):
            raise ValueError(
                f"target_output_size ({self.target_output_size}) exceeds "
                f"available lemmas ({len(frequencies)})"
            )

        # Step 2: Create ID mapping for ALL lemmas upfront
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]
        id_mapping = self._create_id_mapping(frequencies)

        # Step 3: Process batches iteratively until target reached
        all_corrections = []
        processed_idx = 0

        while True:
            # Calculate current output size
            current_output_size = self._calculate_output_size(all_corrections, id_mapping)

            # Check if target reached
            if self.target_output_size and current_output_size >= self.target_output_size:
                logger.info(
                    f"Target output size reached: {current_output_size} >= {self.target_output_size}"
                )
                break

            # Check if all lemmas processed
            if processed_idx >= len(frequencies):
                logger.info(f"All {len(frequencies)} lemmas processed")
                break

            # Collect items until we have batch_size uncached items (or run out or target reached)
            uncached_items = []
            cached_count = 0
            target_reached = False

            while processed_idx < len(frequencies) and len(uncached_items) < self.batch_size:
                item = frequencies[processed_idx]
                item_id = f"{processed_idx + 1:06d}"
                item_with_id = {
                    "id": item_id,
                    "lemma": item["lemma"],
                    "frequency": item["frequency"],
                    "pos": item["pos"],
                }

                # Check cache
                cached_correction = None
                if self.cache is not None:
                    cached_correction = self.cache.get_correction(
                        item["lemma"], item["pos"], self.model_config, prompt_hash
                    )

                if cached_correction is not None:
                    cached_correction["id"] = item_id
                    all_corrections.append(cached_correction)
                    cached_count += 1

                    # Check target after each cached item
                    if self.target_output_size:
                        current_size = self._calculate_output_size(all_corrections, id_mapping)
                        if current_size >= self.target_output_size:
                            target_reached = True
                            processed_idx += 1
                            break
                else:
                    uncached_items.append(item_with_id)

                processed_idx += 1

            # If target reached via cached items, skip LLM call
            if target_reached:
                logger.info(
                    f"Processed {processed_idx}/{len(frequencies)} lemmas "
                    f"({cached_count} cached, 0 via LLM), "
                    f"current output size: {self._calculate_output_size(all_corrections, id_mapping)}"
                )
                continue

            # Process uncached items via LLM
            if uncached_items:
                response = self.provider.correct_lemmas(uncached_items, self.system_prompt)
                batch_corrections = response["corrections"]

                # Cache the new corrections
                if self.cache is not None:
                    for correction in batch_corrections:
                        item_id = correction["id"]
                        matching_item = next(
                            (item for item in uncached_items if item["id"] == item_id), None
                        )
                        if matching_item:
                            self.cache.set_correction(
                                matching_item["lemma"],
                                matching_item["pos"],
                                self.model_config,
                                prompt_hash,
                                correction,
                            )
                    self.cache.save_all()

                all_corrections.extend(batch_corrections)

            logger.info(
                f"Processed {processed_idx}/{len(frequencies)} lemmas "
                f"({cached_count} cached, {len(uncached_items)} via LLM), "
                f"current output size: {self._calculate_output_size(all_corrections, id_mapping)}"
            )

        # Step 4: Trim id_mapping to only processed items
        processed_id_mapping = {
            f"{idx + 1:06d}": id_mapping[f"{idx + 1:06d}"]
            for idx in range(processed_idx)
        }

        # Step 5: Save original list with IDs
        original_path = self.output_dir / "original_with_ids.csv"
        self._save_original(processed_id_mapping, original_path)

        # Step 6: Save correction mapping
        mapping_path = self.output_dir / "llm_corrections.csv"
        self._save_corrections(all_corrections, mapping_path)

        # Step 7: Apply corrections to counter and build correction mapping
        corrected_counter, correction_mapping, deleted_tokens = self._apply_corrections(
            counter, processed_id_mapping, all_corrections
        )

        # Step 8: Save correction mapping to JSON
        mapping_json_path = self.output_dir / "correction_mapping.json"
        self._save_correction_mapping(
            correction_mapping, deleted_tokens, mapping_json_path
        )

        # Step 9: Save cache to disk (if cache is enabled)
        if self.cache is not None:
            self.cache.save_all()

        logger.info(
            f"LLM correction complete. Applied {len(all_corrections)} operations."
        )

        return corrected_counter, correction_mapping, deleted_tokens

    def _create_id_mapping(self, frequencies: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Create ID mapping for all frequency items."""
        id_mapping = {}
        for idx, item in enumerate(frequencies):
            item_id = f"{idx + 1:06d}"
            # Use counter_key if available (original key in counter), else construct from lemma|pos
            key = item.get("counter_key", f"{item['lemma']}|{item['pos']}")
            id_mapping[item_id] = {
                "lemma": item["lemma"],
                "frequency": item["frequency"],
                "pos": item["pos"],
                "key": key,
            }
        return id_mapping

    def _calculate_output_size(
        self, corrections: List[Dict[str, Any]], id_mapping: Dict[str, Dict]
    ) -> int:
        """Calculate resulting lemma count from corrections."""
        kept_keys = set()
        merge_targets = set()

        for corr in corrections:
            op = corr["operation"]
            item_id = corr["id"]

            if op == "KEEP":
                if item_id in id_mapping:
                    kept_keys.add(id_mapping[item_id]["key"])
            elif op == "MERGE":
                merge_targets.add(corr["target"])

        # Merge targets count only if not already in kept_keys
        return len(kept_keys) + len(merge_targets - kept_keys)

    def _apply_corrections(
        self,
        counter: FrequencyCounter,
        id_mapping: Dict[str, Dict],
        corrections: List[Dict[str, Any]],
    ) -> Tuple[FrequencyCounter, CorrectionMapping, Set[LemmaToken]]:
        """
        Apply corrections to create new FrequencyCounter and build correction mapping.

        Three-pass algorithm:
        1. Build deletion and merge maps
        2. Process KEEP operations
        3. Process MERGE operations with frequency accumulation

        Args:
            counter: Original FrequencyCounter
            id_mapping: ID to lemma mapping
            corrections: List of correction operations from LLM

        Returns:
            Tuple of:
                - New corrected FrequencyCounter
                - CorrectionMapping for MERGE operations
                - Set of deleted LemmaTokens
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

        # Build correction mapping and deleted tokens set
        correction_mapping: CorrectionMapping = {}
        deleted_tokens: Set[LemmaToken] = set()

        # Build deleted tokens set
        for item_id in deleted_ids:
            item_data = id_mapping[item_id]
            lemma_token: LemmaToken = (item_data["lemma"], item_data["pos"])
            deleted_tokens.add(lemma_token)

        # Build correction mapping for MERGE operations
        for item_id, target_key in merge_sources.items():
            item_data = id_mapping[item_id]
            original_token: LemmaToken = (item_data["lemma"], item_data["pos"])
            # Parse target_key "lemma|POS" -> (lemma, pos)
            parts = target_key.split("|")
            if len(parts) == 2:
                target_token: LemmaToken = (parts[0], parts[1])
                correction_mapping[original_token] = target_token

        # Second pass: process KEEP operations
        keep_count = 0
        for item_id, item_data in id_mapping.items():
            # Skip deleted and merged items
            if item_id in deleted_ids or item_id in merge_sources:
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

        logger.info(
            f"Applied corrections: "
            f"{len(deleted_ids)} deleted, "
            f"{len(merge_sources)} merged, "
            f"{keep_count} kept"
        )

        return corrected_counter, correction_mapping, deleted_tokens

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

    def _save_correction_mapping(
        self,
        correction_mapping: CorrectionMapping,
        deleted_tokens: Set[LemmaToken],
        output_path: Path,
    ) -> None:
        """
        Save correction mapping to JSON for use by SentenceStore.

        Args:
            correction_mapping: Mapping from original to corrected tokens
            deleted_tokens: Set of deleted tokens
            output_path: Path to output JSON file
        """
        # Convert to JSON-serializable format
        # LemmaToken tuples become lists: [lemma, pos]
        data = {
            "merge_mapping": [
                {"from": list(orig), "to": list(target)}
                for orig, target in correction_mapping.items()
            ],
            "deleted": [list(token) for token in deleted_tokens],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Correction mapping saved: {output_path} "
            f"({len(correction_mapping)} merges, {len(deleted_tokens)} deletions)"
        )

    @staticmethod
    def load_correction_mapping(
        mapping_path: Path,
    ) -> Tuple[CorrectionMapping, Set[LemmaToken]]:
        """
        Load correction mapping from JSON file.

        Args:
            mapping_path: Path to correction_mapping.json

        Returns:
            Tuple of (CorrectionMapping, Set of deleted tokens)
        """
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        correction_mapping: CorrectionMapping = {}
        for item in data.get("merge_mapping", []):
            orig = tuple(item["from"])
            target = tuple(item["to"])
            correction_mapping[orig] = target

        deleted_tokens: Set[LemmaToken] = set()
        for item in data.get("deleted", []):
            deleted_tokens.add(tuple(item))

        return correction_mapping, deleted_tokens
