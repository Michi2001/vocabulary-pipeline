"""LLM-based lemma corrector with caching support."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import csv

from src.frequency_counter import FrequencyCounter
from src.llm_providers.base_provider import BaseLLMProvider
from src.llm_providers.common.rate_limiter import AsyncRateLimiter
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
        self.provider = provider
        self.cache = cache
        self.model_config = provider.get_cache_key_components()
        self.batch_size = config.get("batch_size", 50)
        self.system_prompt = config["system_prompt"]
        self.output_dir = Path(config["output_dir"])
        self.target_output_size = config.get("target_output_size")
        self.max_requests_per_minute = config.get("max_requests_per_minute", 0)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 5)
        self.cache_only = config.get("cache_only", False)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def correct(
        self, counter: FrequencyCounter, min_frequency: int = 1
    ) -> Tuple[FrequencyCounter, CorrectionMapping, Set[LemmaToken]]:
        """Apply LLM corrections to FrequencyCounter."""
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

        # Step 3: Scan frequencies, separate cached vs uncached, process in parallel
        all_corrections = []
        processed_idx = 0

        # Max uncached items to collect before sending to LLM
        max_uncached_per_round = self.batch_size * self.max_concurrent_requests

        while processed_idx < len(frequencies):
            # Scan items: resolve cached, collect uncached up to one parallel round
            uncached_items = []
            scan_start = processed_idx
            target_reached = False

            while processed_idx < len(frequencies):
                item = frequencies[processed_idx]
                item_id = f"{processed_idx + 1:06d}"
                item_with_id = {
                    "id": item_id,
                    "lemma": item["lemma"],
                    "frequency": item["frequency"],
                    "pos": item["pos"],
                }

                cached_correction = None
                if self.cache is not None:
                    cached_correction = self.cache.get_correction(
                        item["lemma"], item["pos"], self.model_config, prompt_hash
                    )

                if cached_correction is not None:
                    cached_correction["id"] = item_id
                    all_corrections.append(cached_correction)

                    # Check target after each cached item
                    if self.target_output_size:
                        current_size = self._calculate_output_size(
                            all_corrections, id_mapping
                        )
                        if current_size >= self.target_output_size:
                            target_reached = True
                            processed_idx += 1
                            break
                else:
                    uncached_items.append(item_with_id)

                processed_idx += 1

                # Stop scanning once we have enough uncached items for one parallel round
                if len(uncached_items) >= max_uncached_per_round:
                    break

            # Process uncached items via LLM in parallel batches
            if uncached_items and not target_reached:
                if self.cache_only:
                    logger.info(
                        f"Cache-only mode: skipping {len(uncached_items)} uncached lemmas"
                    )
                else:
                    batches = [
                        uncached_items[i : i + self.batch_size]
                        for i in range(0, len(uncached_items), self.batch_size)
                    ]
                    logger.info(
                        f"Processing {len(uncached_items)} uncached lemmas "
                        f"in {len(batches)} batches"
                    )
                    batch_corrections = self._run_batches(batches, prompt_hash)
                    all_corrections.extend(batch_corrections)

            current_output_size = self._calculate_output_size(
                all_corrections, id_mapping
            )
            logger.info(
                f"Processed {processed_idx}/{len(frequencies)} lemmas, "
                f"output size: {current_output_size}"
            )

            # Check termination conditions
            if self.target_output_size and current_output_size >= self.target_output_size:
                logger.info(
                    f"Target output size reached: "
                    f"{current_output_size} >= {self.target_output_size}"
                )
                break

            if processed_idx == scan_start:
                break

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

    def _run_batches(
        self,
        batches: List[List[Dict[str, Any]]],
        prompt_hash: str,
    ) -> List[Dict[str, Any]]:
        """Run batch processing — async if multiple concurrent requests, else sync."""
        if not batches:
            return []

        if self.max_concurrent_requests <= 1:
            return self._process_batches_sync(batches, prompt_hash)

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()

        return self._loop.run_until_complete(
            self._process_batches_async(batches, prompt_hash)
        )

    def _process_batches_sync(
        self,
        batches: List[List[Dict[str, Any]]],
        prompt_hash: str,
    ) -> List[Dict[str, Any]]:
        """Process batches sequentially."""
        all_batch_corrections = []

        for i, batch in enumerate(batches):
            logger.info(
                f"  Processing batch {i + 1}/{len(batches)} ({len(batch)} lemmas)"
            )
            response = self.provider.correct_lemmas(batch, self.system_prompt)
            batch_corrections = response["corrections"]

            self._cache_corrections(batch, batch_corrections, prompt_hash)
            all_batch_corrections.extend(batch_corrections)

        return all_batch_corrections

    async def _process_batches_async(
        self,
        batches: List[List[Dict[str, Any]]],
        prompt_hash: str,
    ) -> List[Dict[str, Any]]:
        """Process batches in parallel with concurrency + rate limiting."""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        rate_limiter = AsyncRateLimiter(self.max_requests_per_minute)
        all_batch_corrections: List[Dict[str, Any]] = []
        cache_lock = asyncio.Lock()

        async def process_one_batch(
            batch_idx: int, batch: List[Dict[str, Any]]
        ) -> None:
            async with semaphore:
                await rate_limiter.acquire()
                logger.info(
                    f"  Processing batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch)} lemmas)"
                )
                response = await self.provider.correct_lemmas_async(
                    batch, self.system_prompt
                )
                batch_corrections = response["corrections"]

                # Cache and accumulate under lock (SQLite is not thread-safe)
                async with cache_lock:
                    self._cache_corrections(batch, batch_corrections, prompt_hash)
                    all_batch_corrections.extend(batch_corrections)

        tasks = [process_one_batch(i, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log failed batches
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"  Batch {i + 1}/{len(batches)} failed: {result}")

        return all_batch_corrections

    def _cache_corrections(
        self,
        batch: List[Dict[str, Any]],
        batch_corrections: List[Dict[str, Any]],
        prompt_hash: str,
    ) -> None:
        """Cache individual corrections from a batch response."""
        if self.cache is None:
            return

        for correction in batch_corrections:
            item_id = correction["id"]
            matching_item = next(
                (item for item in batch if item["id"] == item_id), None
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

    def _create_id_mapping(self, frequencies: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Create ID mapping for all frequency items."""
        id_mapping = {}
        for idx, item in enumerate(frequencies):
            item_id = f"{idx + 1:06d}"
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

        for item_id in deleted_ids:
            item_data = id_mapping[item_id]
            lemma_token: LemmaToken = (item_data["lemma"], item_data["pos"])
            deleted_tokens.add(lemma_token)

        for item_id, target_key in merge_sources.items():
            item_data = id_mapping[item_id]
            original_token: LemmaToken = (item_data["lemma"], item_data["pos"])
            parts = target_key.split("|")
            if len(parts) == 2:
                target_token: LemmaToken = (parts[0], parts[1])
                correction_mapping[original_token] = target_token

        # Second pass: process KEEP operations
        keep_count = 0
        for item_id, item_data in id_mapping.items():
            if item_id in deleted_ids or item_id in merge_sources:
                continue

            original_key = item_data["key"]

            freq = counter.lemma_counts.get(original_key, 0)
            if freq > 0:
                corrected_counter.lemma_counts[original_key] = freq
                corrected_counter.total_tokens += freq
                keep_count += 1

                if original_key in counter.lemma_pos_details:
                    corrected_counter.lemma_pos_details[original_key] = (
                        counter.lemma_pos_details[original_key].copy()
                    )

        # Third pass: process MERGE operations with accumulation
        merge_accumulator = {}  # target_key -> accumulated_frequency

        for item_id in merge_sources.keys():
            original_key = id_mapping[item_id]["key"]
            target_key = merge_sources[item_id]

            freq = counter.lemma_counts.get(original_key, 0)

            if target_key not in merge_accumulator:
                merge_accumulator[target_key] = 0
            merge_accumulator[target_key] += freq

        for target_key, accumulated_freq in merge_accumulator.items():
            if target_key in corrected_counter.lemma_counts:
                corrected_counter.lemma_counts[target_key] += accumulated_freq
                corrected_counter.total_tokens += accumulated_freq
            else:
                corrected_counter.lemma_counts[target_key] = accumulated_freq
                corrected_counter.total_tokens += accumulated_freq

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
        """Save original list with IDs to CSV."""
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
        """Save LLM corrections to CSV."""
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
        """Save correction mapping to JSON for use by SentenceStore."""
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
        """Load correction mapping from JSON file."""
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
