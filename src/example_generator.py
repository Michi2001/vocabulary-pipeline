"""LLM-based example sentence generator with caching support."""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.llm_providers.base_provider import BaseLLMProvider
from src.llm_providers.common.rate_limiter import AsyncRateLimiter
from src.lemma_cache import LemmaCache
from src.vocab import DEFAULT_DISAMBIGUATED_POS, Translation

logger = logging.getLogger(__name__)

BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")


class ExampleGenerator:
    """
    Generates diverse example sentences for translation meanings.

    For polysemantic words in disambiguated POS: generates num_examples_poly examples per meaning.
    For all other words: generates 1 example per meaning.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache: Optional[LemmaCache],
        config: Dict[str, Any],
        disambiguated_pos: Optional[set] = None,
    ):
        self.provider = provider
        self.cache = cache
        self.model_config = provider.get_cache_key_components()
        self.batch_size = config.get("batch_size", 50)
        self.prompts_dir = Path(config.get("prompts_dir", "prompts/examples"))
        self.num_examples_poly = config.get("num_examples_poly", 10)
        self.disambiguated_pos = disambiguated_pos or DEFAULT_DISAMBIGUATED_POS
        self.max_requests_per_minute = config.get("max_requests_per_minute", 0)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 5)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Load prompt and compute hash
        self.system_prompt = self._load_prompt()
        self.prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]

    def generate(
        self, translations: List[Translation]
    ) -> Tuple[List[Translation], Dict[str, int]]:
        """Generate examples for all translations."""
        logger.info("Starting example sentence generation...")

        # Collect all meaning items that need examples
        items_to_process = []
        cached_count = 0

        for translation in translations:
            is_poly = len(translation.meanings) > 1 and translation.pos in self.disambiguated_pos
            num_examples = self.num_examples_poly if is_poly else 1

            for meaning_idx, meaning in enumerate(translation.meanings):
                definition_hash = self._compute_definition_hash(meaning.definition_es)

                # Check cache
                cached = None
                if self.cache is not None:
                    cached = self.cache.get_examples(
                        translation.lemma,
                        translation.pos,
                        definition_hash,
                        self.model_config,
                        self.prompt_hash,
                    )

                if cached is not None:
                    self._apply_examples_to_meaning(meaning, cached["examples_es"])
                    cached_count += 1
                else:
                    items_to_process.append(
                        {
                            "translation": translation,
                            "meaning_idx": meaning_idx,
                            "meaning": meaning,
                            "definition_hash": definition_hash,
                            "num_examples": num_examples,
                        }
                    )

        logger.info(
            f"Cache: {cached_count} hits, {len(items_to_process)} meanings need generation"
        )

        # Token usage tracking
        total_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_thinking_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "cache_hits": cached_count,
        }

        if not items_to_process:
            logger.info("All examples loaded from cache")
            return translations, total_usage

        # Process in batches
        batches = self._create_batches(items_to_process)
        logger.info(
            f"Created {len(batches)} batches for {len(items_to_process)} meanings"
        )

        self._run_batches(batches, total_usage)

        # Log usage summary
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE GENERATION - TOKEN USAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"API Calls:          {total_usage['api_calls']}")
        logger.info(f"Cache Hits:         {total_usage['cache_hits']}")
        logger.info(f"Prompt Tokens:      {total_usage['total_prompt_tokens']:,}")
        logger.info(f"Completion Tokens:  {total_usage['total_completion_tokens']:,}")
        logger.info(f"Thinking Tokens:    {total_usage['total_thinking_tokens']:,}")
        logger.info(f"Total Tokens:       {total_usage['total_tokens']:,}")
        logger.info("=" * 60 + "\n")

        return translations, total_usage

    def _run_batches(
        self,
        batches: List[List[Dict]],
        total_usage: Dict[str, int],
    ) -> None:
        """Run batch processing — async if multiple concurrent requests, else sync."""
        if not batches:
            return

        if self.max_concurrent_requests <= 1:
            self._process_batches_sync(batches, total_usage)
            return

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()

        self._loop.run_until_complete(
            self._process_batches_async(batches, total_usage)
        )

    def _process_batches_sync(
        self,
        batches: List[List[Dict]],
        total_usage: Dict[str, int],
    ) -> None:
        """Process batches sequentially (original behavior)."""
        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} meanings)"
            )

            batch_input = self._prepare_batch_input(batch)
            prompt_with_num = self.system_prompt.replace(
                "{num_examples}", str(batch[0]["num_examples"])
            )
            response = self.provider.generate_examples(batch_input, prompt_with_num)

            self._accumulate_usage(total_usage, response)
            self._apply_batch_results(batch, response.get("all_examples", []))

            if self.cache is not None:
                self.cache.save_all()

    async def _process_batches_async(
        self,
        batches: List[List[Dict]],
        total_usage: Dict[str, int],
    ) -> None:
        """Process batches in parallel with concurrency + rate limiting."""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        rate_limiter = AsyncRateLimiter(self.max_requests_per_minute)

        async def process_one_batch(
            batch_idx: int, batch: List[Dict]
        ) -> Tuple[int, List[Dict], Optional[Dict]]:
            async with semaphore:
                await rate_limiter.acquire()
                logger.info(
                    f"Processing batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch)} meanings)"
                )
                batch_input = self._prepare_batch_input(batch)
                prompt_with_num = self.system_prompt.replace(
                    "{num_examples}", str(batch[0]["num_examples"])
                )
                response = await self.provider.generate_examples_async(
                    batch_input, prompt_with_num
                )
                return (
                    batch_idx,
                    response.get("all_examples", []),
                    response.get("_token_usage"),
                )

        tasks = [process_one_batch(i, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i + 1}/{len(batches)} failed: {result}")
                continue

            batch_idx, all_examples, usage = result

            if usage:
                total_usage["total_prompt_tokens"] += usage.get("prompt_tokens") or 0
                total_usage["total_completion_tokens"] += usage.get("completion_tokens") or 0
                total_usage["total_thinking_tokens"] += usage.get("thinking_tokens") or 0
                total_usage["total_tokens"] += usage.get("total_tokens") or 0
                total_usage["api_calls"] += 1

            self._apply_batch_results(batches[batch_idx], all_examples)

        if self.cache is not None:
            self.cache.save_all()

    def _accumulate_usage(self, total_usage: Dict[str, int], response: Dict[str, Any]):
        """Accumulate token usage from a response into total_usage."""
        if "_token_usage" in response:
            usage = response["_token_usage"]
            total_usage["total_prompt_tokens"] += usage.get("prompt_tokens") or 0
            total_usage["total_completion_tokens"] += usage.get("completion_tokens") or 0
            total_usage["total_thinking_tokens"] += usage.get("thinking_tokens") or 0
            total_usage["total_tokens"] += usage.get("total_tokens") or 0
            total_usage["api_calls"] += 1

    def _load_prompt(self) -> str:
        """Load example generation prompt."""
        prompt_file = self.prompts_dir / "general_prompt.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")

    def _compute_definition_hash(self, definition_es: str) -> str:
        """Compute hash of definition for cache key."""
        return hashlib.sha256(definition_es.encode()).hexdigest()[:8]

    def _create_batches(self, items: List[Dict]) -> List[List[Dict]]:
        """Create batches from items, grouping by num_examples."""
        mono = [i for i in items if i["num_examples"] == 1]
        poly = [i for i in items if i["num_examples"] > 1]

        batches = []

        for i in range(0, len(mono), self.batch_size):
            batches.append(mono[i : i + self.batch_size])

        for i in range(0, len(poly), self.batch_size):
            batches.append(poly[i : i + self.batch_size])

        return batches

    def _prepare_batch_input(self, batch: List[Dict]) -> List[Dict]:
        """Prepare input format for LLM."""
        return [
            {
                "lemma": item["translation"].lemma,
                "pos": item["translation"].pos,
                "definition_es": item["meaning"].definition_es,
            }
            for item in batch
        ]

    def _apply_batch_results(self, batch: List[Dict], results: List[Dict]):
        """Apply LLM results to meanings and cache."""
        result_lookup = {}
        for result in results:
            key = (result["lemma"], result["pos"], result["definition_es"])
            result_lookup[key] = result.get("examples_es", [])

        for item in batch:
            key = (
                item["translation"].lemma,
                item["translation"].pos,
                item["meaning"].definition_es,
            )
            examples = result_lookup.get(key, [])

            if not examples:
                logger.warning(
                    f"No examples returned for {item['translation'].lemma} "
                    f"(meaning: {item['meaning'].definition_es[:30]}...)"
                )
                continue

            self._apply_examples_to_meaning(item["meaning"], examples)

            if self.cache is not None:
                self.cache.set_examples(
                    item["translation"].lemma,
                    item["translation"].pos,
                    item["definition_hash"],
                    self.model_config,
                    self.prompt_hash,
                    {"examples_es": examples},
                )

    def _apply_examples_to_meaning(self, meaning, examples: List[str]):
        """Apply examples to meaning, extracting target word positions."""
        cleaned_examples = []
        target_starts = []
        target_ends = []

        for example in examples:
            match = BRACKET_PATTERN.search(example)
            if not match:
                logger.warning(f"No bracketed word in example: '{example[:50]}...'")
                cleaned_examples.append(example)
                target_starts.append(None)
                target_ends.append(None)
            else:
                word = match.group(1)
                char_start = match.start()
                char_end = char_start + len(word)

                cleaned_sentence = example.replace(f"[{word}]", word, 1)
                cleaned_examples.append(cleaned_sentence)
                target_starts.append(char_start)
                target_ends.append(char_end)

        meaning.examples_es = cleaned_examples
        meaning._target_starts = target_starts
        meaning._target_ends = target_ends
