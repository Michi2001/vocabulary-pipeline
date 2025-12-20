"""OpenAI API provider for lemma correction."""

import json
import logging
import time
from typing import List, Dict, Any

from openai import OpenAI

from src.llm_providers.base_provider import BaseLLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for lemma correction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: {
                'api_key': str,
                'model': str (default: 'gpt-5-mini'),
                'max_retries': int (default: 3),
                'retry_delay': int (default: 2),
                'timeout': int (default: 60)
            }
        """
        super().__init__(config)
        self.client = OpenAI(api_key=config["api_key"])
        self.model = config.get("model", "gpt-5-mini")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.timeout = config.get("timeout", 180)

    def correct_lemmas(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send batch to OpenAI API for correction.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: System prompt

        Returns:
            Dict with parsed corrections

        Raises:
            LLMProviderError: If all retries fail
        """
        # Format batch for LLM
        user_message = self._format_batch(batch)

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )

                # Parse JSON response
                content = response.choices[0].message.content
                parsed = json.loads(content)

                # Validate
                if not self.validate_correction_response(parsed):
                    raise LLMProviderError(f"Invalid response structure: {parsed}")

                # Add token usage info to response
                if hasattr(response, "usage"):
                    parsed["_token_usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                logger.info(f"Successfully corrected batch of {len(batch)} lemmas")
                return parsed

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise LLMProviderError(
                        f"Failed after {self.max_retries} attempts: {e}"
                    )

    def _format_batch(self, batch: List[Dict[str, Any]]) -> str:
        """
        Format batch as string for LLM input.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys

        Returns:
            Formatted string (one line per entry: "ID | Lemma | Frequency | POS")
        """
        lines = []
        for item in batch:
            lines.append(
                f"{item['id']} | {item['lemma']} | {item['frequency']} | {item['pos']}"
            )
        return "\n".join(lines)

    def translate_lemmas(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send batch to OpenAI API for translation.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: Combined general + POS-specific translation prompt

        Returns:
            Dict with parsed translations

        Raises:
            LLMProviderError: If all retries fail
        """
        user_message = self._format_batch(batch)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content
                parsed = json.loads(content)

                if not self._validate_translation_response(parsed):
                    raise LLMProviderError(f"Invalid translation response structure")

                if hasattr(response, "usage"):
                    parsed["_token_usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                logger.info(f"Successfully translated batch of {len(batch)} lemmas")
                return parsed

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise LLMProviderError(
                        f"Failed after {self.max_retries} attempts: {e}"
                    )

    def _validate_translation_response(self, response: Dict[str, Any]) -> bool:
        """Validate translation response structure."""
        if "all_lemma_translations" not in response:
            logger.error("Response missing 'all_lemma_translations' key")
            return False

        if not isinstance(response["all_lemma_translations"], list):
            logger.error("'all_lemma_translations' is not a list")
            return False

        for translation in response["all_lemma_translations"]:
            if (
                "id" not in translation
                or "lemma" not in translation
                or "meanings" not in translation
            ):
                logger.error(f"Translation missing required fields: {translation}")
                return False

        return True

    def validate_correction_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate OpenAI response.

        Expected format:
        {
            "corrections": [
                {"id": "001", "operation": "KEEP"},
                {"id": "002", "operation": "DELETE", "reason": "..."},
                {"id": "003", "operation": "MERGE", "target": "target_lemma|POS", "reason": "..."}
            ]
        }

        Args:
            response: Parsed JSON response from LLM

        Returns:
            True if valid, False otherwise
        """
        if "corrections" not in response:
            logger.error("Response missing 'corrections' key")
            return False

        if not isinstance(response["corrections"], list):
            logger.error("'corrections' is not a list")
            return False

        for correction in response["corrections"]:
            # Check required fields
            if "id" not in correction or "operation" not in correction:
                logger.error(f"Correction missing 'id' or 'operation': {correction}")
                return False

            operation = correction["operation"]
            if operation not in ["KEEP", "DELETE", "MERGE"]:
                logger.error(f"Invalid operation: {operation}")
                return False

            # MERGE must have target
            if operation == "MERGE" and "target" not in correction:
                logger.error(f"MERGE operation missing 'target': {correction}")
                return False

        return True
