"""OpenAI API provider for lemma correction."""

import logging
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv
import os

from src.llm_providers.base_provider import BaseLLMProvider, LLMProviderError
from src.llm_providers.common import (
    retry_with_backoff,
    ResponseParser,
    extract_token_usage_openai,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for lemma correction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: {
                'model': str (default: 'gpt-5-mini'),
                'max_retries': int (default: 3),
                'retry_delay': int (default: 2),
                'timeout': int (default: 360)
            }
        """
        super().__init__(config)

        load_dotenv()

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = config.get("model", "gpt-5-mini")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.timeout = config.get("timeout", 360)

    def get_cache_key_components(self) -> str:
        """
        Get cache key components for OpenAI provider.

        For OpenAI, only the model affects the output, so we only include
        the model name in the cache key.

        Returns:
            Model name as cache key component
        """
        return self.model

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
        user_message = self._format_batch(batch)

        def api_call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                timeout=self.timeout,
            )

            # Parse and validate
            content = response.choices[0].message.content
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_correction_response(parsed):
                raise LLMProviderError(f"Invalid response structure")

            # Add token usage
            token_usage = extract_token_usage_openai(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully corrected batch of {len(batch)} lemmas")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="OpenAI correction",
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI correction failed: {e}")

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

        def api_call():
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
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_translation_response(parsed):
                raise LLMProviderError(f"Invalid translation response structure")

            token_usage = extract_token_usage_openai(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully translated batch of {len(batch)} lemmas")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="OpenAI translation",
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI translation failed: {e}")

    def generate_examples(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate example sentences via OpenAI API.

        Args:
            batch: List of dicts with 'lemma', 'pos', 'definition_es' keys
            system_prompt: Example generation prompt

        Returns:
            Dict with parsed examples

        Raises:
            LLMProviderError: If all retries fail
        """
        user_message = self._format_examples_batch(batch)

        def api_call():
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
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_examples_response(parsed):
                raise LLMProviderError("Invalid examples response structure")

            token_usage = extract_token_usage_openai(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully generated examples for {len(batch)} meanings")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="OpenAI example generation",
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI example generation failed: {e}")

