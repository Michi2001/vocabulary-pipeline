"""Google GenAI API provider for lemma correction and translation."""

import logging
from typing import List, Dict, Any

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from src.llm_providers.base_provider import BaseLLMProvider, LLMProviderError
from src.llm_providers.common import (
    retry_with_backoff,
    ResponseParser,
    extract_token_usage_google,
)

logger = logging.getLogger(__name__)


class GoogleGenAIProvider(BaseLLMProvider):
    """Google GenAI API provider for lemma correction and translation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google GenAI provider.

        Args:
            config: {
                'model': str (default: 'gemini-3-flash-preview'),
                'max_retries': int (default: 3),
                'retry_delay': int (default: 2),
                'timeout': int (default: 360),
                'temperature': float (default: 1.0),
                'thinking_level': str (default: 'minimal', options: minimal, low, medium, high)
            }
        """
        super().__init__(config)

        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        self.model = config.get("model", "gemini-3-flash-preview")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.timeout = config.get("timeout", 360)
        self.temperature = config.get("temperature", 1.0)
        self.thinking_level = config.get("thinking_level", "minimal")

    def get_cache_key_components(self) -> str:
        """
        Get cache key components for Google GenAI provider.

        For Google GenAI, model, temperature, and thinking_level all affect
        the output, so we include all three in the cache key.

        Returns:
            Cache key components as "model|temperature|thinking_level"
        """
        return f"{self.model}|{self.temperature}|{self.thinking_level}"

    def correct_lemmas(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send batch to Google GenAI API for correction.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: System prompt

        Returns:
            Dict with parsed corrections

        Raises:
            LLMProviderError: If all retries fail
        """
        user_message = self._format_batch(batch)

        # Create config with system instruction for this call
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            temperature=self.temperature,
            seed=42,
        )

        def api_call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=config,
            )

            # Parse and validate
            content = response.text
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_correction_response(parsed):
                raise LLMProviderError(f"Invalid response structure")

            # Add token usage
            token_usage = extract_token_usage_google(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully corrected batch of {len(batch)} lemmas")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="Google GenAI correction",
            )
        except Exception as e:
            raise LLMProviderError(f"Google GenAI correction failed: {e}")

    def translate_lemmas(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """Send batch to Google GenAI API for translation."""
        user_message = self._format_batch(batch)

        translation_schema = {
            "type": "OBJECT",
            "properties": {
                "all_lemma_translations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "lemma": {"type": "STRING"},
                            "pos": {"type": "STRING"},
                            "meanings": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "definition_es": {"type": "STRING"},
                                        "structure": {"type": "STRING"},
                                        "gender": {"type": "STRING"},
                                        "flags": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"},
                                        },
                                        "translations": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "text": {"type": "STRING"},
                                                    "hint": {"type": "STRING"},
                                                    "gender_target": {"type": "STRING"},
                                                },
                                                "required": ["text"],
                                            },
                                        },
                                    },
                                    "required": ["definition_es", "flags", "translations"],
                                },
                            },
                        },
                        "required": ["id", "lemma", "pos", "meanings"],
                    },
                }
            },
            "required": ["all_lemma_translations"],
        }

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=translation_schema,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            temperature=self.temperature,
        )

        def api_call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=config,
            )

            content = response.text
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_translation_response(parsed):
                raise LLMProviderError(f"Invalid translation response structure")

            token_usage = extract_token_usage_google(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully translated batch of {len(batch)} lemmas")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="Google GenAI translation",
            )
        except Exception as e:
            raise LLMProviderError(f"Google GenAI translation failed: {e}")

    def generate_examples(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate example sentences via Google GenAI API.

        Args:
            batch: List of dicts with 'lemma', 'pos', 'definition_es' keys
            system_prompt: Example generation prompt

        Returns:
            Dict with parsed examples

        Raises:
            LLMProviderError: If all retries fail
        """
        user_message = self._format_examples_batch(batch)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            temperature=self.temperature,
        )

        def api_call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=config,
            )

            content = response.text
            parsed = ResponseParser.parse_json_response(content)

            if not ResponseParser.validate_examples_response(parsed):
                raise LLMProviderError("Invalid examples response structure")

            token_usage = extract_token_usage_google(response)
            if token_usage:
                parsed["_token_usage"] = token_usage.to_dict()

            logger.info(f"Successfully generated examples for {len(batch)} meanings")
            return parsed

        try:
            return retry_with_backoff(
                api_call,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                operation_name="Google GenAI example generation",
            )
        except Exception as e:
            raise LLMProviderError(f"Google GenAI example generation failed: {e}")

