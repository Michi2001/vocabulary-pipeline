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
                'timeout': int (default: 360)
            }
        """
        super().__init__(config)

        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        self.model = config.get("model", "gemini-3-flash-preview")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.timeout = config.get("timeout", 360)

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
            temperature=0.0,
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
        """
        Send batch to Google GenAI API for translation.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: Combined general + POS-specific translation prompt

        Returns:
            Dict with parsed translations

        Raises:
            LLMProviderError: If all retries fail
        """
        user_message = self._format_batch(batch)

        # Create config with system instruction for this call
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.0,
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
