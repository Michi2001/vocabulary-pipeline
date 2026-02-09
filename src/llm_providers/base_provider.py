"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Enables easy integration of different LLM APIs (OpenAI, Anthropic, local models, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM provider with configuration.

        Args:
            config: Provider-specific configuration (api_key, model, etc.)
        """
        self.config = config

    @abstractmethod
    def correct_lemmas(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send batch of lemmas to LLM for correction.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: System prompt for the LLM

        Returns:
            Dict with 'corrections' key containing list of correction operations
            Format: {"corrections": [{"id": "001", "operation": "KEEP/DELETE/MERGE", "target": "...", "reason": "..."}, ...]}

        Raises:
            LLMProviderError: On API errors
        """
        pass

    @abstractmethod
    def translate_lemmas(
        self,
        batch: List[Dict[str, Any]],
        system_prompt: str,
        response_schema: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Send batch of lemmas to LLM for translation.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: Combined general + POS-specific translation prompt
            response_schema: Optional POS-specific JSON schema for structured output

        Returns:
            Dict with 'translations' key containing list of translation objects
            Format: {"translations": [{"id": "001", "lemma": "...", "pos": "...", "meanings": [...]}, ...]}

        Raises:
            LLMProviderError: On API errors
        """
        pass

    async def translate_lemmas_async(
        self,
        batch: List[Dict[str, Any]],
        system_prompt: str,
        response_schema: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Async version of translate_lemmas. Must be overridden by subclasses that support async."""
        raise NotImplementedError("translate_lemmas_async not implemented by this provider")

    @abstractmethod
    def generate_examples(
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate example sentences for meanings.

        Args:
            batch: List of dicts with 'lemma', 'pos', 'definition_es' keys
            system_prompt: Example generation prompt

        Returns:
            Dict with 'all_examples' key containing list of example objects
            Format: {"all_examples": [{"lemma": "...", "pos": "...", "definition_es": "...", "examples_es": [...]}, ...]}

        Raises:
            LLMProviderError: On API errors
        """
        pass

    @abstractmethod
    def get_cache_key_components(self) -> str:
        """
        Get provider-specific cache key components.

        Returns a string that represents the configuration-dependent aspects
        of this provider that should affect caching. Different configurations
        should produce different cache keys.

        Returns:
            String with provider-specific cache key components.
            Examples:
            - OpenAI: "gpt-4o-mini"
            - Google: "gemini-2.0-flash|1.0|minimal"
        """
        pass

    def _format_batch(self, batch: List[Dict[str, Any]]) -> str:
        """Format batch for correction/translation (ID | Lemma | Frequency | POS)."""
        lines = []
        for item in batch:
            lines.append(
                f"{item['id']} | {item['lemma']} | {item['frequency']} | {item['pos']}"
            )
        return "\n".join(lines)

    def _format_examples_batch(self, batch: List[Dict[str, Any]]) -> str:
        """Format batch for example generation (Lemma | POS | Definition)."""
        lines = []
        for item in batch:
            lines.append(
                f"Lemma: {item['lemma']} | POS: {item['pos']} | Definition: {item['definition_es']}"
            )
        return "\n".join(lines)


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""

    pass
