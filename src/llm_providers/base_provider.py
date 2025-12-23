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
        self, batch: List[Dict[str, Any]], system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send batch of lemmas to LLM for translation.

        Args:
            batch: List of dicts with 'id', 'lemma', 'frequency', 'pos' keys
            system_prompt: Combined general + POS-specific translation prompt

        Returns:
            Dict with 'translations' key containing list of translation objects
            Format: {"translations": [{"id": "001", "lemma": "...", "pos": "...", "meanings": [...]}, ...]}

        Raises:
            LLMProviderError: On API errors
        """
        pass

    def _format_batch(self, batch: List[Dict[str, Any]]) -> str:
        """
        Format batch as string for LLM input (shared across providers).

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


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""

    pass
