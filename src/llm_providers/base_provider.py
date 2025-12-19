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

    @abstractmethod
    def validate_correction_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate LLM response structure.

        Args:
            response: Raw response from LLM

        Returns:
            True if valid, False otherwise
        """
        pass


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""

    pass
