"""LLM provider factory for creating provider instances."""

import logging
from typing import Dict, Any
from src.llm_providers.base_provider import BaseLLMProvider
from src.llm_providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


def create_provider(provider_type: str, config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_type: Type of provider ('openai', 'anthropic', etc.)
        config: Provider configuration

    Returns:
        Provider instance

    Raises:
        ValueError: If provider type is unknown
    """
    providers = {
        "openai": OpenAIProvider,
        # Future providers can be added here:
        # 'anthropic': AnthropicProvider,
        # 'local': LocalLLMProvider,
    }

    provider_class = providers.get(provider_type.lower())

    logger.info(f"Creating {provider_type} provider")
    return provider_class(config)
