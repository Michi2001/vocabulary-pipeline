"""LLM provider factory for creating provider instances."""

import logging
from typing import Dict, Any
from src.llm_providers.base_provider import BaseLLMProvider
from src.llm_providers.openai_provider import OpenAIProvider
from src.llm_providers.google_genai_provider import GoogleGenAIProvider

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
        "google": GoogleGenAIProvider,
        # Future providers can be added here:
        # 'anthropic': AnthropicProvider,
        # 'local': LocalLLMProvider,
    }

    provider_class = providers.get(provider_type.lower())

    if provider_class is None:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Available providers: {', '.join(providers.keys())}"
        )

    if provider_type.lower() == "openai":
        logger.warning("OpenAI provider is deprecated. Consider switching to 'google'.")

    logger.info(f"Creating {provider_type} provider")
    return provider_class(config)
