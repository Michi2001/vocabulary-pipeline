"""Provider-agnostic token usage tracking."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TokenUsage:
    """Provider-agnostic token usage data structure."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """
        Initialize token usage.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens used
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format for response embedding."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


def extract_token_usage_openai(response: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from OpenAI response.

    Args:
        response: OpenAI API response object

    Returns:
        TokenUsage object or None if not available
    """
    if hasattr(response, "usage"):
        return TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
    return None


def extract_token_usage_google(response: Any) -> Optional[TokenUsage]:
    """
    Extract token usage from Google GenAI response.

    Args:
        response: Google GenAI API response object

    Returns:
        TokenUsage object or None if not available
    """
    # Google's response structure has usage_metadata
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", 0),
            completion_tokens=getattr(usage, "candidates_token_count", 0),
            total_tokens=getattr(usage, "total_token_count", 0)
        )
    return None
