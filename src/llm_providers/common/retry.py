"""Retry logic with exponential backoff for LLM API calls."""

import asyncio
import time
import logging
from typing import Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: int = 2,
    operation_name: str = "operation"
) -> T:
    """
    Retry function with exponential backoff.

    Args:
        func: Function to execute (should raise exception on failure)
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay in seconds (will be doubled each retry)
        operation_name: Name for logging purposes

    Returns:
        Result from successful function execution

    Raises:
        Exception: The last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            logger.warning(f"{operation_name} attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts")
                raise


async def async_retry_with_backoff(
    func: Callable[[], Coroutine],
    max_retries: int = 3,
    retry_delay: int = 2,
    operation_name: str = "operation"
) -> T:
    """Async version of retry_with_backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            logger.warning(f"{operation_name} attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts")
                raise
