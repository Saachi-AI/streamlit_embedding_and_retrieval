"""
Retry utilities for handling transient API failures.

Provides decorators for automatic retrying of functions that may fail due to
transient issues like network errors or rate limits.
"""

import time
import functools
import logging
from typing import Type, Callable, Any, Optional, List, Union, Tuple

from retrieval_framework.config.constants import MAX_RETRY_ATTEMPTS, RETRY_BACKOFF_FACTOR
from retrieval_framework.utils.errors import APIError, RateLimitError


def retry_with_backoff(
    max_retries: int = MAX_RETRY_ATTEMPTS,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    retry_on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (APIError,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts (default: from constants)
        backoff_factor: Factor to determine wait time between retries in seconds (default: from constants)
        retry_on_exceptions: Exception types that should trigger a retry
        logger: Optional logger to log retry attempts
        
    Returns:
        Decorator function that applies retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_count = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        # Log the final failure if logger provided
                        if logger:
                            logger.error(f"Maximum retries ({max_retries}) exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    # Handle rate limit errors with longer backoff
                    if isinstance(e, RateLimitError):
                        wait_time = (backoff_factor * (2 ** retry_count)) * 2  # Double normal backoff
                    else:
                        wait_time = backoff_factor * (2 ** retry_count)
                    
                    # Log retry attempt if logger provided
                    if logger:
                        logger.warning(
                            f"Retry {retry_count}/{max_retries} for {func.__name__} after error: {str(e)}. "
                            f"Waiting {wait_time:.2f}s before next attempt."
                        )
                    
                    # Wait before retrying
                    time.sleep(wait_time)
        return wrapper
    return decorator 