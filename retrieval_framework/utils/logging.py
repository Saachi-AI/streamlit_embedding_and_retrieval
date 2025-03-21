"""
Logging utilities for the retrieval framework.

Provides a centralized logging system with proper formatting and log levels.
"""

import logging
import sys
from typing import Optional

from retrieval_framework.config.constants import DEFAULT_LOG_LEVEL, LOG_FORMAT


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger, typically __name__ of the calling module
        log_level: Optional log level override (default: from constants)
        
    Returns:
        A configured logger instance
    """
    # Convert string log level to logging constant
    level = log_level or DEFAULT_LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), None)
    
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Add console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Set formatter
        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger


# Create a default application logger
app_logger = get_logger("retrieval_framework")


def log_api_call(logger: logging.Logger, api_name: str, endpoint: str, duration: float):
    """
    Log an API call with timing information.
    
    Args:
        logger: Logger instance
        api_name: Name of the API (e.g., 'OpenAI', 'Cohere')
        endpoint: Endpoint being called
        duration: Call duration in seconds
    """
    logger.info(f"{api_name} API call to {endpoint} completed in {duration:.2f}s")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to log messages.
    
    This adapter can be used to add query IDs, user IDs, or other context
    to log messages for easier tracing and debugging.
    """
    
    def __init__(self, logger: logging.Logger, extra: dict):
        """
        Initialize the adapter with a logger and extra context.
        
        Args:
            logger: The logger to adapt
            extra: Extra context to add to log messages
        """
        super().__init__(logger, extra)
    
    def process(self, msg, kwargs):
        """Process the log message and add context."""
        context_str = ' '.join(f"{k}={v}" for k, v in self.extra.items())
        return f"[{context_str}] {msg}", kwargs 