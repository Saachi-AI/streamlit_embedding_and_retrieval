"""
Configuration package for the retrieval framework.
Includes settings and constants.
"""

from retrieval_framework.config.settings import get_settings, Settings
from retrieval_framework.config.constants import (
    EMBEDDING_DIMENSIONS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_COHERE_MODEL,
    MAX_RETRY_ATTEMPTS,
    RETRY_BACKOFF_FACTOR,
    DEFAULT_VECTOR_METRIC,
    DEFAULT_PINECONE_REGION,
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)

__all__ = [
    'get_settings',
    'Settings',
    'EMBEDDING_DIMENSIONS',
    'DEFAULT_OPENAI_MODEL',
    'DEFAULT_COHERE_MODEL',
    'MAX_RETRY_ATTEMPTS',
    'RETRY_BACKOFF_FACTOR',
    'DEFAULT_VECTOR_METRIC',
    'DEFAULT_PINECONE_REGION',
    'DEFAULT_LOG_LEVEL',
    'LOG_FORMAT',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_MAX_TOKENS'
] 