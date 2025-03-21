"""
Utility package for the retrieval framework.

Contains helper functions, error handling, and logging functionality.
"""

from retrieval_framework.utils.errors import (
    RetrievalFrameworkError,
    ConfigurationError,
    APIError,
    EmbeddingAPIError,
    VectorStoreError,
    FilterExtractionError,
    QueryProcessingError,
    RateLimitError,
    AuthenticationError,
    DataFormatError
)

from retrieval_framework.utils.logging import (
    get_logger,
    app_logger,
    log_api_call,
    LoggerAdapter
)

from retrieval_framework.utils.retry import retry_with_backoff

__all__ = [
    'RetrievalFrameworkError',
    'ConfigurationError',
    'APIError',
    'EmbeddingAPIError',
    'VectorStoreError',
    'FilterExtractionError',
    'QueryProcessingError',
    'RateLimitError',
    'AuthenticationError',
    'DataFormatError',
    'get_logger',
    'app_logger',
    'log_api_call',
    'LoggerAdapter',
    'retry_with_backoff'
] 