"""
Custom error types for the retrieval framework.

Provides specific error types for different failure scenarios in the system.
"""

class RetrievalFrameworkError(Exception):
    """Base exception class for all retrieval framework errors."""
    pass

class ConfigurationError(RetrievalFrameworkError):
    """Raised when there's an issue with configuration or environment variables."""
    pass

class APIError(RetrievalFrameworkError):
    """Base class for API-related errors."""
    pass

class EmbeddingAPIError(APIError):
    """Raised when there's an error with embedding API calls."""
    pass

class VectorStoreError(RetrievalFrameworkError):
    """Raised when there's an error with vector store operations."""
    pass

class FilterExtractionError(RetrievalFrameworkError):
    """Raised when there's an error extracting filters from queries."""
    pass

class QueryProcessingError(RetrievalFrameworkError):
    """Raised when there's an error processing a query."""
    pass

class RateLimitError(APIError):
    """Raised when an API rate limit is hit."""
    pass

class AuthenticationError(APIError):
    """Raised when authentication with an API fails."""
    pass

class DataFormatError(RetrievalFrameworkError):
    """Raised when data is in an unexpected format."""
    pass 