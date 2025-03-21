"""
Constants for the retrieval framework.
Centralizes all hardcoded values to make them easily configurable.
"""

# Embedding dimensions based on model
EMBEDDING_DIMENSIONS = {
    "openai": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 1536,
    "cohere": 1024,
    "embed-english-v3.0": 1024
}

# Default embedding models
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_COHERE_MODEL = "embed-english-v3.0"

# API related constants
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 0.5  # seconds

# Vector database constants
DEFAULT_VECTOR_METRIC = "cosine"
DEFAULT_PINECONE_REGION = "us-east-1"

# Logging constants
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default batch sizes for processing
DEFAULT_BATCH_SIZE = 100

# LLM constants
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024 