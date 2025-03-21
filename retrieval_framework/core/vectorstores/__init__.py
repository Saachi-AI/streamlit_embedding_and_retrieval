"""
Vector stores package for the retrieval framework.

Contains implementations for different vector databases used for similarity search.
"""

from retrieval_framework.core.vectorstores.base import VectorStoreBase
from retrieval_framework.core.vectorstores.pinecone import PineconeStore

__all__ = [
    'VectorStoreBase',
    'PineconeStore'
] 