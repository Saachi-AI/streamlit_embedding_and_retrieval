"""
Filter adapters package for the retrieval framework.

Contains adapters for converting generic filters to database-specific formats.
"""

from retrieval_framework.core.filters.adapters.pinecone import PineconeFilterAdapter

__all__ = [
    'PineconeFilterAdapter'
] 