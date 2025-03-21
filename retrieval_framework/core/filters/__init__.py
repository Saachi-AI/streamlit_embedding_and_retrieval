"""
Filters package for the retrieval framework.

Contains components for extracting structured filters from natural language queries.
"""

from retrieval_framework.core.filters.base import FilterExtractorBase
from retrieval_framework.core.filters.extractor import FilterExtractor

__all__ = [
    'FilterExtractorBase',
    'FilterExtractor'
] 