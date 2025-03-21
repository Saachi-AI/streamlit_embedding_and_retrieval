"""
Base filter extraction interface for the retrieval framework.

Defines the standard interface for extracting structured filters from natural language queries.
"""

import abc
from typing import Dict, Any, Optional


class FilterExtractorBase(abc.ABC):
    """
    Abstract base class for filter extractors.
    
    All filter extractor implementations must extend this class and implement its methods.
    """
    
    @abc.abstractmethod
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract structured filters from a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary of extracted filters
        """
        pass
    
    @abc.abstractmethod
    def process_query(self, query: str, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Process a query and return both extracted filters and database-specific filter format.
        
        Args:
            query: Natural language query string
            strict_mode: Whether to use strict matching criteria
            
        Returns:
            Dictionary containing extracted filters and database-specific filter
        """
        pass 