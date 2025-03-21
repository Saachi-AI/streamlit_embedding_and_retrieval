"""
Base vector store interface for the retrieval framework.

Defines the standard interface that all vector store implementations must follow.
"""

import abc
from typing import List, Dict, Any, Optional, Tuple, Union


class VectorStoreBase(abc.ABC):
    """
    Abstract base class for vector stores.
    
    All vector store implementations must extend this class and implement its methods.
    """
    
    @abc.abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts and their metadata to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries, one per text
            
        Returns:
            List of IDs for the added texts
        """
        pass
    
    @abc.abstractmethod
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Perform a similarity search for the query string.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of documents most similar to the query
        """
        pass
    
    @abc.abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Perform a similarity search with relevance scores.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        pass
    
    @abc.abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            filter: Optional metadata filter to select documents for deletion
        """
        pass
    
    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics (e.g., vector count)
        """
        pass 