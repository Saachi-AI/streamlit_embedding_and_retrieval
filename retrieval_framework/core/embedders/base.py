"""
Base embedder interface for the retrieval framework.

Defines the standard interface that all embedders must implement.
"""

import abc
import time
from typing import Dict, List, Any, Protocol, Optional

from langsmith import Client
from langsmith.run_helpers import traceable

from retrieval_framework.utils import get_logger, log_api_call

logger = get_logger(__name__)


class EmbedderInterface(abc.ABC):
    """
    Abstract base class defining the interface for embedders.
    
    All embedder implementations must extend this class and implement its methods.
    """
    
    @abc.abstractmethod
    def get_embeddings(self):
        """
        Get the embeddings object that can be used with LangChain vector stores.
        
        Returns:
            An embeddings object compatible with LangChain vector stores
        """
        pass
    
    @abc.abstractmethod
    def embed_documents(self, documents: List[Dict[str, Any]], langsmith_client: Optional[Client] = None) -> Dict[str, Any]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of document dictionaries, each with text to embed
            langsmith_client: Optional LangSmith client for tracking
            
        Returns:
            Dictionary containing the texts, embeddings, and metadata
        """
        pass
    
    @abc.abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
        """
        pass
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert complex metadata to formats acceptable by vector stores.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # Simple types are allowed directly
                sanitized[key] = value
            elif isinstance(value, list):
                # For lists, ensure all items are strings
                if all(isinstance(item, str) for item in value):
                    sanitized[key] = value
                else:
                    # Convert complex list items to strings
                    sanitized[key] = [str(item) for item in value]
            else:
                # Convert complex objects to string representation
                sanitized[key] = str(value)
        return sanitized
    
    def _time_api_call(self, api_name: str, endpoint: str, func, *args, **kwargs):
        """
        Time an API call and log it.
        
        Args:
            api_name: Name of the API
            endpoint: Name of the endpoint/operation
            func: Function to call
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result of the function call
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            log_api_call(logger, api_name, endpoint, duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {api_name} API call to {endpoint} after {duration:.2f}s: {str(e)}")
            raise 