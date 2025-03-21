"""
Cohere embedder implementation for the retrieval framework.
"""

import os
import time
from typing import Dict, List, Any, Optional

import cohere
from langsmith import Client
from langsmith.run_helpers import traceable

from retrieval_framework.core.embedders.base import EmbedderInterface
from retrieval_framework.config.constants import DEFAULT_COHERE_MODEL, EMBEDDING_DIMENSIONS
from retrieval_framework.utils import get_logger, EmbeddingAPIError, retry_with_backoff

logger = get_logger(__name__)


class CohereEmbedder(EmbedderInterface):
    """
    Cohere embeddings implementation.
    
    Uses Cohere's embedding models to generate vector representations of text.
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_COHERE_MODEL):
        """
        Initialize the Cohere embedder.
        
        Args:
            api_key: Cohere API key
            model_name: Name of the Cohere embedding model to use (default: from constants)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSIONS.get(model_name, EMBEDDING_DIMENSIONS["cohere"])
        
        logger.info(f"Initializing Cohere embedder with model: {model_name}, dimension: {self.dimension}")
        
        # Set environment variable to help with SSL issues (if needed)
        os.environ['CURL_CA_BUNDLE'] = ''
        
        self.client = cohere.Client(api_key=self.api_key)
    
    def get_embeddings(self):
        """
        Get a LangChain-compatible embeddings object.
        
        Returns:
            A dummy embeddings object for compatibility with LangChain vector stores
        """
        # Create a wrapper class to make Cohere embeddings compatible with LangChain
        return self._DummyEmbeddings(
            api_key=self.api_key,
            model_name=self.model_name
        )
    
    class _DummyEmbeddings:
        """Inner class to provide LangChain compatibility."""
        
        def __init__(self, api_key: str, model_name: str):
            """Initialize with API key and model name."""
            self.api_key = api_key
            self.model_name = model_name
            # Set environment variable to help with SSL issues
            os.environ['CURL_CA_BUNDLE'] = ''
            self.client = cohere.Client(api_key=self.api_key)
            
        def embed_query(self, text: str) -> List[float]:
            """Embed a single query string."""
            return self.embed_documents([text])[0]
            
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed a list of documents."""
            try:
                response = self.client.embed(
                    texts=texts,
                    model=self.model_name,
                    input_type="search_document"
                )
                return response.embeddings
            except Exception as e:
                logger.error(f"Error in DummyEmbeddings: {str(e)}")
                raise EmbeddingAPIError(f"Failed to embed with Cohere: {str(e)}") from e
    
    @traceable(name="cohere_embed_documents")
    @retry_with_backoff(logger=logger)
    def embed_documents(self, documents: List[Dict[str, Any]], langsmith_client: Optional[Client] = None) -> Dict[str, Any]:
        """
        Embed documents using Cohere embeddings.
        
        Args:
            documents: List of document dictionaries containing text
            langsmith_client: Optional LangSmith client for tracking
            
        Returns:
            Dict containing texts, embeddings, and metadata
            
        Raises:
            EmbeddingAPIError: If there's an error with the Cohere API
        """
        # Extract text and metadata from documents
        texts = [doc.get("chunk_text", "") if "chunk_text" in doc else doc.get("text", "") for doc in documents]
        
        # Sanitize metadata to handle complex structures
        raw_metadatas = [doc.get("metadata", {}) for doc in documents]
        metadatas = [self._sanitize_metadata(metadata) for metadata in raw_metadatas]
        
        # Embed texts
        logger.info(f"Embedding {len(texts)} texts with Cohere API")
        try:
            response = self._time_api_call(
                "Cohere", 
                f"embed/{self.model_name}", 
                self.client.embed,
                texts=texts,
                model=self.model_name,
                input_type="search_document"
            )
            embeddings = response.embeddings
        except Exception as e:
            logger.error(f"Error embedding documents with Cohere: {str(e)}")
            raise EmbeddingAPIError(f"Failed to embed documents with Cohere: {str(e)}") from e
        
        # Return embeddings along with original texts and metadata
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
    
    @retry_with_backoff(logger=logger)
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query using Cohere embeddings.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            EmbeddingAPIError: If there's an error with the Cohere API
        """
        try:
            response = self._time_api_call(
                "Cohere", 
                f"embed_query/{self.model_name}", 
                self.client.embed,
                texts=[query],
                model=self.model_name,
                input_type="search_query"
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding query with Cohere: {str(e)}")
            raise EmbeddingAPIError(f"Failed to embed query with Cohere: {str(e)}") from e 