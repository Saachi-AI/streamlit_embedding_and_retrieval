"""
OpenAI embedder implementation for the retrieval framework.
"""

from typing import Dict, List, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langsmith.run_helpers import traceable

from retrieval_framework.core.embedders.base import EmbedderInterface
from retrieval_framework.config.constants import DEFAULT_OPENAI_MODEL, EMBEDDING_DIMENSIONS
from retrieval_framework.utils import get_logger, EmbeddingAPIError, retry_with_backoff

logger = get_logger(__name__)


class OpenAIEmbedder(EmbedderInterface):
    """
    OpenAI embeddings implementation.
    
    Uses OpenAI's text embedding models to generate vector representations of text.
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_OPENAI_MODEL):
        """
        Initialize the OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI embedding model to use (default: from constants)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSIONS.get(model_name, EMBEDDING_DIMENSIONS["openai"])
        
        logger.info(f"Initializing OpenAI embedder with model: {model_name}, dimension: {self.dimension}")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.model_name,
            dimensions=self.dimension
        )
    
    def get_embeddings(self):
        """
        Get the OpenAI embeddings object.
        
        Returns:
            The OpenAIEmbeddings instance for use with LangChain vector stores
        """
        return self.embeddings
    
    @traceable(name="openai_embed_documents")
    @retry_with_backoff(logger=logger)
    def embed_documents(self, documents: List[Dict[str, Any]], langsmith_client: Optional[Client] = None) -> Dict[str, Any]:
        """
        Embed documents using OpenAI's embeddings.
        
        Args:
            documents: List of document dictionaries containing text
            langsmith_client: Optional LangSmith client for tracking
            
        Returns:
            Dict containing texts, embeddings, and metadata
            
        Raises:
            EmbeddingAPIError: If there's an error with the OpenAI API
        """
        # Extract text and metadata from documents
        texts = [doc.get("chunk_text", "") if "chunk_text" in doc else doc.get("text", "") for doc in documents]
        
        # Sanitize metadata to handle complex structures
        raw_metadatas = [doc.get("metadata", {}) for doc in documents]
        metadatas = [self._sanitize_metadata(metadata) for metadata in raw_metadatas]
        
        # Embed texts
        logger.info(f"Embedding {len(texts)} texts with OpenAI API")
        try:
            embeddings = self._time_api_call(
                "OpenAI", 
                f"embed/{self.model_name}", 
                self.embeddings.embed_documents, 
                texts
            )
        except Exception as e:
            logger.error(f"Error embedding documents with OpenAI: {str(e)}")
            raise EmbeddingAPIError(f"Failed to embed documents with OpenAI: {str(e)}") from e
        
        # Return embeddings along with original texts and metadata
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
    
    @retry_with_backoff(logger=logger)
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query using OpenAI embeddings.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            EmbeddingAPIError: If there's an error with the OpenAI API
        """
        try:
            return self._time_api_call(
                "OpenAI", 
                f"embed_query/{self.model_name}", 
                self.embeddings.embed_query, 
                query
            )
        except Exception as e:
            logger.error(f"Error embedding query with OpenAI: {str(e)}")
            raise EmbeddingAPIError(f"Failed to embed query with OpenAI: {str(e)}") from e 