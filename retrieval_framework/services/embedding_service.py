"""
Embedding service for the retrieval framework.

Manages the batch embedding of documents and loading of embeddings.
"""

import json
import os
from typing import Dict, List, Any, Optional
from langsmith import Client

from retrieval_framework.core.embedders.base import EmbedderInterface
from retrieval_framework.config.constants import DEFAULT_BATCH_SIZE
from retrieval_framework.utils import get_logger, EmbeddingAPIError

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for handling the embedding of documents.
    
    Provides utilities for:
    - Batch embedding of large document sets
    - Saving embeddings to local files
    - Loading embeddings from local files
    """
    
    def __init__(
        self,
        embedder: EmbedderInterface,
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_dir: str = "embeddings",
        langsmith_client: Optional[Client] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            embedder: The embedder implementation to use
            batch_size: Batch size for processing documents
            output_dir: Directory for saving embeddings
            langsmith_client: Optional LangSmith client for tracking
        """
        self.embedder = embedder
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.langsmith_client = langsmith_client
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized EmbeddingService with batch size {batch_size}")
    
    def batch_embed_documents(
        self,
        documents: List[Dict[str, Any]],
        save_locally: bool = False,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Embed a list of documents in batches.
        
        Args:
            documents: List of documents to embed
            save_locally: Whether to save embeddings to a local file
            filename: Optional filename for saving embeddings
            
        Returns:
            Dict containing all embedded documents
            
        Raises:
            EmbeddingAPIError: If embedding fails
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return {"texts": [], "embeddings": [], "metadatas": []}
        
        logger.info(f"Embedding {len(documents)} documents in batches of {self.batch_size}")
        
        # Initialize result containers
        all_texts = []
        all_embeddings = []
        all_metadatas = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(documents) // self.batch_size) + 1} ({len(batch)} documents)")
            
            try:
                # Embed batch
                batch_result = self.embedder.embed_documents(batch, self.langsmith_client)
                
                # Add results to containers
                all_texts.extend(batch_result["texts"])
                all_embeddings.extend(batch_result["embeddings"])
                all_metadatas.extend(batch_result["metadatas"])
                
                logger.info(f"Successfully embedded batch {i // self.batch_size + 1}")
            except Exception as e:
                logger.error(f"Error embedding batch {i // self.batch_size + 1}: {str(e)}")
                raise EmbeddingAPIError(f"Failed to embed batch: {str(e)}")
        
        # Assemble final result
        result = {
            "texts": all_texts,
            "embeddings": all_embeddings,
            "metadatas": all_metadatas
        }
        
        # Save locally if requested
        if save_locally:
            model_name = getattr(self.embedder, 'model_name', 'embeddings')
            filename = filename or f"{model_name.lower().replace('-', '_')}_embeddings.json"
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(result, f)
                logger.info(f"Saved {len(all_texts)} embeddings to {filepath}")
            except Exception as e:
                logger.error(f"Error saving embeddings to {filepath}: {str(e)}")
        
        logger.info(f"Successfully embedded all {len(documents)} documents")
        return result
    
    def load_embeddings(self, filepath: str) -> Dict[str, Any]:
        """
        Load embeddings from a local file.
        
        Args:
            filepath: Path to the embeddings file
            
        Returns:
            Dict containing the loaded embeddings
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid data
        """
        logger.info(f"Loading embeddings from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            if not all(key in data for key in ["texts", "embeddings", "metadatas"]):
                raise ValueError("Invalid embeddings file structure")
            
            logger.info(f"Successfully loaded {len(data['texts'])} embeddings from {filepath}")
            return data
        except FileNotFoundError:
            logger.error(f"Embeddings file not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in embeddings file: {str(e)}")
            raise ValueError(f"Invalid JSON in embeddings file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise 