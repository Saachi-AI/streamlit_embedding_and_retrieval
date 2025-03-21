"""
Retrieval service for the retrieval framework.

Coordinates the query processing, filtering, and retrieval from vector stores.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from langsmith import Client
from langsmith.run_helpers import traceable

from retrieval_framework.core.embedders.base import EmbedderInterface
from retrieval_framework.core.filters.base import FilterExtractorBase
from retrieval_framework.core.vectorstores.base import VectorStoreBase
from retrieval_framework.utils import get_logger, QueryProcessingError

logger = get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving documents using vector similarity and metadata filtering.
    
    Coordinates the different components of the retrieval pipeline:
    1. Query understanding
    2. Filter extraction
    3. Embedding generation
    4. Retrieval
    5. Result post-processing
    """
    
    def __init__(
        self,
        embedder: EmbedderInterface,
        vector_store: VectorStoreBase,
        filter_extractor: Optional[FilterExtractorBase] = None,
        langsmith_client: Optional[Client] = None
    ):
        """
        Initialize the retrieval service.
        
        Args:
            embedder: Embedder implementation to use
            vector_store: Vector store implementation to use
            filter_extractor: Optional filter extractor for metadata filtering
            langsmith_client: Optional LangSmith client for tracking
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.filter_extractor = filter_extractor
        self.langsmith_client = langsmith_client
        
        logger.info("Initialized RetrievalService")
    
    @traceable(name="retrieve_documents")
    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        include_scores: bool = True,
        enable_metadata_filtering: bool = True,
        strict_filtering: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve documents based on the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            include_scores: Whether to include relevance scores
            enable_metadata_filtering: Whether to use metadata filtering
            strict_filtering: Whether to use strict matching for filters
            
        Returns:
            Dict containing results and metadata about the retrieval
            
        Raises:
            QueryProcessingError: If retrieval fails
        """
        try:
            logger.info(f"Processing query: {query[:50]}..." + (f" (with filtering)" if enable_metadata_filtering else ""))
            
            # Initialize response object
            response = {
                "query": query,
                "top_k": top_k,
                "metadata_filtering": enable_metadata_filtering,
                "strict_filtering": strict_filtering,
                "results": [],
                "extracted_filters": None,
                "total_documents": 0
            }
            
            # Extract filters if metadata filtering is enabled
            metadata_filter = None
            if enable_metadata_filtering and self.filter_extractor and query:
                try:
                    filter_result = self.filter_extractor.process_query(query, strict_mode=strict_filtering)
                    response["extracted_filters"] = filter_result["extracted_filters"]
                    metadata_filter = filter_result["pinecone_filter"]
                except Exception as e:
                    logger.error(f"Error extracting filters: {str(e)}")
                    response["filter_error"] = str(e)
            
            # If no query is provided, just return stats
            if not query:
                try:
                    stats = self.vector_store.get_stats()
                    response["total_documents"] = stats.get("total_vector_count", 0)
                except Exception as e:
                    logger.error(f"Error getting stats: {str(e)}")
                
                return response
            
            # Perform retrieval
            if include_scores:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    filter=metadata_filter
                )
                
                # Format results
                response["results"] = [
                    {
                        "document": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                ]
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=top_k,
                    filter=metadata_filter
                )
                
                # Format results
                response["results"] = [
                    {
                        "document": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
            
            # Get total document count
            try:
                stats = self.vector_store.get_stats()
                response["total_documents"] = stats.get("total_vector_count", 0)
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
            
            logger.info(f"Retrieved {len(response['results'])} documents for query")
            return response
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            raise QueryProcessingError(f"Failed to retrieve documents: {str(e)}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add, each with text and metadata
            
        Returns:
            List of document IDs
            
        Raises:
            QueryProcessingError: If adding documents fails
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Extract text and metadata
            embedded_data = self.embedder.embed_documents(documents, self.langsmith_client)
            
            # Add to vector store
            ids = self.vector_store.add_texts(
                texts=embedded_data["texts"],
                metadatas=embedded_data["metadatas"]
            )
            
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise QueryProcessingError(f"Failed to add documents: {str(e)}") 