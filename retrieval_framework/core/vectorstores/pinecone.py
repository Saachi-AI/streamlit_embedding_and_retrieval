"""
Pinecone vector store implementation for the retrieval framework.
"""

from typing import List, Dict, Any, Optional, Tuple, cast
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

from retrieval_framework.core.vectorstores.base import VectorStoreBase
from retrieval_framework.config.constants import DEFAULT_VECTOR_METRIC, DEFAULT_PINECONE_REGION
from retrieval_framework.utils import get_logger, VectorStoreError, retry_with_backoff

logger = get_logger(__name__)


class PineconeStore(VectorStoreBase):
    """
    Pinecone vector store implementation.
    
    Wrapper around LangChain's PineconeVectorStore with added functionality.
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding,
        environment: Optional[str] = DEFAULT_PINECONE_REGION,
        host: Optional[str] = None,
        namespace: Optional[str] = None,
        metric: str = DEFAULT_VECTOR_METRIC
    ):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding: Embedding function to use
            environment: Pinecone environment/region (default: from constants)
            host: Optional host URL (for serverless)
            namespace: Optional namespace to partition the index
            metric: Vector similarity metric (default: from constants)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.host = host
        self.namespace = namespace
        self.metric = metric
        self.embedding = embedding
        
        logger.info(f"Initializing Pinecone store with index: {index_name}")
        
        # Initialize Pinecone client
        self.pc = self._init_pinecone()
        
        # Initialize LangChain vector store
        self.store = self._init_vector_store()
        
    @retry_with_backoff(logger=logger)
    def _init_pinecone(self) -> Pinecone:
        """
        Initialize Pinecone client and ensure index exists.
        
        Returns:
            Initialized Pinecone client
            
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            pc = Pinecone(api_key=self.api_key)
            
            # Try to connect to the index directly if host is provided
            if self.host:
                logger.info(f"Connecting to Pinecone via host: {self.host}")
                try:
                    index = pc.Index(host=f"https://{self.host}")
                    logger.info(f"Successfully connected to existing index via host")
                    return pc
                except Exception as e:
                    logger.warning(f"Error connecting via host: {str(e)}")
            
            # Standard initialization
            logger.info(f"Connecting to Pinecone in region: {self.environment}")
            
            # Check if index exists
            try:
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                index_exists = self.index_name in existing_indexes
            except Exception as e:
                logger.warning(f"Error listing Pinecone indexes: {str(e)}")
                index_exists = True  # Assume it exists
            
            # Create index if it doesn't exist
            if not index_exists:
                # Determine dimension based on model
                try:
                    # Try to get dimension from embedding
                    dimension = self.embedding.dimension
                except AttributeError:
                    # Fallback to a default value
                    dimension = 1536
                    logger.warning(f"Could not determine dimension from embedding, using default: {dimension}")
                    
                logger.info(f"Creating Pinecone index '{self.index_name}' with dimension {dimension}")
                
                # Use a valid AWS region format
                region = self.environment
                if "." in region or "http" in region:
                    logger.warning(f"Invalid region format: {region}. Using '{DEFAULT_PINECONE_REGION}' instead.")
                    region = DEFAULT_PINECONE_REGION
                    
                try:
                    pc.create_index(
                        name=self.index_name,
                        dimension=dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(
                            cloud="aws",
                            region=region
                        )
                    )
                    logger.info(f"Successfully created index '{self.index_name}'")
                except Exception as e:
                    logger.error(f"Error creating Pinecone index: {str(e)}")
                    raise VectorStoreError(f"Failed to create Pinecone index: {str(e)}")
            
            return pc
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    @retry_with_backoff(logger=logger)
    def _init_vector_store(self) -> PineconeVectorStore:
        """
        Initialize the LangChain Pinecone vector store.
        
        Returns:
            Initialized PineconeVectorStore
            
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            logger.info(f"Initializing PineconeVectorStore for index: {self.index_name}")
            return PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embedding,
                namespace=self.namespace
            )
        except Exception as e:
            logger.error(f"Error initializing PineconeVectorStore: {str(e)}")
            
            # Try alternative initialization approach
            try:
                logger.info("Trying alternative initialization approach")
                return PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embedding,
                    namespace=self.namespace
                )
            except Exception as e2:
                logger.error(f"Error with alternative initialization: {str(e2)}")
                raise VectorStoreError(f"Failed to initialize PineconeVectorStore: {str(e)}, {str(e2)}")
    
    @retry_with_backoff(logger=logger)
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts and their metadata to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries, one per text
            
        Returns:
            List of IDs for the added texts
            
        Raises:
            VectorStoreError: If adding texts fails
        """
        try:
            logger.info(f"Adding {len(texts)} texts to Pinecone index: {self.index_name}")
            start_time = time.time()
            ids = self.store.add_texts(texts=texts, metadatas=metadatas)
            duration = time.time() - start_time
            logger.info(f"Added {len(texts)} texts to Pinecone in {duration:.2f}s")
            return ids
        except Exception as e:
            logger.error(f"Error adding texts to Pinecone: {str(e)}")
            raise VectorStoreError(f"Failed to add texts to Pinecone: {str(e)}")
    
    @retry_with_backoff(logger=logger)
    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Perform a similarity search for the query string.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of documents most similar to the query
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.info(f"Performing similarity search with k={k}" + (", filter applied" if filter else ""))
            start_time = time.time()
            docs = self.store.similarity_search(query, k=k, filter=filter)
            duration = time.time() - start_time
            logger.info(f"Similarity search completed in {duration:.2f}s, returned {len(docs)} results")
            return docs
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise VectorStoreError(f"Failed to perform similarity search: {str(e)}")
    
    @retry_with_backoff(logger=logger)
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
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.info(f"Performing similarity search with scores, k={k}" + (", filter applied" if filter else ""))
            start_time = time.time()
            results = self.store.similarity_search_with_score(query, k=k, filter=filter)
            duration = time.time() - start_time
            logger.info(f"Similarity search with scores completed in {duration:.2f}s, returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise VectorStoreError(f"Failed to perform similarity search with scores: {str(e)}")
    
    @retry_with_backoff(logger=logger)
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            filter: Optional metadata filter to select documents for deletion
            
        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            logger.info(f"Deleting documents from Pinecone index: {self.index_name}")
            
            if not ids and not filter:
                # Delete all vectors in the namespace
                logger.warning(f"Deleting ALL vectors in namespace '{self.namespace}'")
                index = self.pc.Index(name=self.index_name)
                index.delete(delete_all=True, namespace=self.namespace)
                logger.info(f"Deleted all vectors in namespace '{self.namespace}'")
            elif ids:
                # Delete specific IDs
                logger.info(f"Deleting {len(ids)} specific vectors by ID")
                index = self.pc.Index(name=self.index_name)
                index.delete(ids=ids, namespace=self.namespace)
                logger.info(f"Deleted {len(ids)} vectors by ID")
            elif filter:
                # Delete by filter is not directly supported in Pinecone
                # First we need to query to get the IDs
                logger.info(f"Deleting vectors by filter (requires query first)")
                # This is a simplified implementation and might need to be enhanced
                # to handle pagination for large result sets
                index = self.pc.Index(name=self.index_name)
                query_response = index.query(
                    namespace=self.namespace,
                    vector=[0] * 1536,  # Dummy vector
                    filter=filter,
                    top_k=10000,
                    include_metadata=False
                )
                
                if query_response.matches:
                    ids_to_delete = [match.id for match in query_response.matches]
                    if ids_to_delete:
                        index.delete(ids=ids_to_delete, namespace=self.namespace)
                        logger.info(f"Deleted {len(ids_to_delete)} vectors by filter")
                    else:
                        logger.info("No vectors matched the filter criteria")
                else:
                    logger.info("No vectors matched the filter criteria")
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            raise VectorStoreError(f"Failed to delete from Pinecone: {str(e)}")
    
    @retry_with_backoff(logger=logger)
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
            
        Raises:
            VectorStoreError: If getting stats fails
        """
        try:
            logger.info(f"Getting stats for Pinecone index: {self.index_name}")
            index = self.pc.Index(name=self.index_name)
            stats = index.describe_index_stats()
            
            # Extract namespace-specific stats
            namespace_stats = {}
            if self.namespace and self.namespace in stats.get('namespaces', {}):
                namespace_stats = {
                    'namespace': self.namespace,
                    'vector_count': stats['namespaces'][self.namespace].get('vector_count', 0)
                }
            
            # Build response
            response = {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'namespaces': len(stats.get('namespaces', {}))
            }
            
            if namespace_stats:
                response['current_namespace'] = namespace_stats
                
            logger.info(f"Retrieved Pinecone stats: total vectors = {response['total_vector_count']}")
            return response
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {str(e)}")
            raise VectorStoreError(f"Failed to get Pinecone stats: {str(e)}") 