"""
Retrieval Framework - A modular framework for document retrieval systems.

The framework provides components for:
- Embedding documents using different embedding models
- Filtering and metadata extraction
- Vector similarity search
- Integrated retrieval pipelines

Usage:
    from retrieval_framework.services import RetrievalService, EmbeddingService
    from retrieval_framework.core.embedders import OpenAIEmbedder
    from retrieval_framework.core.filters import FilterExtractor
    from retrieval_framework.core.vectorstores import PineconeStore
    
    # Set up components
    embedder = OpenAIEmbedder(api_key="your-api-key")
    filter_extractor = FilterExtractor(api_key="your-groq-api-key")
    vector_store = PineconeStore(
        api_key="your-pinecone-api-key",
        index_name="your-index-name",
        embedding=embedder.get_embeddings()
    )
    
    # Create retrieval service
    retrieval_service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        filter_extractor=filter_extractor
    )
    
    # Retrieve documents
    results = retrieval_service.retrieve_documents(
        query="Find candidates who speak fluent Japanese with at least 5 years of experience",
        top_k=5,
        enable_metadata_filtering=True
    )
"""

from retrieval_framework.config import get_settings, Settings
from retrieval_framework.core.embedders import EmbedderInterface, OpenAIEmbedder, CohereEmbedder
from retrieval_framework.core.filters import FilterExtractorBase, FilterExtractor
from retrieval_framework.core.vectorstores import VectorStoreBase, PineconeStore
from retrieval_framework.services import RetrievalService, EmbeddingService

__all__ = [
    'get_settings',
    'Settings',
    'EmbedderInterface',
    'OpenAIEmbedder',
    'CohereEmbedder',
    'FilterExtractorBase',
    'FilterExtractor',
    'VectorStoreBase',
    'PineconeStore',
    'RetrievalService',
    'EmbeddingService'
]

__version__ = "0.1.0" 