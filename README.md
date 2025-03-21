# Retrieval Framework

A modular Python framework for building and deploying document retrieval systems with embedding models, vector stores, and structured filtering.

## Features

- **Modular Architecture**: Easily swap out components for different embedding models, vector stores, and filter adapters
- **Multiple Embedding Options**: Support for OpenAI and Cohere embedding models
- **Structured Filtering**: Extract structured filters from natural language queries
- **Vector Store Integration**: Ready-to-use integration with Pinecone
- **High-Level Services**: Simplified interfaces for common retrieval tasks
- **Configuration Management**: Centralized settings management with environment variable support
- **Robust Error Handling**: Custom error types and automatic retries for reliability

## Installation

```bash
pip install retrieval-framework
```

## Quick Start

```python
from retrieval_framework.services import RetrievalService, EmbeddingService
from retrieval_framework.core.embedders import OpenAIEmbedder
from retrieval_framework.core.filters import FilterExtractor
from retrieval_framework.core.vectorstores import PineconeStore

# Initialize components
embedder = OpenAIEmbedder(api_key="your-openai-api-key")
filter_extractor = FilterExtractor(api_key="your-groq-api-key")
vector_store = PineconeStore(
    api_key="your-pinecone-api-key",
    index_name="your-index-name"
)

# Create retrieval service
retrieval_service = RetrievalService(
    embedder=embedder,
    vector_store=vector_store,
    filter_extractor=filter_extractor
)

# Search with natural language filtering
results = retrieval_service.retrieve_documents(
    query="Find candidates who speak fluent Japanese with at least 5 years of experience",
    top_k=5,
    enable_metadata_filtering=True
)

print(f"Found {len(results)} matching documents")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.metadata.get('title')} - Score: {doc.score}")
```

## Architecture

The framework is organized into the following components:

- **Core Modules**:
  - `embedders`: Classes for generating vector embeddings from text
  - `filters`: Components for extracting structured filters from queries
  - `vectorstores`: Interfaces for vector database operations
  
- **Services**:
  - `RetrievalService`: High-level API for document retrieval workflows
  - `EmbeddingService`: Manages batch embedding operations

- **Configuration**: Centralized settings management

## Extending the Framework

### Adding a New Embedder

```python
from retrieval_framework.core.embedders import EmbedderInterface
import your_embedding_library

class CustomEmbedder(EmbedderInterface):
    def __init__(self, api_key, model_name="custom-model"):
        self.client = your_embedding_library.Client(api_key)
        self.model_name = model_name
        
    def embed_query(self, text):
        response = self.client.embed(text, model=self.model_name)
        return response.embeddings
        
    def embed_documents(self, documents):
        # Implementation for batch embedding
        pass
```

### Adding a New Vector Store

```python
from retrieval_framework.core.vectorstores import VectorStoreBase
import your_vector_db

class CustomVectorStore(VectorStoreBase):
    def __init__(self, connection_string, collection_name):
        self.client = your_vector_db.connect(connection_string)
        self.collection = self.client.collection(collection_name)
        
    def similarity_search(self, query_vector, top_k=5, filters=None):
        results = self.collection.query(
            vector=query_vector,
            top_k=top_k,
            filter=filters
        )
        return self._format_results(results)
```

## License

MIT License 