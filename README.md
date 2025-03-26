# Retrieval Framework

A modular Python framework for building and deploying document retrieval systems with embedding models, vector stores, and structured filtering.

## Features

- **Modular Architecture**: Easily swap out components for different embedding models, vector stores, and filter adapters
- **Multiple Embedding Options**: Support for OpenAI and Cohere embedding models
- **Structured Filtering**: Extract structured filters from natural language queries
- **Vector Store Integration**: Ready-to-use integration with Pinecone
- **Reranking Capability**: Enhance retrieval quality with Cohere's Rerank API
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
from retrieval_framework.core.rerankers import CohereReranker

# Initialize components
embedder = OpenAIEmbedder(api_key="your-openai-api-key")
filter_extractor = FilterExtractor(api_key="your-groq-api-key")
vector_store = PineconeStore(
    api_key="your-pinecone-api-key",
    index_name="your-index-name"
)
reranker = CohereReranker(api_key="your-cohere-api-key")

# Create retrieval service
retrieval_service = RetrievalService(
    embedder=embedder,
    vector_store=vector_store,
    filter_extractor=filter_extractor,
    reranker=reranker
)

# Search with natural language filtering and reranking
results = retrieval_service.retrieve_documents(
    query="Find candidates who speak fluent Japanese with at least 5 years of experience",
    top_k=10,
    rerank_top_k=5,
    enable_metadata_filtering=True,
    enable_reranking=True
)

print(f"Found {len(results)} matching documents")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.metadata.get('title')} - Score: {doc.score}")
```

## Environment Variables

The framework supports configuration via environment variables:

```
OPENAI_API_KEY=your-openai-api-key
COHERE_API_KEY=your-cohere-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
GROQ_API_KEY=your-groq-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
SEMANTIC_TOP_K=10  # Number of results to fetch from vector store
RERANK_TOP_K=5     # Number of results to keep after reranking
```

## Architecture

The framework is organized into the following components:

- **Core Modules**:
  - `embedders`: Classes for generating vector embeddings from text
  - `filters`: Components for extracting structured filters from queries
  - `vectorstores`: Interfaces for vector database operations
  - `rerankers`: Components for reranking retrieval results
  
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

### Adding a New Reranker

```python
from retrieval_framework.core.rerankers import RerankerInterface
import your_reranking_library

class CustomReranker(RerankerInterface):
    def __init__(self, api_key, model_name="custom-reranker"):
        self.client = your_reranking_library.Client(api_key)
        self.model_name = model_name
        
    def rerank(self, query, documents, top_k=5):
        document_texts = [doc.page_content for doc in documents]
        reranked = self.client.rerank(
            query=query,
            documents=document_texts,
            model=self.model_name,
            top_n=top_k
        )
        
        # Map reranked results back to original documents
        return [
            (documents[result.index], result.score)
            for result in reranked.results
        ]
```

## License

MIT License

# Job Description Parser and Semantic Search

## New Features

### Job Description Parser
The application now includes a document parser that:
- Accepts uploaded job description files (PDF, DOC, DOCX)
- Uses Upstage AI's document digitization API to extract text
- Processes the extracted text with GROQ's DeepSeek R1 Distill Llama 70B model
- Automatically generates optimized prompts for semantic candidate search
- Extracts key skills, experience requirements, and language proficiency needs
- Creates a structured representation of the job requirements

### How to Use the Job Description Parser
1. Go to the "Job Description Parser" tab
2. Upload a job description document (PDF, DOC, DOCX)
3. Click "Parse Document" to process it
4. View the extracted information and generated search prompt
5. Click "Use this prompt for candidate search" to use it in the search tab

### Required API Keys
You'll need to add these API keys to your .env file:
- `UPSTAGE_API_KEY`: For document parsing via Upstage AI
- `GROQ_API_KEY`: For LLM processing using DeepSeek R1 Distill Llama 70B

## Retrieval Framework Documentation 