# Retrieval Framework

A modular Python framework for building and deploying document retrieval systems with embedding models, vector stores, and structured filtering.

## Features

- **Modular Architecture**: Easily swap out components for different embedding models, vector stores, and filter adapters
- **Multiple Embedding Options**: Support for OpenAI and Cohere embedding models
- **Structured Filtering**: Extract structured filters from natural language queries
- **Vector Store Integration**: Ready-to-use integration with Pinecone
- **Reranking Capability**: Enhance retrieval quality with Cohere's Rerank API
- **Multiple LLM Options**: Use either Grok or Gemini for profile evaluation
- **High-Level Services**: Simplified interfaces for common retrieval tasks
- **Configuration Management**: Centralized settings management with environment variable support
- **Robust Error Handling**: Custom error types and automatic retries for reliability

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env` file:
   ```
   XAI_API_KEY=your_grok_api_key
   GOOGLE_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the Application

To run the application with Grok LLM (default):
```
streamlit run app.py
```

To run the application with Gemini LLM:
```
streamlit run app.py llm=gemini
```

The application will automatically detect the LLM choice from the command line parameters and use the appropriate API key.

## Environment Variables

- `XAI_API_KEY`: Required for Grok LLM
- `GOOGLE_API_KEY`: Required for Gemini LLM
- `OPENAI_API_KEY`: Required for OpenAI embeddings
- `COHERE_API_KEY`: Required for Cohere embeddings and reranking
- `PINECONE_API_KEY`: Required for vector store access
- `PINECONE_INDEX_NAME`: Required for vector store access
- `GROQ_API_KEY`: Required for filter extraction

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

## Get All Profile Chunks
```bash
python get_all_chunks.py <profile_id>
```
