#!/usr/bin/env python3
"""
Example demonstrating how to use the EmbeddingService to process documents in batches.

This example shows:
1. Setting up the embedder and embedding service
2. Batch processing documents
3. Storing results in a vector store
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path to allow imports in example
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_framework.core.embedders import OpenAIEmbedder
from retrieval_framework.core.vectorstores import PineconeStore
from retrieval_framework.services import EmbeddingService

# Load environment variables from .env file
load_dotenv()

# Sample documents to embed - in a real scenario, these might come from a database or file
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "content": "John Smith is a software engineer with 8 years of experience in Python and JavaScript.",
        "metadata": {
            "title": "John Smith",
            "years_of_experience": 8,
            "gender": "male",
            "languages": {
                "English": "Native or Bilingual proficiency",
                "Spanish": "Professional working proficiency",
                "Japanese": "Elementary proficiency"
            }
        }
    },
    {
        "id": "doc2",
        "content": "Sarah Johnson is a data scientist with 5 years of experience in machine learning and statistical analysis.",
        "metadata": {
            "title": "Sarah Johnson",
            "years_of_experience": 5,
            "gender": "female",
            "languages": {
                "English": "Native or Bilingual proficiency",
                "French": "Limited working proficiency"
            }
        }
    },
    {
        "id": "doc3",
        "content": "Alex Chen is a product manager with 10 years of experience in tech companies.",
        "metadata": {
            "title": "Alex Chen",
            "years_of_experience": 10,
            "gender": "not_mentioned",
            "languages": {
                "English": "Professional working proficiency",
                "Mandarin": "Native or Bilingual proficiency"
            }
        }
    }
]

def main():
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not all([openai_api_key, pinecone_api_key, pinecone_index_name]):
        print("Error: Missing required environment variables.")
        print("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME.")
        return
    
    # Initialize components
    print("Initializing components...")
    embedder = OpenAIEmbedder(api_key=openai_api_key)
    
    vector_store = PineconeStore(
        api_key=pinecone_api_key,
        index_name=pinecone_index_name
    )
    
    # Create embedding service
    embedding_service = EmbeddingService(
        embedder=embedder,
        vector_store=vector_store,
        batch_size=2  # Process 2 documents at a time
    )
    
    # Embed and store the documents
    print(f"Processing {len(SAMPLE_DOCUMENTS)} documents...")
    
    # Convert documents to the format expected by the embedding service
    formatted_docs = [
        {
            "id": doc["id"],
            "text": doc["content"],
            "metadata": doc["metadata"]
        }
        for doc in SAMPLE_DOCUMENTS
    ]
    
    # Process documents in batches and store them
    result = embedding_service.process_documents(
        documents=formatted_docs,
        namespace="example-namespace"
    )
    
    print(f"Successfully processed {result['successful']} documents")
    if result['failed']:
        print(f"Failed to process {result['failed']} documents")
    
    # Optionally perform a test query to verify storage
    query = "software engineer with Python experience"
    print(f"\nPerforming test query: '{query}'")
    
    query_embedding = embedder.embed_query(query)
    search_results = vector_store.similarity_search(
        query_vector=query_embedding,
        top_k=3,
        namespace="example-namespace"
    )
    
    print(f"Retrieved {len(search_results)} results:")
    for i, doc in enumerate(search_results):
        print(f"Result {i+1}: {doc.metadata.get('title', 'Untitled')} - Score: {doc.score:.4f}")
        print(f"  Content: {doc.content}")
        print()

if __name__ == "__main__":
    main() 