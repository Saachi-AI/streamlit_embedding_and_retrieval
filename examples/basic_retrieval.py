#!/usr/bin/env python3
"""
Basic example demonstrating how to use the retrieval framework for document search.

This example shows:
1. Setting up the embedder, filter extractor, and vector store
2. Creating the retrieval service
3. Performing retrieval with and without metadata filtering
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to allow imports in example
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_framework.core.embedders import OpenAIEmbedder
from retrieval_framework.core.filters import FilterExtractor
from retrieval_framework.core.vectorstores import PineconeStore
from retrieval_framework.services import RetrievalService

# Load environment variables from .env file
load_dotenv()

def main():
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not all([openai_api_key, pinecone_api_key, pinecone_index_name, groq_api_key]):
        print("Error: Missing required environment variables.")
        print("Please set OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, and GROQ_API_KEY.")
        return
    
    # Initialize components
    print("Initializing components...")
    embedder = OpenAIEmbedder(api_key=openai_api_key)
    
    filter_extractor = FilterExtractor(api_key=groq_api_key)
    
    vector_store = PineconeStore(
        api_key=pinecone_api_key,
        index_name=pinecone_index_name
    )
    
    # Create retrieval service
    retrieval_service = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        filter_extractor=filter_extractor
    )
    
    # Example query without metadata filtering
    query = "Find candidates with experience in machine learning"
    print(f"\nPerforming retrieval for query: '{query}' (without metadata filtering)")
    
    results = retrieval_service.retrieve_documents(
        query=query,
        top_k=3,
        enable_metadata_filtering=False
    )
    
    print(f"Found {len(results)} matching documents:")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.metadata.get('title', 'Untitled')} - Score: {doc.score:.4f}")
        print(f"  Experience: {doc.metadata.get('years_of_experience', 'Unknown')} years")
        print(f"  Gender: {doc.metadata.get('gender', 'Unknown')}")
        print(f"  Languages: {doc.metadata.get('languages', 'None specified')}")
        print()
    
    # Example query with metadata filtering
    query = "Find male candidates with 5+ years of experience who speak fluent Japanese"
    print(f"\nPerforming retrieval for query: '{query}' (with metadata filtering)")
    
    results = retrieval_service.retrieve_documents(
        query=query,
        top_k=3,
        enable_metadata_filtering=True
    )
    
    print(f"Found {len(results)} matching documents:")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.metadata.get('title', 'Untitled')} - Score: {doc.score:.4f}")
        print(f"  Experience: {doc.metadata.get('years_of_experience', 'Unknown')} years")
        print(f"  Gender: {doc.metadata.get('gender', 'Unknown')}")
        print(f"  Languages: {doc.metadata.get('languages', 'None specified')}")
        print()
    
    # Show the extracted filters
    query = "Find female candidates with 3+ years of experience who speak native English and business-level Korean"
    print(f"\nExample filter extraction for query: '{query}'")
    
    filters = filter_extractor.extract_filters(query)
    print("Extracted filters:")
    print(filters)

if __name__ == "__main__":
    main() 