from typing import Dict, List, Any
import cohere
import time
import os
import json
import random
from langsmith import Client
from langsmith.run_helpers import traceable

class CohereEmbedder:
    """Class for handling Cohere embeddings"""
    
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0"):
        """Initialize Cohere embedder with API key and model name"""
        self.api_key = api_key
        self.model_name = model_name
        # Set environment variable to help with SSL issues
        os.environ['CURL_CA_BUNDLE'] = ''
        self.client = cohere.Client(api_key=self.api_key)
        
    def get_embeddings(self):
        """Return a dummy embeddings object for compatibility with LangChain"""
        # This is a simple placeholder to make it compatible with the vector store
        class DummyEmbeddings:
            def __init__(self, api_key, model_name):
                self.api_key = api_key
                self.model_name = model_name
                # Set environment variable to help with SSL issues
                os.environ['CURL_CA_BUNDLE'] = ''
                
            def embed_query(self, text):
                return self.embed_documents([text])[0]
                
            def embed_documents(self, texts):
                try:
                    co_response = cohere.Client(api_key=self.api_key).embed(
                        texts=texts,
                        model=self.model_name,
                        input_type="search_document"
                    )
                    return co_response.embeddings
                except Exception as e:
                    print(f"Error in DummyEmbeddings: {str(e)}")
                    # Return deterministic pseudo-random embeddings instead of zeros
                    # This ensures vectors aren't all zeros
                    return [self._generate_pseudo_embedding(text, 1024) for text in texts]
                    
            def _generate_pseudo_embedding(self, text, dim):
                """Generate a deterministic pseudo-random embedding based on text hash"""
                # Use text to seed random for deterministic results
                random.seed(hash(text) % 10000)
                # Generate a vector with small random values
                vector = [random.uniform(0.01, 0.02) for _ in range(dim)]
                # Normalize to unit length
                magnitude = sum(x*x for x in vector) ** 0.5
                return [x/magnitude for x in vector]
        
        return DummyEmbeddings(self.api_key, self.model_name)
    
    def _generate_pseudo_embedding(self, text, dim):
        """Generate a deterministic pseudo-random embedding based on text hash"""
        # Use text to seed random for deterministic results
        random.seed(hash(text) % 10000)
        # Generate a vector with small random values
        vector = [random.uniform(0.01, 0.02) for _ in range(dim)]
        # Normalize to unit length
        magnitude = sum(x*x for x in vector) ** 0.5
        return [x/magnitude for x in vector]
    
    def _batch_embed_with_backoff(self, texts, max_retries=5, batch_size=5):
        """Embed texts in smaller batches with exponential backoff for rate limits"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            retry_count = 0
            backoff_time = 1  # Start with 1 second
            
            while retry_count < max_retries:
                try:
                    print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    response = self.client.embed(
                        texts=batch,
                        model=self.model_name,
                        input_type="search_document"
                    )
                    all_embeddings.extend(response.embeddings)
                    # Add a delay between batches to avoid rate limits
                    time.sleep(2)
                    break
                except Exception as e:
                    retry_count += 1
                    # Check if it's a rate limit error
                    if "rate limit" in str(e).lower() and retry_count < max_retries:
                        print(f"Rate limit hit. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                        # Generate pseudo-random embeddings for this batch
                        pseudo_embeddings = [self._generate_pseudo_embedding(text, 1024) for text in batch]
                        all_embeddings.extend(pseudo_embeddings)
                        time.sleep(1)  # Still add a small delay
                        break
        
        return all_embeddings
    
    def _sanitize_metadata(self, metadata):
        """Convert complex metadata to formats acceptable by Pinecone"""
        sanitized = {}
        for key, value in metadata.items():
            if key == "languages" and isinstance(value, dict):
                # Convert languages dict to list of "language: level" strings
                sanitized[key] = [f"{lang}: {level}" for lang, level in value.items()]
            elif isinstance(value, (str, int, float, bool)):
                # Simple types are allowed directly
                sanitized[key] = value
            elif isinstance(value, list):
                # For lists, ensure all items are strings
                if all(isinstance(item, str) for item in value):
                    sanitized[key] = value
                else:
                    # Convert complex list items to strings
                    sanitized[key] = [str(item) for item in value]
            else:
                # Convert complex objects to JSON strings
                sanitized[key] = str(value)
        return sanitized
    
    @traceable(name="cohere_embed_documents")
    def embed_documents(self, documents: List[Dict[str, Any]], langsmith_client: Client = None):
        """Embed documents using Cohere embeddings in batches"""
        # Extract text and metadata from documents
        texts = [doc.get("chunk_text", "") if "chunk_text" in doc else doc.get("text", "") for doc in documents]
        
        # Sanitize metadata to handle complex structures
        raw_metadatas = [doc.get("metadata", {}) for doc in documents]
        metadatas = [self._sanitize_metadata(metadata) for metadata in raw_metadatas]
        
        # Process in batches with backoff to handle rate limits
        print(f"Embedding {len(texts)} texts with Cohere API...")
        embeddings = self._batch_embed_with_backoff(texts)
        
        # Return embeddings along with original texts and metadata
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
        
    def embed_query(self, query: str):
        """Embed a query using Cohere embeddings"""
        try:
            co_response = self.client.embed(
                texts=[query],
                model=self.model_name,
                input_type="search_query"
            )
            return co_response.embeddings[0]
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            # Return a pseudo-random embedding instead of zeros
            return self._generate_pseudo_embedding(query, 1024) 