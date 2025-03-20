from typing import Dict, List, Any
import cohere
import time
import os
import json
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
                    raise e
        
        return DummyEmbeddings(self.api_key, self.model_name)
    
    def _sanitize_metadata(self, metadata):
        """Convert complex metadata to formats acceptable by Pinecone"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
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
        """Embed documents using Cohere embeddings"""
        # Extract text and metadata from documents
        texts = [doc.get("chunk_text", "") if "chunk_text" in doc else doc.get("text", "") for doc in documents]
        
        # Sanitize metadata to handle complex structures
        raw_metadatas = [doc.get("metadata", {}) for doc in documents]
        metadatas = [self._sanitize_metadata(metadata) for metadata in raw_metadatas]
        
        # Process all texts at once
        print(f"Embedding {len(texts)} texts with Cohere API...")
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"
            )
            embeddings = response.embeddings
        except Exception as e:
            print(f"Error in embedding: {str(e)}")
            raise e
        
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
            raise e 