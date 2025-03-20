from typing import Dict, List, Any
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langsmith.run_helpers import traceable

class OpenAIEmbedder:
    """Class for handling OpenAI embeddings"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """Initialize OpenAI embedder with API key and model name"""
        self.api_key = api_key
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.model_name,
            dimensions=1536
        )
        
    def get_embeddings(self):
        """Return the embeddings object"""
        return self.embeddings
    
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
    
    @traceable(name="openai_embed_documents")
    def embed_documents(self, documents: List[Dict[str, Any]], langsmith_client: Client = None):
        """Embed documents using OpenAI embeddings"""
        # Extract text and metadata from documents
        texts = [doc.get("chunk_text", "") if "chunk_text" in doc else doc.get("text", "") for doc in documents]
        
        # Sanitize metadata to handle complex structures
        raw_metadatas = [doc.get("metadata", {}) for doc in documents]
        metadatas = [self._sanitize_metadata(metadata) for metadata in raw_metadatas]
        
        # Embed texts
        print(f"Embedding {len(texts)} texts with OpenAI API...")
        try:
            embeddings = self.embeddings.embed_documents(texts)
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
        """Embed a query using OpenAI embeddings"""
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise e 