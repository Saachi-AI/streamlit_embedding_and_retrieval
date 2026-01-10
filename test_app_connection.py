import os
import sys
sys.path.append('.')  # Add current directory to path

from utils import load_environment, get_vector_store

# Load environment variables using app's method
env_vars = load_environment()

# Get embeddings class (create a dummy with no methods required)
class DummyEmbeddings:
    def embed_query(self, text):
        return [0.0] * 1024  # Dummy embedding
    
    def embed_documents(self, texts):
        return [[0.0] * 1024 for _ in texts]  # Dummy embeddings

# Try to get vector store like app does
print(f"Connecting to Pinecone using app's approach...")
print(f"API Key: {env_vars['pinecone_api_key'][:10]}...")
print(f"Index Name: {env_vars['pinecone_index_name']}")
print(f"Host: {env_vars['pinecone_host']}")

try:
    embeddings = DummyEmbeddings()
    namespace = ""
    vector_store = get_vector_store(embeddings, env_vars, namespace)
    print("\nSuccess! Vector store connection successful.")
    
    # Try a simple operation
    result = vector_store.similarity_search("test query", k=1)
    print(f"Query test successful. Results: {len(result)} items")
    
except Exception as e:
    print(f"\nError connecting to vector store: {str(e)}")
    print("\nPossible causes:")
    print("1. API key is incorrect")
    print("2. Index name is incorrect")
    print("3. Host is incorrect")
    print("4. Vector store integration issues")
    print("5. No vectors in the index yet")

print("\nTest complete.") 