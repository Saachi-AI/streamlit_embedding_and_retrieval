import os
import argparse
import json
import time
from typing import Dict, List, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_environment, init_pinecone, get_vector_store, load_data

def get_embedder(model_name: str, env_vars: Dict[str, str]):
    """Factory function to get the appropriate embedder based on model name"""
    if "openai" in model_name.lower():
        try:
            from openai_embedder import OpenAIEmbedder
            return OpenAIEmbedder(api_key=env_vars["openai_api_key"])
        except ImportError as e:
            print(f"Error importing OpenAI embedder: {str(e)}")
            raise ValueError(f"Unable to import OpenAI embedder. Make sure dependencies are installed correctly.")
    elif "cohere" in model_name.lower():
        try:
            from cohere_embedder import CohereEmbedder
            return CohereEmbedder(api_key=env_vars["cohere_api_key"])
        except ImportError as e:
            print(f"Error importing Cohere embedder: {str(e)}")
            raise ValueError(f"Unable to import Cohere embedder. Make sure dependencies are installed correctly.")
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: 'openai', 'cohere'")

def embed_documents(model_name: str, data_path: str, env_vars: Dict[str, str], save_locally: bool = False, clear_existing: bool = False):
    """Embed documents using the specified model and store in Pinecone"""
    # Initialize LangSmith client if available
    try:
        from langsmith import Client
        os.environ["LANGCHAIN_API_KEY"] = env_vars["langchain_api_key"]
        if env_vars.get("langchain_project"):
            os.environ["LANGCHAIN_PROJECT"] = env_vars["langchain_project"]
        langsmith_client = Client()
    except ImportError:
        print("LangSmith not available. Continuing without tracking.")
        langsmith_client = None
    
    # Get the appropriate embedder
    embedder = get_embedder(model_name, env_vars)
    
    # Load data
    documents = load_data(data_path)
    print(f"Loaded {len(documents)} documents")
    
    # Set up output directories if saving locally
    local_output_file = None
    if save_locally:
        os.makedirs("embeddings", exist_ok=True)
        local_output_file = f"embeddings/{model_name.lower()}_embeddings.json"
        print(f"Will save embeddings locally to {local_output_file}")
    
    # Initialize Pinecone if not saving locally
    if not save_locally:
        pc = init_pinecone(env_vars, model_name)
        namespace = f"{model_name.lower().replace('-', '_')}_embeddings"
        
        # Clear existing vectors if requested
        if clear_existing and pc:
            print(f"Clearing all vectors in namespace '{namespace}'...")
            try:
                index_name = env_vars["pinecone_index_name"]
                index = pc.Index(name=index_name)
                # Get the current count before deletion
                stats = index.describe_index_stats()
                if namespace in stats.namespaces:
                    count = stats.namespaces[namespace].vector_count
                    print(f"Found {count} existing vectors in namespace '{namespace}'")
                
                # Delete all vectors in the namespace
                index.delete(delete_all=True, namespace=namespace)
                print(f"Successfully cleared namespace '{namespace}'")
            except Exception as e:
                print(f"Error clearing namespace: {str(e)}")
        
        vector_store = get_vector_store(embedder.get_embeddings(), env_vars, namespace)
    else:
        vector_store = None
    
    # Process all documents at once (no batching)
    print(f"Processing all {len(documents)} documents at once")
    
    try:
        # Embed all documents
        print(f"Embedding {len(documents)} texts with Cohere API...")
        embedded_data = embedder.embed_documents(documents, langsmith_client)
        
        # If saving locally, store the data
        if save_locally:
            with open(local_output_file, 'w') as f:
                json.dump(embedded_data, f)
            print(f"Saved {len(embedded_data['texts'])} embeddings to {local_output_file}")
        else:
            # Add documents to vector store
            vector_store.add_texts(
                texts=embedded_data["texts"],
                metadatas=embedded_data["metadatas"]
            )
            print(f"Successfully embedded all {len(documents)} documents to Pinecone")
            
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        raise e
    
    if not save_locally:
        print(f"Finished embedding process with {model_name} in Pinecone namespace: {namespace}")
    else:
        print(f"Finished embedding process with {model_name} and saved locally")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Embed documents using different models')
    parser.add_argument('--model', choices=['openai', 'cohere'], required=True, help='Embedding model to use')
    parser.add_argument('--data', default='profile_chunks.json', help='Path to the data file')
    parser.add_argument('--save-locally', action='store_true', help='Save embeddings locally instead of uploading to Pinecone')
    parser.add_argument('--clear-existing', action='store_true', help='Clear existing vectors in the namespace before embedding')
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_environment()
    
    # Embed documents
    embed_documents(args.model, args.data, env_vars, save_locally=args.save_locally, clear_existing=args.clear_existing)

if __name__ == "__main__":
    main() 