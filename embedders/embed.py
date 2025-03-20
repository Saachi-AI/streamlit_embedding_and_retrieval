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

def embed_documents(model_name: str, data_path: str, env_vars: Dict[str, str], batch_size: int = 20, save_locally: bool = False):
    """Embed documents using the specified model and store in Pinecone in batches"""
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
        init_pinecone(env_vars, model_name)
        namespace = f"{model_name.lower().replace('-', '_')}_embeddings"
        vector_store = get_vector_store(embedder.get_embeddings(), env_vars, namespace)
    else:
        vector_store = None
    
    # Store all successfully embedded documents
    all_embedded_data = {
        "texts": [],
        "embeddings": [],
        "metadatas": []
    }
    
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        try:
            # Embed current batch
            embedded_data = embedder.embed_documents(batch, langsmith_client)
            
            # If saving locally, store the data
            if save_locally:
                all_embedded_data["texts"].extend(embedded_data["texts"])
                all_embedded_data["embeddings"].extend(embedded_data["embeddings"])
                all_embedded_data["metadatas"].extend(embedded_data["metadatas"])
                print(f"Added batch {batch_num} to local storage")
            else:
                # Add documents to vector store
                vector_store.add_texts(
                    texts=embedded_data["texts"],
                    metadatas=embedded_data["metadatas"]
                )
                print(f"Successfully embedded batch {batch_num}/{total_batches} to Pinecone")
            
            # Add a delay between batches to avoid overwhelming APIs
            if i + batch_size < len(documents):
                print(f"Waiting before processing next batch...")
                time.sleep(3)
                
        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            print("Retrying with a smaller batch size...")
            
            # Try with a smaller batch size for this problematic batch
            smaller_batch_size = max(1, batch_size // 2)
            for j in range(0, len(batch), smaller_batch_size):
                sub_batch = batch[j:j+smaller_batch_size]
                try:
                    embedded_sub_data = embedder.embed_documents(sub_batch, langsmith_client)
                    
                    if save_locally:
                        all_embedded_data["texts"].extend(embedded_sub_data["texts"])
                        all_embedded_data["embeddings"].extend(embedded_sub_data["embeddings"])
                        all_embedded_data["metadatas"].extend(embedded_sub_data["metadatas"])
                        print(f"Added sub-batch {j//smaller_batch_size + 1} to local storage")
                    else:
                        vector_store.add_texts(
                            texts=embedded_sub_data["texts"],
                            metadatas=embedded_sub_data["metadatas"]
                        )
                        print(f"Successfully embedded sub-batch {j//smaller_batch_size + 1}")
                    
                    time.sleep(2)
                except Exception as sub_e:
                    print(f"Error processing sub-batch: {str(sub_e)}")
                    print(f"Skipping {len(sub_batch)} documents in problematic sub-batch")
    
    # Save embeddings locally if requested
    if save_locally and all_embedded_data["texts"]:
        with open(local_output_file, 'w') as f:
            json.dump(all_embedded_data, f)
        print(f"Saved {len(all_embedded_data['texts'])} embeddings to {local_output_file}")
    
    if not save_locally:
        print(f"Finished embedding process with {model_name} in Pinecone namespace: {namespace}")
    else:
        print(f"Finished embedding process with {model_name} and saved locally")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Embed documents using different models')
    parser.add_argument('--model', choices=['openai', 'cohere'], required=True, help='Embedding model to use')
    parser.add_argument('--data', default='profile_chunks.json', help='Path to the data file')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of documents to process in each batch')
    parser.add_argument('--save-locally', action='store_true', help='Save embeddings locally instead of uploading to Pinecone')
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_environment()
    
    # Embed documents
    embed_documents(args.model, args.data, env_vars, args.batch_size, args.save_locally)

if __name__ == "__main__":
    main() 