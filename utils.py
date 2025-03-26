import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME"
    ]
    
    # Check if all required variables are set
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Return environment variables as a dictionary
    return {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),  # Default to us-east-1 if not specified
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME"),
        "pinecone_host": os.getenv("PINECONE_HOST"),  # Add host information
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "langchain_api_key": os.getenv("LANGCHAIN_API_KEY"),
        "langchain_project": os.getenv("LANGCHAIN_PROJECT"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),  # Add Groq API key
        "upstage_api_key": os.getenv("UPSTAGE_API_KEY"),  # Add Upstage API key
    }

def init_pinecone(env_vars: Dict[str, str], embedding_model_name: str):
    """Initialize Pinecone client and ensure index exists"""
    try:
        # Check if we have host information
        if env_vars.get("pinecone_host"):
            print(f"Connecting to Pinecone index via host: {env_vars['pinecone_host']}")
            # Initialize with host parameter
            pc = Pinecone(
                api_key=env_vars["pinecone_api_key"]
            )
            
            index_name = env_vars["pinecone_index_name"]
            
            # Try to connect directly to the index using the host
            try:
                index = pc.Index(
                    host=f"https://{env_vars['pinecone_host']}"
                )
                print(f"Successfully connected to existing index '{index_name}'")
                return pc
            except Exception as e:
                print(f"Error connecting to index via host: {str(e)}")
        
        # Fallback to standard initialization
        print(f"Connecting to Pinecone in region: {env_vars['pinecone_environment']}")
        pc = Pinecone(api_key=env_vars["pinecone_api_key"])
        
        index_name = env_vars["pinecone_index_name"]
        
        # Check if index exists
        try:
            existing_indexes = [index.name for index in pc.list_indexes()]
            index_exists = index_name in existing_indexes
        except Exception as e:
            print(f"Error listing Pinecone indexes: {str(e)}")
            print("Will attempt to use the index directly.")
            index_exists = True  # Assume it exists
        
        # Determine dimension based on model
        dimension = 1536 if "openai" in embedding_model_name.lower() else 1024
        
        if not index_exists:
            # Use a valid AWS region format
            region = env_vars["pinecone_environment"]
            if "." in region or "http" in region:
                print(f"Invalid region format: {region}. Using 'us-east-1' instead.")
                region = "us-east-1"
                
            try:
                print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=region
                    )
                )
                print(f"Successfully created index '{index_name}'")
            except Exception as e:
                print(f"Error creating Pinecone index: {str(e)}")
                print("Will attempt to use existing index if available.")
        
        return pc
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        print("Will attempt to continue with vector store operations.")
        return None

def get_vector_store(embedding, env_vars: Dict[str, str], namespace: str = None):
    """Get vector store for the given embedding model"""
    try:
        # Try a simpler approach that should work with multiple versions of the API
        return PineconeVectorStore.from_existing_index(
            index_name=env_vars["pinecone_index_name"],
            embedding=embedding,
            namespace=namespace
        )
    except Exception as e:
        print(f"Error connecting to vector store: {str(e)}")
        # Try an alternative approach
        try:
            pc = Pinecone(api_key=env_vars["pinecone_api_key"])
            return PineconeVectorStore(
                index_name=env_vars["pinecone_index_name"],
                embedding=embedding,
                namespace=namespace
            )
        except Exception as e2:
            print(f"Error with alternative approach: {str(e2)}")
            raise ValueError(f"Failed to connect to Pinecone vector store. Make sure your Pinecone configuration is correct.")

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file"""
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
        # If data is in array format, return as is
        if isinstance(data, list):
            return data
        # If data is in object format with a 'data' field
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            return data 