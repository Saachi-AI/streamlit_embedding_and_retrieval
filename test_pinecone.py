from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
host = os.getenv("PINECONE_HOST")

print(f"Testing connection to Pinecone index: {index_name}")
print(f"Host: {host}")
print(f"API Key (first 10 chars): {api_key[:10]}...")

try:
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Connect to the index - using both name and host (important for SDK v5+)
    index = pc.Index(
        name=index_name,
        host=host
    )
    
    # Test with a basic stats call
    stats = index.describe_index_stats()
    print("\nSuccess! Index stats:")
    print(f"Total vector count: {stats.get('total_vector_count', 'N/A')}")
    
    # Print namespace information if available
    if 'namespaces' in stats:
        print("\nNamespaces:")
        for ns_name, ns_data in stats['namespaces'].items():
            print(f"  - {ns_name}: {ns_data.get('vector_count', 'N/A')} vectors")
    
except Exception as e:
    print(f"\nError connecting to Pinecone: {str(e)}")
    print("\nPossible issues:")
    print("1. API key might be incorrect")
    print("2. Index name might be incorrect")
    print("3. Host might be incorrect or not specified")
    print("4. Network issues or firewall restrictions")

print("\nTesting complete") 