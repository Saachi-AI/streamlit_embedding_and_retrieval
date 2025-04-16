import os
from pinecone import Pinecone

# Get API key directly from environment (not loading .env)
api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")
host = os.environ.get("PINECONE_HOST")

print(f"Testing with environment variables (not loading .env file)")
print(f"API Key: {api_key[:10]}... (if exists)")
print(f"Index Name: {index_name}")
print(f"Host: {host}")

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(
        name=index_name,
        host=host
    )
    
    # Try to get index stats
    stats = index.describe_index_stats()
    print("\nSuccess! Connection to Pinecone successful.")
    print(f"Total vector count: {stats.get('total_vector_count', 'Unknown')}")
    
    # Print namespaces if any
    if 'namespaces' in stats:
        print("\nNamespaces:")
        for ns_name, ns_data in stats['namespaces'].items():
            ns_display = f"'{ns_name}'" if ns_name else "'Default'"
            print(f"  - {ns_display}: {ns_data.get('vector_count', 'Unknown')} vectors")
    
except Exception as e:
    print(f"\nError connecting to Pinecone: {str(e)}")
    print("\nPossible causes:")
    print("1. API key is incorrect")
    print("2. Index name is incorrect")
    print("3. Host is incorrect")
    print("4. Network or firewall issues")

print("\nTest complete.") 