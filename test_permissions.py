import os
from pinecone import Pinecone
import sys
sys.path.append('.')  # Add current directory to path

from utils import load_environment

# Load environment variables
env_vars = load_environment()
api_key = env_vars["pinecone_api_key"]
index_name = env_vars["pinecone_index_name"]
host = env_vars["pinecone_host"]

print(f"Testing Pinecone permissions with:")
print(f"API Key: {api_key[:10]}...")
print(f"Index Name: {index_name}")
print(f"Host: {host}")

# 1. Test basic connection
print("\n=== Testing basic connection ===")
try:
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client created successfully")
except Exception as e:
    print(f"❌ Failed to create Pinecone client: {e}")

# 2. Test index access
print("\n=== Testing index access ===")
try:
    index = pc.Index(
        name=index_name,
        host=host
    )
    print("✅ Connected to index successfully")
except Exception as e:
    print(f"❌ Failed to connect to index: {e}")

# 3. Test stats retrieval
print("\n=== Testing stats retrieval ===")
try:
    stats = index.describe_index_stats()
    print("✅ Retrieved stats successfully")
    print(f"Total vectors: {stats.get('total_vector_count', 'Unknown')}")
except Exception as e:
    print(f"❌ Failed to retrieve stats: {e}")

# 4. Test query capability
print("\n=== Testing query capability ===")
try:
    # Create a simple query vector
    vector = [0.0] * 1024  # Assuming 1024 dimensions
    results = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True
    )
    print("✅ Query executed successfully")
    print(f"Results: {len(results['matches'])} matches")
except Exception as e:
    print(f"❌ Failed to execute query: {e}")

# 5. Test upsert capability (only if explicitly requested)
print("\n=== Testing upsert capability (NOT EXECUTING) ===")
print("Upsert test skipped to avoid modifying your index")
# This would test whether you have write permissions, but we won't run it
# to avoid modifying your index

print("\nTest complete. If all tests pass, you have proper permissions.")
print("If any test fails, the error message should indicate the permission issue.") 