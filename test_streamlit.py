import streamlit as st
import os
from pinecone import Pinecone
import sys
sys.path.append('.')  # Add current directory to path

from utils import load_environment, get_vector_store

st.title("Pinecone Connection Test")

# Load environment variables using app's method
env_vars = load_environment()

# Display environment variables
st.subheader("Environment Variables")
st.write(f"API Key: {env_vars['pinecone_api_key'][:10]}...")
st.write(f"Index Name: {env_vars['pinecone_index_name']}")
st.write(f"Host: {env_vars['pinecone_host']}")

# Create simple embeddings class
class DummyEmbeddings:
    def embed_query(self, text):
        return [0.0] * 1024
    
    def embed_documents(self, texts):
        return [[0.0] * 1024 for _ in texts]

# Test direct connection
st.subheader("Direct Pinecone Connection")
try:
    pc = Pinecone(api_key=env_vars["pinecone_api_key"])
    index = pc.Index(
        name=env_vars["pinecone_index_name"],
        host=env_vars["pinecone_host"]
    )
    
    # Get stats
    stats = index.describe_index_stats()
    st.success("✅ Direct connection successful!")
    st.write(f"Total vectors: {stats.get('total_vector_count', 'Unknown')}")
    
    # Show namespaces
    if 'namespaces' in stats:
        st.write("Namespaces:")
        for ns_name, ns_data in stats['namespaces'].items():
            ns_display = f"'{ns_name}'" if ns_name else "'Default'"
            st.write(f"  - {ns_display}: {ns_data.get('vector_count', 'Unknown')} vectors")
            
except Exception as e:
    st.error(f"❌ Direct connection failed: {str(e)}")

# Test vector store connection
st.subheader("Vector Store Connection")
try:
    embeddings = DummyEmbeddings()
    namespace = ""
    vector_store = get_vector_store(embeddings, env_vars, namespace)
    st.success("✅ Vector store connection successful!")
    
    # Try a simple query
    if st.button("Test Query"):
        result = vector_store.similarity_search("test query", k=1)
        st.write(f"Query returned {len(result)} results")
        
except Exception as e:
    st.error(f"❌ Vector store connection failed: {str(e)}")

# Show debugging information
st.subheader("Debug Information")
st.code(f"""
Python Path: {sys.executable}
Streamlit Version: {st.__version__}
Pinecone in sys.modules: {'pinecone' in sys.modules}
""") 