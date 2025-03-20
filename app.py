import os
import streamlit as st
from langsmith import Client
from langsmith.run_helpers import traceable

from utils import load_environment, get_vector_store
from embedders.openai_embedder import OpenAIEmbedder
from embedders.cohere_embedder import CohereEmbedder

# Set page config
st.set_page_config(
    page_title="RAG Retrieval Demo",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
env_vars = load_environment()

# Initialize LangSmith client
os.environ["LANGCHAIN_API_KEY"] = env_vars["langchain_api_key"]
if env_vars.get("langchain_project"):
    os.environ["LANGCHAIN_PROJECT"] = env_vars["langchain_project"]
langsmith_client = Client()

# Initialize embedders
@st.cache_resource
def get_embedders():
    openai_embedder = OpenAIEmbedder(api_key=env_vars["openai_api_key"])
    cohere_embedder = CohereEmbedder(api_key=env_vars["cohere_api_key"])
    return {
        "openai": openai_embedder,
        "cohere": cohere_embedder
    }

embedders = get_embedders()

# Set up the Streamlit app
st.title("RAG Retrieval Demo")
st.subheader("Query your documents with different embedding models")

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Embedding Model",
    options=["openai", "cohere"],
    index=0
)

num_results = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=20,
    value=5
)

# Main query input
query = st.text_input("Enter your query:", key="query_input")

# Display traceable information about the retrieval process
@traceable(name="retrieve_documents")
def retrieve_documents(query, model_name, top_k):
    # Get the embedder
    embedder = embedders[model_name]
    
    # Get the vector store
    namespace = f"{model_name.lower().replace('-', '_')}_embeddings"
    vector_store = get_vector_store(embedder.get_embeddings(), env_vars, namespace)
    
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    # Sort results by relevance (higher percentage first)
    results.sort(key=lambda x: x[1])
    
    return results

# Perform retrieval when query is provided
if query:
    with st.spinner(f"Retrieving with {model_choice} embeddings..."):
        try:
            results = retrieve_documents(query, model_choice, num_results)
            
            # Display results
            st.subheader(f"Retrieved {len(results)} chunks")
            
            for i, (doc, score) in enumerate(results):
                # Calculate relevance score as percentage with 2 decimal points
                relevance_percentage = (1 - score) * 100
                
                # Extract profile_id and section from metadata
                profile_id = doc.metadata.get("profile_id", "N/A")
                # Remove decimal point if it exists in profile_id
                if isinstance(profile_id, (int, float)):
                    profile_id = str(int(profile_id))
                section = doc.metadata.get("section", "N/A")
                
                # Create header with score, profile_id, and section in the specified order
                header_html = f"""
                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                    <div style="background-color: #2196F3; color: white; padding: 5px 12px; border-radius: 15px; font-weight: bold; min-width: 75px; text-align: center;">
                        {relevance_percentage:.2f}%
                    </div>
                    <div style="background-color: #ECEFF1; padding: 5px 12px; border-radius: 15px; font-weight: 500;">
                        <span style="color: #546E7A;">Profile ID:</span> <span style="color: #263238;">{profile_id}</span>
                    </div>
                    <div style="background-color: #ECEFF1; padding: 5px 12px; border-radius: 15px; font-weight: 500;">
                        <span style="color: #546E7A;">Section:</span> <span style="color: #263238;">{section}</span>
                    </div>
                </div>
                """
                
                # Create expander with custom header
                with st.expander(f"Result {i+1}", expanded=(i == 0)):
                    # Display custom header
                    st.markdown(header_html, unsafe_allow_html=True)
                    
                    # Metadata section with improved styling (now displayed first)
                    st.markdown("<h3 style='margin-top: 15px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Metadata</h3>", unsafe_allow_html=True)
                    # Create a cleaner metadata display
                    st.json(doc.metadata)
                    
                    # Content section with improved styling (moved after metadata)
                    st.markdown("<h3 style='margin-top: 20px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Content</h3>", unsafe_allow_html=True)
                    st.markdown(f"""<div style="background-color: #FAFAFA; color: #37474F; padding: 15px; 
                                border-radius: 5px; border-left: 4px solid #2196F3; line-height: 1.6; 
                                font-family: 'Segoe UI', system-ui, sans-serif;">{doc.page_content}</div>""", 
                                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.info("Make sure you've embedded documents with this model first. Run `python embedders/embed.py --model {model_choice}`")

# Information about the app
with st.sidebar.expander("About this app"):
    st.markdown("""
    This app demonstrates a modular RAG (Retrieval-Augmented Generation) pipeline:
    
    - **Embedding Models**: Switch between OpenAI and Cohere
    - **Retrieval**: Find relevant document chunks based on your query
    - **Monitoring**: All operations are tracked with LangSmith
    
    Make sure you've embedded documents first by running:
    ```
    python embedders/embed.py --model [openai|cohere]
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangChain, Pinecone, and LangSmith") 