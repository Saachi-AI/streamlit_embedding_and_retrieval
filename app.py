import os
# Don't hardcode the environment variable - we'll use session state directly
# os.environ["SEMANTIC_TOP_K"] = "20"  # Force this to use a higher value

import streamlit as st
from langsmith import Client
from langsmith.run_helpers import traceable
import time
import warnings

from utils import load_environment, get_vector_store
from embedders.openai_embedder import OpenAIEmbedder
from embedders.cohere_embedder import CohereEmbedder
from filter_extractor import FilterExtractor
from rerankers.cohere_reranker import CohereReranker
from post_rerank_aggregator import ProfileAggregator
from profile_retriever import ProfileRetriever
from profile_preprocessor import preprocess_profiles
from pinecone import Pinecone
from document_parser import DocumentParser
from prompt_generator import PromptGenerator

# Import feature modules
from features.job_description import render_job_description_tab
from features.custom_query import render_custom_query_tab

# Import UI utilities
from core.ui_components import create_tab_specific_sidebar

# Configure the page
st.set_page_config(
    page_title="RAG Retrieval Demo",
    page_icon="🔍",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Ignore specific warning from langchain
warnings.filterwarnings("ignore", message="You are trying to use a chat model")

# Set up the Streamlit app
st.title("RAG Retrieval Demo")

# Create session state variables for tracking which tab is active
if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

# Load environment variables
env_vars = load_environment()

# Initialize LangSmith client if API key is provided
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")
if langsmith_api_key:
    langsmith_client = Client()
else:
    langsmith_client = None

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

# Initialize filter extractor
@st.cache_resource
def get_filter_extractor():
    return FilterExtractor(api_key=env_vars["groq_api_key"])

filter_extractor = get_filter_extractor()

# Initialize Cohere reranker
@st.cache_resource
def get_cohere_reranker():
    return CohereReranker(api_key=env_vars["cohere_api_key"])

cohere_reranker = get_cohere_reranker()

# Initialize Profile Aggregator
@st.cache_resource
def get_profile_aggregator():
    return ProfileAggregator()

profile_aggregator = get_profile_aggregator()

# Initialize Profile Retriever
@st.cache_resource
def get_profile_retriever():
    return ProfileRetriever()

profile_retriever = get_profile_retriever()

# Initialize Document Parser
@st.cache_resource
def get_document_parser():
    return DocumentParser(api_key=env_vars["upstage_api_key"])

try:
    document_parser = get_document_parser()
except Exception as e:
    st.error(f"Error initializing document parser: {str(e)}")
    st.warning("Make sure UPSTAGE_API_KEY is set in your .env file")
    document_parser = None

# Initialize Prompt Generator
@st.cache_resource
def get_prompt_generator():
    return PromptGenerator(api_key=env_vars["groq_api_key"])

try:
    prompt_generator = get_prompt_generator()
except Exception as e:
    st.error(f"Error initializing prompt generator: {str(e)}")
    st.warning("Make sure GROQ_API_KEY is set in your .env file")
    prompt_generator = None

# Display traceable information about the retrieval process
@traceable(name="retrieve_documents")
def retrieve_documents(query, model_name, top_k, metadata_filter=None):
    """
    Retrieve documents from vector store.
    
    If a metadata_filter is provided, it is applied FIRST at the database level,
    then semantic search is performed only on the filtered subset.
    
    Args:
        query: Search query text
        model_name: Which embedding model to use
        top_k: Number of results to return
        metadata_filter: Optional filter to apply at the database level before semantic search
        
    Returns:
        results: List of (document, score) tuples
        total_chunks: Total number of vectors in the namespace
    """
    # Use the top_k parameter passed from the UI slider
    # Ensure top_k is an integer
    k = int(top_k)
    
    # Get the embedder
    embedder = embedders[model_name]
    
    # Get the vector store
    namespace = f"{model_name.lower().replace('-', '_')}_embeddings"
    vector_store = get_vector_store(embedder.get_embeddings(), env_vars, namespace)
    
    # Get total count of vectors
    total_chunks = 0
    try:
        # Initialize Pinecone directly
        pc = Pinecone(api_key=env_vars["pinecone_api_key"])
        index = pc.Index(env_vars["pinecone_index_name"])
        
        # Get stats
        stats = index.describe_index_stats()
        if namespace in stats.get('namespaces', {}):
            total_chunks = stats['namespaces'][namespace]['vector_count']
        else:
            total_chunks = stats.get('total_vector_count', 0)
    except Exception as e:
        st.warning(f"Unable to get total vector count: {e}")
        total_chunks = "Unknown"
    
    # If no query is provided, just return the counts (for stats purposes)
    if not query:
        return [], total_chunks
    
    # Perform similarity search with metadata filter if provided
    # The metadata filter is applied FIRST at the database level
    # Then semantic search is performed only on the filtered subset
    if metadata_filter:
        results = vector_store.similarity_search_with_score(
            query, 
            k=k,  # Use the passed top_k parameter
            filter=metadata_filter
        )
    else:
        results = vector_store.similarity_search_with_score(query, k=k)  # Use the passed top_k parameter
    
    # Explicitly limit results to k just to be sure
    results = results[:k]
    
    # Sort results by relevance (higher percentage first)
    results.sort(key=lambda x: x[1])
    
    return results, total_chunks

# Set up a simple mechanism to track the active tab
# Use a radio button with the same options as the tabs
active_tab = st.radio(
    "Select Tab",
    ["Upload Job Description", "Custom Search"],
    horizontal=True,
    label_visibility="collapsed",  # Hide the label
    key="tab_selector"
)

# Set active_tab_index based on the selected radio button
if active_tab == "Upload Job Description":
    st.session_state.active_tab_index = 0
else:  # Custom Search
    st.session_state.active_tab_index = 1

# Now create the sidebar with the correct tab index
config = create_tab_specific_sidebar(st.session_state.active_tab_index)

# Set a fixed rerank model (always use English)
rerank_model = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
os.environ["RERANK_MODEL"] = rerank_model

# Create tabs that match the radio button selection
tab_names = ["Upload Job Description", "Custom Search"]
tabs = st.tabs(tab_names)

# With Tab 0 (Upload Job Description)
with tabs[0]:
    # Render the job description tab if this tab is active
    if active_tab == "Upload Job Description":
        render_job_description_tab(
            document_parser=document_parser,
            prompt_generator=prompt_generator,
            filter_extractor=filter_extractor,
            embedders=embedders,
            retrieve_documents=retrieve_documents,
            cohere_reranker=cohere_reranker,
            profile_aggregator=profile_aggregator,
            profile_retriever=profile_retriever
        )

# With Tab 1 (Custom Query)
with tabs[1]:
    # Render the custom query tab if this tab is active
    if active_tab == "Custom Search":
        render_custom_query_tab(
            filter_extractor=filter_extractor,
            embedders=embedders,
            retrieve_documents=retrieve_documents,
            cohere_reranker=cohere_reranker,
            profile_aggregator=profile_aggregator,
            profile_retriever=profile_retriever
        )