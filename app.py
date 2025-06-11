import os
# Don't hardcode the environment variable - we'll use session state directly
# os.environ["SEMANTIC_TOP_K"] = "20"  # Force this to use a higher value

import streamlit as st
from langsmith import Client
from langsmith.run_helpers import traceable
import time
import warnings
import sys

# Import authentication
from auth_config import setup_authentication, create_login_page, show_user_info

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
from profile_evaluator import IndividualProfileEvaluator

# Import feature modules
from features.job_description import render_job_description_tab
from features.custom_query import render_custom_query_tab

# Import UI utilities
from core.ui_components import create_tab_specific_sidebar

# Configure the page
st.set_page_config(
    page_title="Saachi AI - Candidate Search",
    page_icon="🔍",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Ignore specific warning from langchain
warnings.filterwarnings("ignore", message="You are trying to use a chat model")

# ===== AUTHENTICATION SECTION =====
# Set up authentication
authenticator = setup_authentication()

# Create login form - streamlit-authenticator v0.4.2 format
authenticator.login()

# Check authentication status from session state 
if st.session_state.get("authentication_status") == False:
    st.error('Username/password is incorrect')
    create_login_page()
    st.stop()
elif st.session_state.get("authentication_status") == None:
    st.warning('Please enter your username and password')
    create_login_page() 
    st.stop()
elif st.session_state.get("authentication_status"):
    # User is authenticated - show logout button and user info
    authenticator.logout(location='sidebar')
    show_user_info(st.session_state["name"], st.session_state["username"])
# ===== END AUTHENTICATION SECTION =====

# Set up the Streamlit app and display custom title with logo
st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQwIiBoZWlnaHQ9IjE1MSIgdmlld0JveD0iMCAwIDE0MCAxNTEiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNODguNTAzIDQuOTU3ODZDNzcuMDUzMyAtMS42NTI2MiA2Mi45NDY3IC0xLjY1MjYyIDUxLjQ5NyA0Ljk1Nzg1TDE4LjUzMTMgMjMuOTkwNkM3LjA4MTYxIDMwLjYwMTEgMC4wMjgzMjAzIDQyLjgxNzggMC4wMjgzMjAzIDU2LjAzODdWOTQuMTA0MkMwLjAyODMyMDMgMTA3LjMyNSA3LjA4MTYgMTE5LjU0MiAxOC41MzEzIDEyNi4xNTJMNTEuNDk3IDE0NS4xODVDNjIuOTQ2NyAxNTEuNzk2IDc3LjA1MzMgMTUxLjc5NiA4OC41MDI5IDE0NS4xODVMMTIxLjQ2OSAxMjYuMTUyQzEzMi45MTggMTE5LjU0MiAxMzkuOTcyIDEwNy4zMjUgMTM5Ljk3MiA5NC4xMDQyVjU2LjAzODdDMTM5Ljk3MiA0Mi44MTc4IDEzMi45MTggMzAuNjAxMSAxMjEuNDY5IDIzLjk5MDZMODguNTAzIDQuOTU3ODZaTTEwOC41ODcgNTkuNzg1N0M5Ny43MTY1IDU4LjEyMjYgODcuMjkwNCA1Ni40ODgzIDgxLjI0NzIgNTAuNDUyNkM3NS4yMDM0IDQ0LjQxMjUgNzMuNDE4OSAzMy44MzM3IDcxLjU5NzMgMjIuODE1MUM3MS41MjY4IDIyLjM4MjIgNzEuMTQ2NiAyMi4wNzgyIDcwLjcyMDIgMjIuMDg3OEM3MC4yOTM3IDIyLjA3ODIgNjkuOTEzNSAyMi4zODIzIDY5Ljg0MzEgMjIuODE1MUM2OC4wMjE1IDMzLjgzMzcgNjYuMjM3IDQ0LjQxMjUgNjAuMTkzMSA1MC40NTI2QzU0LjE0OSA1Ni40ODkyIDQzLjcyMDggNTguMTIzMSAzMi44NDg5IDU5Ljc4NjRDMzIuMzc1MiA1OS44NTk5IDMyLjA0OTIgNjAuMzA0MiAzMi4xMjI3IDYwLjc3NzlDMzIuMTI5MSA2MC44MjE5IDMyLjEzODcgNjAuODY0NiAzMi4xNTEzIDYwLjkwNTlMMzIuMTUwNyA2MC45MDU3SDMyLjE0NzFMMzIuMTM5NyA2MC45MTMxQzMyLjEyNSA2MC45NDk4IDMyLjExNzcgNjAuOTkwMiAzMi4xMTAzIDYxLjAzMDVDMzIuMDM2OSA2MS41MTE2IDMyLjM2NzQgNjEuOTU5NSAzMi44NDg0IDYyLjAzM0MzNy40NzQ5IDYyLjc0MTYgNDIuMDUgNjMuNDQyOSA0Ni4yNjQ4IDY0LjUwNDFDNTEuNTcwNiA2NS44MzcgNTYuMjA4IDY3LjcxNyA1OS41OTAxIDcwLjc5NzZINTkuNTg2NEM1OS43OTIxIDcwLjk4NDkgNTkuOTkwMyA3MS4xNzIxIDYwLjE4NSA3MS4zNjY3QzY2LjIyODggNzcuNDAzMiA2OC4wMDk2IDg3Ljk3ODIgNjkuODMxMSA5OC45OTc3QzY5LjkwNDUgOTkuNDMxIDcwLjI3NTQgOTkuNzM1NyA3MC42OTc3IDk5LjczNTdINzAuNzA4N1Y5OS43Mjg0SDcwLjcxMjNMNzAuNzE5NyA5OS43MzU3SDcwLjcyMzRINzAuNzM0NEg3MC43NDE3QzcwLjc1NTkgOTkuNzM1NyA3MC43Njg0IDk5LjczNTIgNzAuNzgwNiA5OS43MzQzQzcwLjc5MzggOTkuNzMzMyA3MC44MDY2IDk5LjczMiA3MC44MjA1IDk5LjczMDVMNzAuODQwOSA5OS43Mjg0SDcwLjg0NDVWOTkuNzI0N0M3MC44OTI4IDk5LjcxNjUgNzAuOTM5NSA5OS43MDQ2IDcwLjk4NDUgOTkuNjg5MkM3MS4yOTQ5IDk5LjU5OCA3MS41NDE5IDk5LjMzNzMgNzEuNTk3MyA5OC45OTY4QzcxLjY1MyA5OC42NjAxIDcxLjcwODYgOTguMzIzOCA3MS43NjQzIDk3Ljk4OEg3MS43NjYyTDcxLjc2NDYgOTcuOTg2NEw3MS43NjU1IDk3Ljk4MDZINzEuNzc3MkM3My40NzcyIDg3LjcyNTMgNzUuMjYxNyA3Ny45NjkyIDgwLjYyMjcgNzIuMDMyMkw4MC42MTkgNzIuMDM1OUw4MC42MjI3IDcyLjAxNzZMODAuNjIxOSA3Mi4wMTY4QzgwLjgyNTIgNzEuNzkyMSA4MS4wMzM1IDcxLjU3MjkgODEuMjQ3MiA3MS4zNTkzQzg3LjI5MTMgNjUuMzIyOCA5Ny43MTk2IDYzLjY4ODggMTA4LjU5MSA2Mi4wMjU1QzEwOS4wNjUgNjEuOTUyMSAxMDkuMzkxIDYxLjUwNzcgMTA5LjMxOCA2MS4wMzQxQzEwOS4zMTEgNjAuOTkgMTA5LjMwMiA2MC45NDczIDEwOS4yODkgNjAuOTA2QzEwOS4zMDIgNjAuODY0NyAxMDkuMzExIDYwLjgyMTkgMTA5LjMxOCA2MC43Nzc5QzEwOS4zNTggNjAuNTE5NSAxMDkuMjc5IDYwLjI2OTggMTA5LjEyMiA2MC4wODQ0QzEwOC45OTMgNTkuOTI5NCAxMDguODA4IDU5LjgxOTQgMTA4LjU5MiA1OS43ODY0TDEwOC41ODcgNTkuNzg1N1pNNjUuMTU5OCA3NS4xNjAxQzY1LjE5NDEgNzUuMjIyIDY1LjIyODMgNzUuMjg0IDY1LjI2MjIgNzUuMzQ2MUM2NS4yNTgxIDc1LjMxNTcgNjUuMjI0IDc1LjI1MzggNjUuMTU5OCA3NS4xNjAxWiIgZmlsbD0id2hpdGUiLz4KPC9zdmc+Cg==" alt="Saachi AI Logo" width="30">
        <h1>Saachi AI - Candidate Search</h1>
    </div>
""", unsafe_allow_html=True)

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

# Parse command line args for Streamlit
def get_streamlit_query_params():
    """Get query parameters that can be passed to Streamlit."""
    query_params = {}
    
    # Check for query parameters in format: key=value
    for arg in sys.argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            query_params[key] = value
    
    return query_params

# Get query parameters
query_params = get_streamlit_query_params()

# Initialize Profile Evaluator
@st.cache_resource
def get_profile_evaluator():
    # Get LLM choice from query parameters or environment variable
    llm_choice = query_params.get("llm", "gemini")  # Changed default to Gemini
    
    # Get batch size from query parameters (default is 20)
    batch_size = 20  # Changed default to 20
    if "batch" in query_params:
        try:
            batch_size = int(query_params.get("batch"))
            st.sidebar.info(f"Using batch size: {batch_size}")
        except ValueError:
            st.sidebar.warning(f"Invalid batch size: {query_params.get('batch')}. Using default: 6")
    
    # Set API keys based on LLM choice
    api_key = None
    if llm_choice == "gemini":
        # For Gemini, use GOOGLE_API_KEY
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found in environment variables")
            st.warning("Please add your Google API key to your .env file or Streamlit secrets")
    else:
        # For Grok, use XAI_API_KEY
        api_key = env_vars["xai_api_key"]
    
    # Display LLM choice in sidebar
    st.sidebar.info(f"Using LLM: {llm_choice}")
    
    # Initialize the evaluator with the batch size
    evaluator = IndividualProfileEvaluator(api_key=api_key, llm_choice=llm_choice)
    
    # Store the batch size for later use in evaluation
    evaluator.batch_size = batch_size
    
    return evaluator

try:
    profile_evaluator = get_profile_evaluator()
except Exception as e:
    st.error(f"Error initializing profile evaluator: {str(e)}")
    st.warning("Make sure XAI_API_KEY is set in your .env file")
    profile_evaluator = None

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
    namespace = ""
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
    
    # If there's an existing metadata filter, use it as is
    if metadata_filter:
        # Use the metadata filter as provided
        pass
    else:
        # If no metadata filter exists, use an empty one
        metadata_filter = {}
    
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
            profile_retriever=profile_retriever,
            profile_evaluator=profile_evaluator
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
            profile_retriever=profile_retriever,
            profile_evaluator=profile_evaluator
        )