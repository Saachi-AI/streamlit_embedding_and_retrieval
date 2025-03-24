import os
import json
import streamlit as st
from langsmith import Client
from langsmith.run_helpers import traceable

from utils import load_environment, get_vector_store
from embedders.openai_embedder import OpenAIEmbedder
from embedders.cohere_embedder import CohereEmbedder
from filter_extractor import FilterExtractor
from rerankers.cohere_reranker import CohereReranker
from post_rerank_aggregator import ProfileAggregator
from pinecone import Pinecone

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

# Get default values from environment variables
default_semantic_top_k = int(os.getenv("SEMANTIC_TOP_K", 10))
default_rerank_top_k = int(os.getenv("RERANK_TOP_K", 5))
default_top_k_profiles = int(os.getenv("TOP_K_PROFILES", 5))

# Initialize sliders in session state if not already set
if 'semantic_top_k' not in st.session_state:
    st.session_state.semantic_top_k = default_semantic_top_k
if 'rerank_top_k' not in st.session_state:
    st.session_state.rerank_top_k = min(default_rerank_top_k, default_semantic_top_k)
if 'top_k_profiles' not in st.session_state:
    st.session_state.top_k_profiles = default_top_k_profiles

# Add sliders for semantic_top_k and rerank_top_k (using session state to persist values)
semantic_top_k = st.sidebar.slider(
    "Number of Retrieval Chunks",
    min_value=1,
    max_value=20,
    value=st.session_state.semantic_top_k,
    key='semantic_top_k_slider'
)

# Update the session state value
st.session_state.semantic_top_k = semantic_top_k

# Make rerank_top_k slider depend on semantic_top_k
rerank_top_k = st.sidebar.slider(
    "Number of Reranked Chunks",
    min_value=1,
    max_value=semantic_top_k,  # Limit to the number of retrieved chunks
    value=min(st.session_state.rerank_top_k, semantic_top_k),  # Ensure default doesn't exceed semantic_top_k
    key='rerank_top_k_slider'
)

# Update the session state value
st.session_state.rerank_top_k = rerank_top_k

# Add input for number of profiles to show
top_k_profiles = st.sidebar.number_input(
    "Number of Top Profiles to Show",
    min_value=1,
    max_value=20,
    value=st.session_state.top_k_profiles,
    key='top_k_profiles_input'
)

# Update the session state value
st.session_state.top_k_profiles = top_k_profiles

# Set a fixed rerank model (always use English)
rerank_model = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
st.session_state.rerank_model = rerank_model

# Make sure we set the environment variable immediately
os.environ["RERANK_MODEL"] = rerank_model

# Override environment variables with the values from the UI
os.environ["SEMANTIC_TOP_K"] = str(semantic_top_k)
os.environ["RERANK_TOP_K"] = str(rerank_top_k)
os.environ["TOP_K_PROFILES"] = str(top_k_profiles)

# Enable/disable metadata filtering
enable_metadata_filtering = st.sidebar.checkbox("Enable Metadata Filtering", value=True)

# Main query input
st.text_area(
    "Enter your query:", 
    key="query_input", 
    height=150,
    placeholder="Example: Find candidates who speak fluent Japanese with at least 5 years of experience"
)

# Submit button for retrieval
query_submitted = st.button("Search", type="primary", use_container_width=True)

# Store query in session state
if query_submitted and st.session_state.query_input:
    query = st.session_state.query_input
elif not 'query' in locals():
    query = ""

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
            k=top_k,
            filter=metadata_filter
        )
    else:
        results = vector_store.similarity_search_with_score(query, k=top_k)
    
    # Sort results by relevance (higher percentage first)
    results.sort(key=lambda x: x[1])
    
    return results, total_chunks

# Perform retrieval when query is provided
if query and query_submitted:
    # Process the query to extract metadata filters if enabled
    metadata_filter = None
    extracted_filters = None
    
    if enable_metadata_filtering:
        with st.spinner("Extracting metadata filters..."):
            try:
                filter_result = filter_extractor.process_query(query, strict_mode=False)
                metadata_filter = filter_result["pinecone_filter"]
                extracted_filters = filter_result["extracted_filters"]
            except Exception as e:
                st.error(f"Error extracting metadata filters: {str(e)}")
    
    # Display the extracted filters if any
    if extracted_filters and metadata_filter and metadata_filter != {}:
        st.subheader("Detected Filters")
        
        # Create a visually appealing filter display
        filter_cols = st.columns(3)
        col_index = 0  # Keep track of which column we're using
        
        # Experience filter
        if "years_of_experience" in extracted_filters:
            with filter_cols[col_index % 3]:
                st.markdown(f"""
                <div style="background-color: #2a4d3e; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <span style="font-weight: bold;">Experience:</span> {extracted_filters["years_of_experience"]}+ years
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
                
        # Gender filter
        if "gender" in extracted_filters:
            with filter_cols[col_index % 3]:
                st.markdown(f"""
                <div style="background-color: #263e5a; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <span style="font-weight: bold;">Gender:</span> {extracted_filters["gender"]}
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
        
        # Last contacted filter
        if "last_contacted" in extracted_filters:
            with filter_cols[col_index % 3]:
                st.markdown(f"""
                <div style="background-color: #554927; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <span style="font-weight: bold;">Last Contacted:</span> Within {extracted_filters["last_contacted"]} years
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
                
        # Is candidate filter - only if explicitly mentioned
        if "is_candidate" in extracted_filters:
            with filter_cols[col_index % 3]:
                st.markdown(f"""
                <div style="background-color: #44304d; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <span style="font-weight: bold;">Is Candidate:</span> {extracted_filters["is_candidate"]}
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
        
        # Placed filter - only if explicitly mentioned
        if "placed" in extracted_filters:
            with filter_cols[col_index % 3]:
                st.markdown(f"""
                <div style="background-color: #4d2e2a; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <span style="font-weight: bold;">Placed:</span> {extracted_filters["placed"]}
                </div>
                """, unsafe_allow_html=True)
                col_index += 1
        
        # Display language requirements
        if "languages" in extracted_filters and isinstance(extracted_filters["languages"], dict):
            st.markdown("<span style='font-weight: bold; color: white; margin-top: 15px; display: block;'>Languages:</span>", unsafe_allow_html=True)
            language_cols = st.columns(2)
            
            for i, (language, level) in enumerate(extracted_filters["languages"].items()):
                col_idx = i % 2
                with language_cols[col_idx]:
                    st.markdown(f"""
                    <div style="background-color: #2a4e5a; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                        <span style="font-weight: bold;">{language.capitalize()}:</span> {level}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Display the raw filter (collapsible)
        with st.expander("View Raw Pinecone Filter"):
            st.code(json.dumps(metadata_filter, indent=2), language="json")
    
    # Perform document retrieval
    with st.spinner(f"Retrieving with {model_choice} embeddings..."):
        try:
            # Get the total vector count first
            _, total_chunks = retrieve_documents("", model_choice, 1, None)
            
            # Get actual search results - if metadata filtering is enabled, it's applied first at the database level
            results, _ = retrieve_documents(query, model_choice, semantic_top_k, metadata_filter)
            
            # Display results
            if results:
                if metadata_filter and metadata_filter != {} and extracted_filters:
                    # Get the approximate size of the filtered set from Pinecone
                    try:
                        # Initialize Pinecone directly for more detailed stats
                        pc = Pinecone(api_key=env_vars["pinecone_api_key"])
                        index = pc.Index(env_vars["pinecone_index_name"])
                        namespace = f"{model_choice.lower().replace('-', '_')}_embeddings"
                        
                        # Get filtered stats by directly querying Pinecone with the filter
                        # This is an approximation since we can't get exact count without a query
                        count_query = index.query(
                            namespace=namespace,
                            vector=[0] * (1536 if model_choice == "openai" else 1024),  # Dummy vector
                            filter=metadata_filter,
                            top_k=1000,  # Request more results to get a better estimate
                            include_metadata=False
                        )
                        # Get approximate filtered count based on returned results
                        filtered_size = len(count_query.get('matches', []))
                        
                        # Ensure consistency in the reported numbers
                        # The final results count should not exceed the filtered count
                        if filtered_size < len(results):
                            # If our estimate is off, adjust it to match reality
                            filtered_size = max(len(results), filtered_size)
                            
                        filtered_out = total_chunks - filtered_size if filtered_size > 0 else "Unknown"
                    except Exception as e:
                        st.warning(f"Unable to get precise filter stats: {e}")
                        filtered_size = len(results)  # Fall back to results count
                        filtered_out = "Unknown"
                    
                    # Display statistics about the filtering process
                    st.markdown(f"""
                    <div style="background-color: #37474F; color: white; padding: 10px; border-radius: 5px; margin: 15px 0;">
                        <div style="font-weight: bold; margin-bottom: 5px;">Vector Retrieval Pipeline:</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li>Total corpus size: {total_chunks} vectors</li>
                            <li>After metadata filtering: {filtered_size} vectors (filtered out {filtered_out if isinstance(filtered_out, str) else total_chunks - filtered_size} vectors)</li>
                            <li>Top semantic matches: {len(results)} vectors</li>
                            <li>After reranking: {rerank_top_k} vectors</li>
                        </ul>
                        <p style="margin-top: 8px; font-size: 0.9em;">
                            Metadata filtering mode: <span style="font-weight: bold;">Inclusive</span><br>
                            Metadata filters are applied first at the database level, then semantic search retrieves the most relevant matches.
                            Finally, Cohere reranking re-scores documents based on semantic relevance.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader(f"Retrieved {len(results)} chunks (filtered from {total_chunks} total vectors)")
                else:
                    st.subheader(f"Retrieved {len(results)} chunks (from {total_chunks} total vectors)")
                
                # Display a quick summary of the initial retrieval results
                with st.expander("Initial Retrieval Results", expanded=True):
                    # Create a quick summary table of the initial results
                    st.markdown("<div style='margin-bottom: 15px;'>These are the initial semantic search results before reranking:</div>", unsafe_allow_html=True)
                    
                    # Create columns for the table header
                    cols = st.columns([0.15, 0.15, 0.25, 0.45])
                    cols[0].markdown("<div style='font-weight: bold;'>Rank</div>", unsafe_allow_html=True)
                    cols[1].markdown("<div style='font-weight: bold;'>Score</div>", unsafe_allow_html=True)
                    cols[2].markdown("<div style='font-weight: bold;'>Profile ID</div>", unsafe_allow_html=True)
                    cols[3].markdown("<div style='font-weight: bold;'>Section</div>", unsafe_allow_html=True)
                    
                    # Display summary of each result
                    for i, (doc, score) in enumerate(results):
                        relevance_percentage = (1 - score) * 100
                        profile_id = doc.metadata.get("profile_id", "N/A")
                        if isinstance(profile_id, (int, float)):
                            profile_id = str(int(profile_id))
                        section = doc.metadata.get("section", "N/A")
                        
                        cols = st.columns([0.15, 0.15, 0.25, 0.45])
                        cols[0].markdown(f"{i+1}")
                        cols[1].markdown(f"{relevance_percentage:.2f}%")
                        cols[2].markdown(f"{profile_id}")
                        cols[3].markdown(f"{section}")
                
                # Display detailed results for each document separately (not inside an expander)
                st.subheader("Detailed Initial Results")
                detailed_tabs = st.tabs([f"Result {i+1}" for i in range(len(results))])

                for i, (tab, (doc, score)) in enumerate(zip(detailed_tabs, results)):
                    with tab:
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
                        
                        # Display custom header
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # Metadata section with improved styling (now displayed first)
                        st.markdown("<h3 style='margin-top: 15px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Metadata</h3>", unsafe_allow_html=True)
                        # Create a cleaner metadata display
                        st.json(doc.metadata)
                        
                        # Content section with improved styling (moved after metadata)
                        st.markdown("<h3 style='margin-top: 20px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Content</h3>", unsafe_allow_html=True)
                        
                        # Display content in a scrollable text area for better readability
                        st.text_area(
                            label="",
                            value=doc.page_content,
                            height=200,
                            disabled=True,
                            key=f"initial_result_{i}"
                        )

                # Add spacing between sections
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #555;'>", unsafe_allow_html=True)

                # Perform reranking
                with st.spinner("Reranking results with Cohere..."):
                    # No need to pass model explicitly since it will use the environment variable
                    reranked_results = cohere_reranker.rerank(
                        query, 
                        results, 
                        rerank_top_k
                    )
                
                # Display reranked results
                if reranked_results:
                    # Enhanced title for reranked results with better styling
                    st.markdown(f"""
                    <div style="background-color: #1E3A5F; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h2 style="margin: 0; font-size: 1.5em;">Top {len(reranked_results)} Reranked Results</h2>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Reranked using Cohere's rerank-english-v3.0 model</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a summary table for reranked results
                    st.markdown("<div style='margin-bottom: 15px;'>These are the reranked results after applying Cohere's reranking:</div>", unsafe_allow_html=True)
                    
                    # Create columns for the table header
                    cols = st.columns([0.15, 0.15, 0.25, 0.45])
                    cols[0].markdown("<div style='font-weight: bold;'>Rank</div>", unsafe_allow_html=True)
                    cols[1].markdown("<div style='font-weight: bold;'>Score</div>", unsafe_allow_html=True)
                    cols[2].markdown("<div style='font-weight: bold;'>Profile ID</div>", unsafe_allow_html=True)
                    cols[3].markdown("<div style='font-weight: bold;'>Section</div>", unsafe_allow_html=True)
                    
                    # Display summary of each reranked result
                    for i, (doc, score) in enumerate(reranked_results):
                        relevance_percentage = score * 100
                        profile_id = doc.metadata.get("profile_id", "N/A")
                        if isinstance(profile_id, (int, float)):
                            profile_id = str(int(profile_id))
                        section = doc.metadata.get("section", "N/A")
                        
                        cols = st.columns([0.15, 0.15, 0.25, 0.45])
                        cols[0].markdown(f"{i+1}")
                        cols[1].markdown(f"{relevance_percentage:.2f}%")
                        cols[2].markdown(f"{profile_id}")
                        cols[3].markdown(f"{section}")
                    
                    # Create tabs for reranked results
                    reranked_tabs = st.tabs([f"Result {i+1}" for i in range(len(reranked_results))])
                    
                    # Display each reranked result in a tab
                    for i, (tab, (doc, score)) in enumerate(zip(reranked_tabs, reranked_results)):
                        with tab:
                            # Calculate relevance score as percentage with 2 decimal points
                            relevance_percentage = score * 100
                            
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
                            
                            # Display custom header
                            st.markdown(header_html, unsafe_allow_html=True)
                            
                            # Metadata section with improved styling (now displayed first)
                            st.markdown("<h3 style='margin-top: 15px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Metadata</h3>", unsafe_allow_html=True)
                            # Create a cleaner metadata display
                            st.json(doc.metadata)
                            
                            # Content section with improved styling (moved after metadata)
                            st.markdown("<h3 style='margin-top: 20px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Content</h3>", unsafe_allow_html=True)
                            
                            # Display content in a scrollable text area for better readability
                            st.text_area(
                                label="",
                                value=doc.page_content,
                                height=200,
                                disabled=True,
                                key=f"reranked_result_{i}"
                            )
                    
                    # Add spacing between sections
                    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #555;'>", unsafe_allow_html=True)
                    
                    # Perform profile aggregation
                    with st.spinner("Aggregating profiles..."):
                        profile_scores = profile_aggregator.aggregate_profiles(
                            reranked_results,
                            top_k=top_k_profiles
                        )
                    
                    # Display profile-level results
                    if profile_scores:
                        # Enhanced title for profile results
                        st.markdown(f"""
                        <div style="background-color: #1B5E20; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
                            <h2 style="margin: 0; font-size: 1.5em;">Top {len(profile_scores)} Profile Results</h2>
                            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Using Max + Bonus aggregation formula</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create profile results tabs
                        profile_tabs = st.tabs([f"#{i+1} Profile {score.profile_id}" for i, score in enumerate(profile_scores)])
                        
                        # Display each profile result in a tab
                        for i, (tab, profile_score) in enumerate(zip(profile_tabs, profile_scores)):
                            with tab:
                                # Format the score as percentage for display
                                score_percentage = profile_score.final_score * 100
                                best_chunk_percentage = profile_score.best_chunk_score * 100
                                
                                # Create header with profile info
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                                    <div style="background-color: #4CAF50; color: white; padding: 8px 16px; border-radius: 15px; font-weight: bold; min-width: 100px; text-align: center; font-size: 1.2em;">
                                        {score_percentage:.2f}%
                                    </div>
                                    <div style="background-color: #E8F5E9; padding: 8px 16px; border-radius: 15px; font-weight: 500; font-size: 1.1em;">
                                        <span style="color: #2E7D32;">Profile:</span> <span style="color: #1B5E20;">{profile_score.profile_id}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Score explanation section
                                st.markdown(f"""
                                <div style="background-color: #F1F8E9; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #8BC34A;">
                                    <h3 style="margin-top: 0; margin-bottom: 10px; color: #33691E; font-size: 1.1em;">Score Explanation</h3>
                                    <p style="margin: 0; color: #33691E;">
                                        <span style="font-weight: bold;">Best chunk score:</span> {best_chunk_percentage:.2f}%<br>
                                        <span style="font-weight: bold;">Chunks above threshold ({profile_aggregator.threshold*100:.0f}%):</span> {profile_score.above_threshold_count}<br>
                                        <span style="font-weight: bold;">Bonus factor (α):</span> {profile_aggregator.alpha}<br>
                                        <span style="font-weight: bold;">Bonus amount:</span> {profile_aggregator.alpha * profile_score.above_threshold_count:.3f}<br>
                                        <span style="font-weight: bold; font-size: 1.1em;">Final score = {best_chunk_percentage:.2f}% + {profile_aggregator.alpha * profile_score.above_threshold_count:.3f} = {score_percentage:.2f}%</span>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Relevant chunks
                                st.markdown(f"<h3 style='margin-top: 25px; margin-bottom: 15px; color: #33691E;'>Top chunks for this profile ({len(profile_score.chunks)})</h3>", unsafe_allow_html=True)
                                
                                # Display each relevant chunk with its score
                                for j, (doc, chunk_score) in enumerate(profile_score.chunks):
                                    chunk_score_percentage = chunk_score * 100
                                    section = doc.metadata.get("section", "N/A")
                                    
                                    # Determine if this chunk contributed to the bonus (above threshold)
                                    is_above_threshold = chunk_score >= profile_aggregator.threshold
                                    threshold_badge = ""
                                    if is_above_threshold:
                                        threshold_badge = f"""<div style="background-color: #689F38; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; display: inline-block; margin-left: 10px;">
                                            Above threshold
                                        </div>"""
                                    
                                    # Sanitize page content to avoid raw HTML display
                                    content = doc.page_content
                                    # Remove any HTML tags
                                    content = content.replace("<", "&lt;").replace(">", "&gt;")
                                    
                                    # Create the chunk header with score and section info, but NOT the content
                                    st.markdown(f"""
                                    <div style="background-color: #FAFAFA; padding: 15px; border-radius: 5px; margin-bottom: 15px; border: 1px solid #E0E0E0;">
                                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                            <div style="background-color: {'#689F38' if is_above_threshold else '#9E9E9E'}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold; min-width: 70px; text-align: center; margin-right: 10px;">
                                                {chunk_score_percentage:.2f}%
                                            </div>
                                            <div style="color: #424242; font-weight: 500;">
                                                Chunk #{j+1} | Section: {section}
                                            </div>
                                            {threshold_badge}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display content in a scrollable text area for better readability
                                    st.text_area(
                                        label="",
                                        value=doc.page_content[:1000] + ('...' if len(doc.page_content) > 1000 else ''),
                                        height=150,
                                        disabled=True,
                                        key=f"profile_{profile_score.profile_id}_chunk_{j}"
                                    )
                        
                        # Add a summary table for quick reference
                        st.subheader("Profile Score Summary")
                        
                        # Create columns for the table header
                        cols = st.columns([0.1, 0.2, 0.25, 0.25, 0.2])
                        cols[0].markdown("<div style='font-weight: bold;'>Rank</div>", unsafe_allow_html=True)
                        cols[1].markdown("<div style='font-weight: bold;'>Profile ID</div>", unsafe_allow_html=True)
                        cols[2].markdown("<div style='font-weight: bold;'>Best Chunk Score</div>", unsafe_allow_html=True)
                        cols[3].markdown("<div style='font-weight: bold;'>Chunks Above Threshold</div>", unsafe_allow_html=True)
                        cols[4].markdown("<div style='font-weight: bold;'>Final Score</div>", unsafe_allow_html=True)
                        
                        # Display summary of each profile score
                        for i, profile_score in enumerate(profile_scores):
                            best_chunk_percentage = profile_score.best_chunk_score * 100
                            final_score_percentage = profile_score.final_score * 100
                            
                            cols = st.columns([0.1, 0.2, 0.25, 0.25, 0.2])
                            cols[0].markdown(f"#{i+1}")
                            cols[1].markdown(f"{profile_score.profile_id}")
                            cols[2].markdown(f"{best_chunk_percentage:.2f}%")
                            cols[3].markdown(f"{profile_score.above_threshold_count}")
                            cols[4].markdown(f"{final_score_percentage:.2f}%")
                    else:
                        st.warning("No profiles could be aggregated from the reranked results.")
                else:
                    st.warning("Reranking failed. Displaying original results.")
                    # Fall back to original display code if reranking fails
                    
                    # Enhanced title for fallback results with styling similar to reranked results
                    st.markdown(f"""
                    <div style="background-color: #6D4C41; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h2 style="margin: 0; font-size: 1.5em;">Top {min(rerank_top_k, len(results))} Results (Fallback)</h2>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Using original retrieval results as fallback because reranking failed</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a summary table for fallback results
                    st.markdown("<div style='margin-bottom: 15px;'>Showing top results from initial retrieval (reranking failed):</div>", unsafe_allow_html=True)
                    
                    # Create columns for the table header
                    cols = st.columns([0.15, 0.15, 0.25, 0.45])
                    cols[0].markdown("<div style='font-weight: bold;'>Rank</div>", unsafe_allow_html=True)
                    cols[1].markdown("<div style='font-weight: bold;'>Score</div>", unsafe_allow_html=True)
                    cols[2].markdown("<div style='font-weight: bold;'>Profile ID</div>", unsafe_allow_html=True)
                    cols[3].markdown("<div style='font-weight: bold;'>Section</div>", unsafe_allow_html=True)
                    
                    # Display summary of each fallback result
                    for i, (doc, score) in enumerate(results[:rerank_top_k]):
                        relevance_percentage = (1 - score) * 100
                        profile_id = doc.metadata.get("profile_id", "N/A")
                        if isinstance(profile_id, (int, float)):
                            profile_id = str(int(profile_id))
                        section = doc.metadata.get("section", "N/A")
                        
                        cols = st.columns([0.15, 0.15, 0.25, 0.45])
                        cols[0].markdown(f"{i+1}")
                        cols[1].markdown(f"{relevance_percentage:.2f}%")
                        cols[2].markdown(f"{profile_id}")
                        cols[3].markdown(f"{section}")
                    
                    # Create tabs for fallback results
                    fallback_tabs = st.tabs([f"Result {i+1}" for i in range(min(rerank_top_k, len(results)))])
                    
                    # Display each fallback result in a tab
                    for i, (tab, (doc, score)) in enumerate(zip(fallback_tabs, results[:rerank_top_k])):
                        with tab:
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
                            
                            # Display custom header
                            st.markdown(header_html, unsafe_allow_html=True)
                            
                            # Metadata section with improved styling (now displayed first)
                            st.markdown("<h3 style='margin-top: 15px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Metadata</h3>", unsafe_allow_html=True)
                            # Create a cleaner metadata display
                            st.json(doc.metadata)
                            
                            # Content section with improved styling (moved after metadata)
                            st.markdown("<h3 style='margin-top: 20px; margin-bottom: 8px; color: #37474F; font-size: 1.2em;'>Content</h3>", unsafe_allow_html=True)
                            
                            # Display content in a scrollable text area for better readability
                            st.text_area(
                                label="",
                                value=doc.page_content,
                                height=200,
                                disabled=True,
                                key=f"fallback_result_{i}"
                            )
            else:
                st.info("No results found. Try adjusting your query or filters.")
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.info(f"Make sure you've embedded documents with this model first. Run `python embedders/embed.py --model {model_choice}`")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"Using {semantic_top_k} retrieval chunks, {rerank_top_k} reranked chunks")
st.sidebar.markdown("Built with LangChain, Pinecone, Groq, Cohere, and LangSmith")