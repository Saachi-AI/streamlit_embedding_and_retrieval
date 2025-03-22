import os
import json
import streamlit as st
from langsmith import Client
from langsmith.run_helpers import traceable

from utils import load_environment, get_vector_store
from embedders.openai_embedder import OpenAIEmbedder
from embedders.cohere_embedder import CohereEmbedder
from filter_extractor import FilterExtractor
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
            results, _ = retrieve_documents(query, model_choice, num_results, metadata_filter)
            
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
                        </ul>
                        <p style="margin-top: 8px; font-size: 0.9em;">
                            Metadata filtering mode: <span style="font-weight: bold;">Inclusive</span><br>
                            Metadata filters are applied first at the database level, then semantic search retrieves the most relevant matches.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader(f"Retrieved {len(results)} chunks (filtered from {total_chunks} total vectors)")
                else:
                    st.subheader(f"Retrieved {len(results)} chunks (from {total_chunks} total vectors)")
                
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
                    with st.expander(f"Result {i+1} - Profile: {profile_id}, Section: {section}", expanded=(i == 0)):
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
            else:
                st.info("No results found. Try adjusting your query or filters.")
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.info(f"Make sure you've embedded documents with this model first. Run `python embedders/embed.py --model {model_choice}`")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangChain, Pinecone, Groq, and LangSmith")