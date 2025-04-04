import streamlit as st
import os
from core.ui_components import (
    display_retrieval_stats,
    display_initial_results_summary,
    display_detailed_results,
    display_reranked_results_header,
    display_reranked_results_summary,
    display_reranked_detailed_results,
    display_profile_results,
    display_profile_summary,
    create_sidebar_configuration,
    add_section_separator,
    display_fallback_results_header,
    display_fallback_results
)
from core.state_management import initialize_tab_state, get_tab_state, set_tab_state

def initialize_custom_query_state():
    """Initialize custom query tab-specific state variables."""
    defaults = {
        "semantic_top_k": int(os.getenv("SEMANTIC_TOP_K", 10)),
        "rerank_top_k": min(int(os.getenv("RERANK_TOP_K", 5)), int(os.getenv("SEMANTIC_TOP_K", 10))),
        "top_k_profiles": int(os.getenv("TOP_K_PROFILES", 5)),
        "enable_metadata_filtering": True,
        "query_executed": False,
        "query": None
    }
    initialize_tab_state("tab1", defaults)

def handle_query_submission(query):
    """Handle query submission."""
    if query:
        set_tab_state("tab1", "query", query)
        set_tab_state("tab1", "query_executed", True)
        return True
    return False

def process_custom_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator):
    """Process the custom query and display results."""
    if not query:
        return
    
    # Unpack settings
    semantic_top_k = settings["semantic_top_k"] 
    rerank_top_k = settings["rerank_top_k"]
    top_k_profiles = settings["top_k_profiles"]
    enable_metadata_filtering = settings["enable_metadata_filtering"]
    
    # Fixed model choice for custom query tab
    model_choice = "cohere"
    
    # Process the query
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
    
    # Perform document retrieval
    with st.spinner(f"Retrieving with {model_choice} embeddings..."):
        try:
            # Get the total vector count first
            _, total_chunks = retrieve_documents("", model_choice, 1, None)
            
            # Get actual search results
            results, _ = retrieve_documents(query, model_choice, semantic_top_k, metadata_filter)
            
            # Display results
            if results:
                # Get filtered size information if metadata filtering is enabled
                filtered_size = None
                filtered_out = None
                
                if metadata_filter and metadata_filter != {} and extracted_filters:
                    try:
                        from pinecone import Pinecone
                        import os
                        
                        # Get environment variables for Pinecone
                        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
                        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
                        
                        # Initialize Pinecone directly for more detailed stats
                        pc = Pinecone(api_key=pinecone_api_key)
                        index = pc.Index(pinecone_index_name)
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
                
                # Display retrieval statistics
                display_retrieval_stats(
                    total_chunks=total_chunks,
                    filtered_size=filtered_size,
                    results_count=len(results),
                    metadata_filter=metadata_filter,
                    filtered_out=filtered_out
                )
                
                # Display initial results summary
                with st.expander("Initial Retrieval Results", expanded=True):
                    display_initial_results_summary(results)
                
                # Display detailed results for each document
                display_detailed_results(results, tab_prefix="tab1_")
                
                # Add spacing between sections
                add_section_separator()

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
                    # Header for reranked results
                    display_reranked_results_header(len(reranked_results))
                    
                    # Display reranked results summary
                    display_reranked_results_summary(reranked_results)
                    
                    # Display detailed reranked results
                    display_reranked_detailed_results(reranked_results, tab_prefix="tab1_")
                    
                    # Add spacing between sections
                    add_section_separator()
                    
                    # Perform profile aggregation
                    with st.spinner("Aggregating profiles..."):
                        profile_scores = profile_aggregator.aggregate_profiles(
                            reranked_results,
                            top_k=top_k_profiles
                        )
                    
                    # Display profile-level results
                    if profile_scores:
                        # Display profile results
                        display_profile_results(profile_scores, profile_aggregator, tab_prefix="tab1_")
                        
                        # Display profile summary
                        display_profile_summary(profile_scores)
                    else:
                        st.warning("No profiles could be aggregated from the reranked results.")
                else:
                    st.warning("Reranking failed. Displaying original results.")
                    
                    # Display fallback results
                    display_fallback_results_header(min(rerank_top_k, len(results)))
                    display_fallback_results(results, rerank_top_k, tab_prefix="tab1_")
            else:
                st.info("No results found. Try adjusting your query or filters.")
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.info(f"Make sure you've embedded documents with this model first. Run `python embedders/embed.py --model {model_choice}`")

def render_custom_query_tab(filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator):
    """Render the custom query tab content."""
    # Initialize tab state if not already initialized
    initialize_custom_query_state()
    
    # Set the active tab in session state
    if "active_tab_index" in st.session_state:
        st.session_state.active_tab_index = 1
    
    # Tab header
    st.subheader("Custom Search")
    
    # Get sidebar configuration without rendering UI elements
    settings = create_sidebar_configuration("tab1")
    
    # Main query input
    query_input = st.text_area(
        "Enter your query:", 
        key="query_input_tab1",  # Tab-specific key for Tab 1
        height=150,
        placeholder="Example: Find candidates who speak fluent Japanese with at least 5 years of experience"
    )
    
    # Submit button with tab-specific key
    search_query_submitted = st.button("Search", type="primary", use_container_width=True, key="search_button_tab1")
    
    # Store tab1-specific query and status
    if search_query_submitted and query_input:
        handle_query_submission(query_input)
    
    # Only execute the search if the tab1 query has been submitted
    if get_tab_state("tab1", "query_executed"):
        query = get_tab_state("tab1", "query")
        process_custom_query(
            query, 
            settings, 
            filter_extractor, 
            embedders, 
            retrieve_documents, 
            cohere_reranker, 
            profile_aggregator
        )
