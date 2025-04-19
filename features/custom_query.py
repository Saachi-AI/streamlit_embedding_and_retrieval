import streamlit as st
import json
import os
import logging

from pinecone import Pinecone
from core.ui_components import (
    display_retrieval_stats, 
    display_initial_results_summary,
    display_detailed_results,
    add_section_separator,
    display_reranked_results_header,
    display_reranked_results_summary,
    display_reranked_detailed_results,
    display_profile_results,
    display_profile_summary,
    display_profile_retrieval_and_preprocessing,
    display_ranked_candidates,
    display_individual_profile_evaluations
)
from core.filter_editor_components import render_filter_editor

from profile_preprocessor import preprocess_profiles
from llm_profile_ranking import LLMProfileRanker
from individual_profile_evaluator import IndividualProfileEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_custom_query_state():
    """Initialize session state variables for the custom query tab."""
    # If not already set, create default values for our settings
    if "tab1_semantic_top_k" not in st.session_state:
        st.session_state.tab1_semantic_top_k = 15  # Default value
        
    if "tab1_rerank_top_k" not in st.session_state:
        st.session_state.tab1_rerank_top_k = 10  # Default value
        
    # Feature flag for individual profile evaluation
    if "use_individual_profile_evaluator_custom" not in st.session_state:
        st.session_state.use_individual_profile_evaluator_custom = True

def process_custom_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator, profile_retriever):
    """Process the custom query and display results."""
    if not query:
        return
    
    # Get values directly from session state (highest priority)
    semantic_top_k = st.session_state.get("tab1_semantic_top_k", 15)  # Use our new default value
    rerank_top_k = st.session_state.get("tab1_rerank_top_k", 10)  # Use our new default value
    model_choice = settings.get("model_name", "cohere")
    
    # Always extract metadata filters
    metadata_filter = None
    extracted_filters = None
    
    with st.spinner("Extracting metadata filters..."):
        try:
            filter_result = filter_extractor.process_query(query, strict_mode=False)
            metadata_filter = filter_result["pinecone_filter"]
            extracted_filters = filter_result["extracted_filters"]
            
            # Show filter editor and wait for user confirmation
            modified_filters = render_filter_editor(extracted_filters)
            
            # If user hasn't confirmed yet, stop here
            if modified_filters is None:
                st.info("Please review the filters above and click 'Confirm & Proceed' to continue with the search.")
                return
            
            # User has confirmed, build new Pinecone filter with modified filters
            metadata_filter = filter_extractor.build_pinecone_filter(modified_filters, strict_mode=False)
            st.success("Filters confirmed! Proceeding with search...")
                
        except Exception as e:
            st.error(f"Error extracting metadata filters: {str(e)}")
    
    # Perform document retrieval
    with st.spinner(f"Retrieving with {model_choice} embeddings..."):
        try:
            # Get the total vector count first
            _, total_chunks = retrieve_documents("", model_choice, 1, None)
            
            
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
                with st.expander("Initial Retrieval Results", expanded=False):
                    display_initial_results_summary(results)
                
                # Display detailed results for each document
                display_detailed_results(results, tab_prefix="tab1_")
                
                # Add spacing between sections
                add_section_separator()
                
                # Perform reranking
                with st.spinner("Reranking results with Cohere..."):
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
                    
                    # After reranking and before displaying results
                    # Aggregate profiles
                    profile_scores = profile_aggregator.aggregate_profiles(reranked_results)
                    
                    # Prepare profile entries for retrieval
                    profile_entries = profile_aggregator.prepare_for_profile_retrieval(profile_scores)
                    if not profile_entries:
                        st.warning("No valid profiles found for retrieval")
                        return
                    
                    # Retrieve and preprocess profile data
                    try:
                        # Retrieve profile data (set preprocess=False to get raw dictionary data)
                        profile_data = profile_retriever.retrieve_profile_data(
                            profile_entries=profile_entries,
                            preprocess=False
                        )
                        
                        # Preprocess the profiles
                        processed_profiles = preprocess_profiles(
                            profile_data=profile_data
                        )
                        
                        if not processed_profiles:
                            st.warning("No valid profiles after preprocessing")
                            return
                            
                    except Exception as e:
                        st.error(f"Error in profile retrieval and preprocessing: {str(e)}")
                        logger.error(f"Error in profile retrieval and preprocessing: {str(e)}")
                        return
                    
                    # Display profile-level results
                    if profile_scores:
                        # Display profile results
                        display_profile_results(profile_scores, profile_aggregator, tab_prefix="tab1_")
                        
                        # Display profile summary
                        display_profile_summary(profile_scores)
                    else:
                        st.warning("No profiles could be aggregated from the reranked results.")
                    
                    # Display profile retrieval and preprocessing results for debugging
                    display_profile_retrieval_and_preprocessing(profile_data, processed_profiles)
                    
                    # Check feature flag for individual profile evaluation
                    if st.session_state.get("use_individual_profile_evaluator_custom", True):
                        # New approach: Evaluate profiles individually
                        with st.spinner("Evaluating profiles individually..."):
                            st.info("Using individual profile evaluation approach")
                            
                            # Initialize the individual profile evaluator
                            individual_evaluator = IndividualProfileEvaluator()
                            
                            # Evaluate profiles individually
                            evaluation_results = individual_evaluator.evaluate_profiles(
                                processed_profiles=processed_profiles,
                                raw_job_description=query,  # Use the custom query as job description
                            )
                            
                            # Display individual profile evaluations
                            if evaluation_results:
                                add_section_separator()
                                display_individual_profile_evaluations(evaluation_results)
                            else:
                                st.warning("No individual profile evaluation results available.")
                    else:
                        # Original approach: Evaluate all profiles together
                        with st.spinner("Ranking profiles with LLM..."):
                            st.info("Using original profile ranking approach")
                            
                            # Call LLM for profile ranking
                            llm_ranker = LLMProfileRanker()
                            llm_ranking_results = llm_ranker.rank_profiles_custom_query(
                                processed_profiles=processed_profiles,
                                custom_query=query
                            )
                            
                            # Process ranked profiles for display
                            if llm_ranking_results:
                                from profile_rank_processor import ProfileRankProcessor
                                profile_rank_processor = ProfileRankProcessor()
                                processed_candidates = profile_rank_processor.process_ranked_profiles(
                                    llm_ranking_results=llm_ranking_results,
                                    profile_data=profile_data
                                )
                                
                                # Display ranked candidates
                                add_section_separator()
                                display_ranked_candidates(processed_candidates)
                            else:
                                st.warning("No profile ranking results available.")
                else:
                    st.warning("No reranked results found.")
            else:
                st.warning("No results found for the given query and filters.")
                
        except Exception as e:
            st.error(f"Error in document retrieval and reranking: {str(e)}")
            logger.error(f"Error in document retrieval and reranking: {str(e)}")

def render_custom_query_tab(
    filter_extractor,
    embedders,
    retrieve_documents,
    cohere_reranker,
    profile_aggregator,
    profile_retriever
):
    """Render the custom query tab."""
    # Initialize session state
    initialize_custom_query_state()
    
    # Set up the UI
    st.header("🔍 Custom Search Query")
    st.markdown("""
    Enter a free-form query to search for candidates with specific skills, experience, or qualifications.
    """)
    
    # Display toggle for profile evaluation method
    st.sidebar.markdown("### Profile Evaluation Method")
    use_individual = st.sidebar.toggle(
        "Use Individual Profile Evaluation", 
        value=st.session_state.get("use_individual_profile_evaluator_custom", True),
        help="When enabled, each profile is evaluated individually against your query."
    )
    st.session_state.use_individual_profile_evaluator_custom = use_individual
    
    # Create a text area for input
    query = st.text_area(
        "Custom Search Query",
        value="",
        height=150,
        key="tab1_query_input",
        help="Describe the type of candidate you're looking for."
    )
    
    # Display button
    if st.button("🔍 Search", key="tab1_search_button", use_container_width=True):
        if query:
            # Create a dictionary of settings (will be extended later)
            settings = {
                "model_name": "cohere",
                "enable_metadata_filtering": True,
                "semantic_top_k": st.session_state.tab1_semantic_top_k,
                "rerank_top_k": st.session_state.tab1_rerank_top_k
            }
            
            process_custom_query(
                query, 
                settings, 
                filter_extractor, 
                embedders,
                retrieve_documents,
                cohere_reranker,
                profile_aggregator,
                profile_retriever
            )
        else:
            st.warning("Please enter a search query first.")
