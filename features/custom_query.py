import streamlit as st
import os
import logging
from profile_preprocessor import preprocess_profiles
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
    display_fallback_results,
    display_profile_retrieval_and_preprocessing,
    display_ranked_candidates
)
from core.state_management import initialize_tab_state, get_tab_state, set_tab_state
from llm_profile_ranking import LLMProfileRanker
from profile_evaluator import IndividualProfileEvaluator
from profile_rank_processor import ProfileRankProcessor
from core.filter_editor_components import render_filter_editor

# Configure logging
logger = logging.getLogger(__name__)

def initialize_custom_query_state():
    """Initialize custom query tab-specific state variables."""
    # Use hardcoded defaults instead of environment variables
    defaults = {
        "semantic_top_k": 15,
        "rerank_top_k": 10,
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

def process_custom_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator, profile_retriever, profile_evaluator):
    """Process the custom query and display results."""
    if not query:
        return
    
    # Get values directly from session state (highest priority)
    semantic_top_k = st.session_state.get("tab1_semantic_top_k", 15)  # Use our new default value
    rerank_top_k = st.session_state.get("tab1_rerank_top_k", 10)  # Use our new default value
    
    # Fixed model choice for custom query tab
    model_choice = "cohere"
    
    # Always extract metadata filters
    metadata_filter = None
    extracted_filters = None
    
    with st.spinner("Extracting metadata filters..."):
        try:
            filter_result = filter_extractor.process_query(query, strict_mode=False)
            metadata_filter = filter_result["pinecone_filter"]
            extracted_filters = filter_result["extracted_filters"]
            
            # Pass extracted filters to the filter editor
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
    
    # Create a progress bar and message display area for showing process steps
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Vector retrieval and filtering (20%)
        status_text.text("Searching vector database for matching candidates...")
        
        # Get the total vector count first
        _, total_chunks = retrieve_documents("", model_choice, 1, None)
        results, _ = retrieve_documents(query, model_choice, semantic_top_k, metadata_filter)
        
        if not results:
            progress_bar.empty()
            status_text.empty()
            st.info("No results found. Try adjusting your query or filters.")
            return
        
        progress_bar.progress(20)
        
        # Step 2: Reranking (40%)
        status_text.text("Reranking candidates by relevance...")
        reranked_results = cohere_reranker.rerank(
            query, 
            results, 
            rerank_top_k
        )
            
        if not reranked_results:
            progress_bar.empty()
            status_text.empty()
            st.warning("Reranking failed. Please try again.")
            return
        
        progress_bar.progress(40)
        
        # Step 3: Aggregate profiles (60%)
        status_text.text("Scoring and aggregating profiles...")
        profile_scores = profile_aggregator.aggregate_profiles(reranked_results)
        
        # Prepare profile entries for retrieval
        profile_entries = profile_aggregator.prepare_for_profile_retrieval(profile_scores)
        if not profile_entries:
            progress_bar.empty()
            status_text.empty()
            st.warning("No valid profiles found for retrieval")
            return
        
        progress_bar.progress(60)
        
        # Step 4: Retrieve and preprocess profile data (80%)
        status_text.text("Retrieving latest data from Tamago and LinkedIn...")
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
                progress_bar.empty()
                status_text.empty()
                st.warning("No valid profiles after preprocessing")
                return
                
            # Store processed profiles for later use
            st.session_state.processed_profiles = processed_profiles
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error in profile retrieval and preprocessing: {str(e)}")
            logger.error(f"Error in profile retrieval and preprocessing: {str(e)}")
            return
        
        progress_bar.progress(80)
        
        # Step 5: Call profile evaluator for individual profile assessment (100%)
        status_text.text("Evaluating top profiles with LLM...")
        try:
            custom_query_text = get_tab_state("tab1", "query")
            
            # Use the passed-in profile_evaluator if available
            if profile_evaluator:
                evaluation_results = profile_evaluator.evaluate_profiles_custom_query(
                    processed_profiles=processed_profiles,
                    custom_query=custom_query_text,
                    batch_size=getattr(profile_evaluator, 'batch_size', 6)
                )
            else:
                # Fall back to creating a new one if not available
                logger.info("Creating profile evaluator instance since none was passed")
                profile_evaluator = IndividualProfileEvaluator()
                evaluation_results = profile_evaluator.evaluate_profiles_custom_query(
                    processed_profiles=processed_profiles,
                    custom_query=custom_query_text
                )
            
            # Process evaluated profiles for display
            if evaluation_results and evaluation_results.get("profiles"):
                profile_rank_processor = ProfileRankProcessor()
                processed_candidates = profile_rank_processor.process_evaluated_profiles(
                    evaluation_results=evaluation_results,
                    profile_data=profile_data
                )
                
                # Complete the progress
                progress_bar.progress(100)
                
                # Clear the status message and progress bar
                progress_bar.empty()
                status_text.empty()
                
                # Display only the final ranked candidates
                add_section_separator()
                display_ranked_candidates(processed_candidates)
            
            logger.info("Profile Evaluation completed")
        except Exception as e:
            logger.error(f"Error during profile evaluation: {str(e)}")
            
            # Fall back to old ranking method if new evaluation fails
            try:
                status_text.text("Falling back to alternative ranking method...")
                logger.info("Falling back to legacy LLM profile ranking")
                llm_ranker = LLMProfileRanker()
                llm_ranking_results = llm_ranker.rank_profiles_custom_query(
                    processed_profiles=processed_profiles,
                    custom_query=custom_query_text
                )
                
                # Process ranked profiles for display
                if llm_ranking_results:
                    profile_rank_processor = ProfileRankProcessor()
                    processed_candidates = profile_rank_processor.process_ranked_profiles(
                        llm_ranking_results=llm_ranking_results,
                        profile_data=profile_data
                    )
                    
                    # Complete the progress
                    progress_bar.progress(100)
                    
                    # Clear the status message and progress bar
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display ranked candidates
                    add_section_separator()
                    display_ranked_candidates(processed_candidates)
                
                logger.info("Legacy LLM Profile Ranking completed")
            except Exception as e2:
                # Clear the status message and progress bar
                progress_bar.empty()
                status_text.empty()
                
                logger.error(f"Error during fallback LLM profile ranking: {str(e2)}")
                st.error(f"Error finding matching candidates: {str(e2)}")
            
    except Exception as e:
        # Clear the status message and progress bar
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"Error during search: {str(e)}")
        logger.error(f"Error during search: {str(e)}")

def render_custom_query_tab(
    filter_extractor,
    embedders,
    retrieve_documents,
    cohere_reranker,
    profile_aggregator,
    profile_retriever,
    profile_evaluator
):
    """Render the custom query tab content."""
    # Initialize tab state if not already initialized
    initialize_custom_query_state()
    
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
            profile_aggregator,
            profile_retriever,
            profile_evaluator
        )
