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
    display_ranked_candidates,
    display_reference_dimensions
)
from core.state_management import initialize_tab_state, get_tab_state, set_tab_state
from llm_profile_ranking import LLMProfileRanker
from profile_evaluator import IndividualProfileEvaluator
from profile_rank_processor import ProfileRankProcessor
from core.filter_editor_components import render_filter_editor
from core.search_parameters import render_search_parameters, get_search_parameters
from core.llm_handler import update_profile_evaluator_settings

# Configure logging
logger = logging.getLogger(__name__)

# Helper function for session state keys (if not already in a shared core module)
def get_tab_state_key(tab_id: str, key: str) -> str:
    return f"{tab_id}_{key}"

def initialize_custom_query_state():
    """Initialize custom query tab-specific state variables."""
    defaults = {
        "semantic_top_k": 30,
        "rerank_top_k": 30,
        "enable_metadata_filtering": True,
        "query_executed": False,
        "query": None,
        "editable_custom_dimensions": None, # For custom query tab
        "custom_weights_editor_initialized": False,
        "custom_weights_confirmed": False,
    }
    initialize_tab_state("tab1", defaults)

def adjust_custom_dimension_weights(changed_dim_id: str):
    """Callback to adjust custom query dimension weights proportionally."""
    editable_dims_key = get_tab_state_key("tab1", "editable_custom_dimensions")
    dims = st.session_state.get(editable_dims_key)

    if not dims or not isinstance(dims, list):
        logger.warning("adjust_custom_dimension_weights called with no dimensions in session state for tab1.")
        return

    changed_dim_widget_key = get_tab_state_key("tab1", f"dim_weight_{changed_dim_id}")
    new_weight_for_changed_dim = st.session_state.get(changed_dim_widget_key, None)

    if new_weight_for_changed_dim is None:
        logger.warning(f"Could not find new weight for {changed_dim_id} in session state via key {changed_dim_widget_key} for tab1")
        return
        
    new_weight_for_changed_dim = float(new_weight_for_changed_dim)

    changed_idx = -1
    for i, dim in enumerate(dims):
        if dim.get('id') == changed_dim_id:
            changed_idx = i
            break
    
    if changed_idx == -1:
        logger.warning(f"Dimension with id {changed_dim_id} not found in editable_custom_dimensions for tab1.")
        return

    dims[changed_idx]['weight'] = max(0.0, min(100.0, new_weight_for_changed_dim))
    
    num_dims = len(dims)
    if num_dims == 0:
        return

    if num_dims == 1:
        dims[0]['weight'] = 100.0
        st.session_state[editable_dims_key] = dims
        return

    current_weight_changed_dim = dims[changed_idx]['weight']
    sum_others_before_adjustment = sum(
        d.get('weight', 0.0) for i, d in enumerate(dims) if i != changed_idx
    )
    target_sum_others = 100.0 - current_weight_changed_dim

    for i, dim in enumerate(dims):
        if i == changed_idx:
            continue
        if sum_others_before_adjustment == 0:
            dim['weight'] = target_sum_others / (num_dims - 1) if (num_dims - 1) > 0 else 0.0
        else:
            dim['weight'] = (dim.get('weight', 0.0) / sum_others_before_adjustment) * target_sum_others
        dim['weight'] = max(0.0, min(100.0, dim['weight']))

    for dim in dims:
        dim['weight'] = round(dim['weight'])

    sum_rounded_weights = sum(dim.get('weight', 0) for dim in dims)
    error = 100 - sum_rounded_weights

    if error != 0 and num_dims > 0:
        potential_new_weight_changed_dim = dims[changed_idx]['weight'] + error
        if 0 <= potential_new_weight_changed_dim <= 100:
            dims[changed_idx]['weight'] = potential_new_weight_changed_dim
        else:
            for i, dim in enumerate(dims):
                if error > 0 and dim['weight'] < 100:
                    dim['weight'] += error
                    break
                elif error < 0 and dim['weight'] > 0:
                    dim['weight'] += error
                    break
    
    for dim in dims:
        dim['weight'] = int(round(dim.get('weight',0)))

    st.session_state[editable_dims_key] = dims

def handle_query_submission(query):
    """Handle query submission."""
    if query:
        set_tab_state("tab1", "query", query)
        set_tab_state("tab1", "query_executed", True)
        # Reset states for dimension editing for the new custom query
        set_tab_state("tab1", "editable_custom_dimensions", None)
        set_tab_state("tab1", "custom_weights_editor_initialized", False)
        set_tab_state("tab1", "custom_weights_confirmed", False)
        return True
    return False

def process_custom_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator, profile_retriever, profile_evaluator):
    """Process the custom query and display results."""
    if not query:
        return
    
    # Get search parameters (highest priority)
    search_params = get_search_parameters(tab_id="shared", key_prefix="search_params")
    semantic_top_k = search_params["top_k_profiles"]
    rerank_top_k = search_params["top_k_profiles"]  # Use the full top_k_profiles value
    threshold = search_params["threshold"]
    
    # Update profile evaluator settings based on user choice
    if not update_profile_evaluator_settings(profile_evaluator, search_params):
        st.error("Failed to update LLM settings. Please check your API keys.")
        return
    
    # Fixed model choice for custom query tab
    model_choice = "cohere"
    
    # Always extract metadata filters
    metadata_filter = None
    extracted_filters = None
    
    # STEP 1: Extract metadata filters and allow user to edit them
    with st.status("Step 1: Processing Custom Query for Filters...", expanded=True):
        try:
            st.write("Extracting metadata filters from your custom query...")
            filter_result = filter_extractor.process_query(query, strict_mode=False)
            metadata_filter = filter_result["pinecone_filter"]
            extracted_filters = filter_result["extracted_filters"]
            
            st.write("Please review and confirm the extracted filters below.")
            modified_filters = render_filter_editor(extracted_filters) # Pass tab_id for unique keys - REMOVED tab_id
            
            if modified_filters is None:
                st.info("Please review and adjust filters above, then click 'Confirm Filters & Proceed' to continue.")
                return # Stop if filters not confirmed
            
            metadata_filter = filter_extractor.build_pinecone_filter(modified_filters, strict_mode=False)
            st.success("Filters confirmed!")
                
        except Exception as e:
            st.error(f"Error extracting metadata filters: {str(e)}")
            logger.error(f"Error during filter extraction for custom query: {str(e)}")
            return
    add_section_separator()

    # STEP 2: Extract Custom Query Dimensions and Allow Weight Customization
    custom_weights_editor_initialized = get_tab_state("tab1", "custom_weights_editor_initialized")
    custom_weights_confirmed = get_tab_state("tab1", "custom_weights_confirmed")
    editable_custom_dimensions = get_tab_state("tab1", "editable_custom_dimensions")

    if not custom_weights_editor_initialized:
        with st.spinner("Extracting key evaluation dimensions from your Custom Query..."):
            try:
                # For custom query, the query itself is used as both raw and summarized for dimension extraction
                extracted_dimensions_data = profile_evaluator.extract_job_dimensions(raw_job_description=query, job_description_prompt=query)
                profile_evaluator.job_dimensions = extracted_dimensions_data # Ensure evaluator instance has it
                initial_dimensions = extracted_dimensions_data.get("dimensions", [])
                
                if not initial_dimensions:
                    st.error("Could not extract evaluation dimensions from your custom query. Please refine your query or try again.")
                    logger.error("Failed to extract initial dimensions for custom query.")
                    return

                # Initialize weights (similar logic to job_description.py)
                current_total_weight = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if current_total_weight == 0 and initial_dimensions:
                    equal_weight = round(100 / len(initial_dimensions))
                    for dim in initial_dimensions: dim['weight'] = equal_weight
                current_total_weight = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if current_total_weight != 100 and current_total_weight != 0 and initial_dimensions:
                    for dim in initial_dimensions: dim['weight'] = round((dim.get('weight',0) / current_total_weight) * 100)
                final_sum_check = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if final_sum_check != 100 and initial_dimensions:
                    diff = 100 - final_sum_check
                    initial_dimensions[0]['weight'] = initial_dimensions[0].get('weight',0) + diff
                for dim in initial_dimensions: dim['weight'] = int(dim.get('weight',0))

                set_tab_state("tab1", "editable_custom_dimensions", initial_dimensions)
                set_tab_state("tab1", "custom_weights_editor_initialized", True)
                st.rerun()
            except Exception as e:
                st.error(f"Error extracting dimensions for custom query: {str(e)}")
                logger.error(f"Error during dimension extraction for custom query: {str(e)}")
                return

    if not custom_weights_confirmed and custom_weights_editor_initialized and editable_custom_dimensions:
        with st.container():
            st.subheader("Step 2: Customize Evaluation Weights for Custom Query")
            st.markdown("Review and adjust the weights for each evaluation dimension. The total must sum to 100%.")
            
            total_current_weight = 0
            for dim in editable_custom_dimensions:
                dim_id = dim.get('id', 'unknown_id_custom') # Ensure unique IDs if structure is same
                dim_name = dim.get('name', 'Unknown Dimension')
                dim_weight = int(dim.get('weight', 0))
                total_current_weight += dim_weight

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{dim_name}**")
                with col2:
                    st.number_input(
                        label=f"Weight for {dim_name}", 
                        value=dim_weight, 
                        min_value=0, 
                        max_value=100, 
                        step=1, 
                        key=get_tab_state_key("tab1", f"dim_weight_{dim_id}"),
                        on_change=adjust_custom_dimension_weights,
                        args=(dim_id,),
                        label_visibility="collapsed"
                    )
                with st.expander(f"Details for {dim_name}", expanded=False):
                    st.markdown(f"**Description:** {dim.get('description', 'N/A')}")
                    st.markdown("**Key Success Factors:**")
                    for factor in dim.get('key_success_factors', []):
                        st.markdown(f"- {factor}")
                add_section_separator()

            st.markdown(f"**Total Weight: {total_current_weight}%**")
            if total_current_weight != 100:
                st.warning("Total weight does not sum to 100%. Please adjust the weights.")

            if st.button("Confirm Weights and Proceed to Evaluation", type="primary", use_container_width=True, key=get_tab_state_key("tab1", "confirm_weights_button")):
                final_dims_to_use = get_tab_state("tab1", "editable_custom_dimensions")
                if sum(d.get('weight',0) for d in final_dims_to_use) != 100:
                    st.error("Cannot proceed. Weights must sum to 100%.")
                else:
                    profile_evaluator.job_dimensions = {"dimensions": final_dims_to_use} # Set for the evaluator
                    set_tab_state("tab1", "custom_weights_confirmed", True)
                    logger.info(f"Custom query dimension weights confirmed by user: {final_dims_to_use}")
                    st.rerun()
            return

    # Proceed with search and evaluation only if filters and weights are confirmed
    if not (get_tab_state("tab1", "query_executed") and custom_weights_confirmed):
        if not custom_weights_confirmed and get_tab_state("tab1", "custom_weights_editor_initialized"):
            logger.debug("Waiting for custom query weight confirmation.")
        else:
            logger.debug("Custom query or its weights not confirmed. Halting before search.")
        return

    # Display the confirmed dimensions as a reference
    if custom_weights_confirmed and editable_custom_dimensions:
        display_reference_dimensions(editable_custom_dimensions, title="Reference Custom Query Weights")
        add_section_separator()
    
    # Create a progress bar and message display area for showing process steps
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Vector retrieval and filtering (20%)
        status_text.text("Searching vector database for matching candidates...")
        
        # Get the total vector count first
        _, total_chunks = retrieve_documents("", model_choice, 1, None)
        results, _ = retrieve_documents(query, model_choice, semantic_top_k, metadata_filter, threshold=threshold)
        
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
        profile_scores = profile_aggregator.aggregate_profiles(reranked_results, threshold=threshold, top_k=semantic_top_k)
        
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
