import streamlit as st
import os
import json
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

# Helper function for session state keys
def get_tab_state_key(tab_id: str, key: str) -> str:
    return f"{tab_id}_{key}"

def initialize_job_description_state():
    """Initialize job description tab-specific state variables."""
    defaults = {
        "semantic_top_k": 30,
        "rerank_top_k": 30,
        "enable_metadata_filtering": True,
        "query_executed": False,
        "parsed_text": None,
        "generated_prompt": None,
        "prompt_data": None,
        "query": None,
        "editable_job_dimensions": None,
        "jd_weights_editor_initialized": False,
        "jd_weights_confirmed": False,
    }
    initialize_tab_state("tab0", defaults)

def adjust_job_dimension_weights(changed_dim_id: str):
    """Callback to adjust job dimension weights proportionally."""
    # Construct the session state key for editable_job_dimensions for tab0
    editable_dims_key = get_tab_state_key("tab0", "editable_job_dimensions")
    dims = st.session_state.get(editable_dims_key)

    if not dims or not isinstance(dims, list):
        logger.warning("adjust_job_dimension_weights called with no dimensions in session state.")
        return

    # Find the changed dimension and its new weight from the widget's state
    changed_dim_widget_key = get_tab_state_key("tab0", f"dim_weight_{changed_dim_id}")
    new_weight_for_changed_dim = st.session_state.get(changed_dim_widget_key, None)

    if new_weight_for_changed_dim is None:
        logger.warning(f"Could not find new weight for {changed_dim_id} in session state via key {changed_dim_widget_key}")
        return
        
    # Convert to float for calculations, then will be rounded to int.
    new_weight_for_changed_dim = float(new_weight_for_changed_dim)

    changed_idx = -1
    for i, dim in enumerate(dims):
        if dim.get('id') == changed_dim_id:
            changed_idx = i
            break
    
    if changed_idx == -1:
        logger.warning(f"Dimension with id {changed_dim_id} not found in editable_job_dimensions.")
        return

    # Update the weight of the dimension that was directly changed by the user
    # Clamp it to be within [0, 100]
    dims[changed_idx]['weight'] = max(0.0, min(100.0, new_weight_for_changed_dim))
    
    num_dims = len(dims)
    if num_dims == 0:
        return

    if num_dims == 1:
        dims[0]['weight'] = 100.0 # Single dimension must be 100%
        st.session_state[editable_dims_key] = dims
        return

    # Current weight of the changed dimension (after clamping)
    current_weight_changed_dim = dims[changed_idx]['weight']

    # Calculate the sum of weights of OTHER dimensions (before this adjustment round)
    sum_others_before_adjustment = sum(
        d.get('weight', 0.0) for i, d in enumerate(dims) if i != changed_idx
    )

    # Target sum for other dimensions
    target_sum_others = 100.0 - current_weight_changed_dim

    # Adjust other dimensions
    for i, dim in enumerate(dims):
        if i == changed_idx:
            continue  # Skip the dimension that was manually changed

        if sum_others_before_adjustment == 0: # If all other weights were zero
            # Distribute target_sum_others equally among them (num_dims - 1 of them)
            dim['weight'] = target_sum_others / (num_dims - 1) if (num_dims - 1) > 0 else 0.0
        else:
            # Distribute proportionally based on their original share of sum_others_before_adjustment
            dim['weight'] = (dim.get('weight', 0.0) / sum_others_before_adjustment) * target_sum_others
        
        # Clamp individual weights during adjustment too
        dim['weight'] = max(0.0, min(100.0, dim['weight']))

    # Final pass to round and ensure sum is exactly 100 due to potential float inaccuracies
    # Round all weights first
    for dim in dims:
        dim['weight'] = round(dim['weight'])

    # Calculate sum of rounded weights
    sum_rounded_weights = sum(dim.get('weight', 0) for dim in dims)
    error = 100 - sum_rounded_weights

    if error != 0 and num_dims > 0:
        # Distribute the error. Add to the largest weight, subtract from largest (if error negative) or smallest.
        # A simpler way: add/subtract from the dimension that was changed, if it doesn't violate bounds.
        # Or distribute among all dimensions that can absorb it.
        # For simplicity, try to add to the changed dimension first.
        potential_new_weight_changed_dim = dims[changed_idx]['weight'] + error
        if 0 <= potential_new_weight_changed_dim <= 100:
            dims[changed_idx]['weight'] = potential_new_weight_changed_dim
        else:
            # If that fails, try to distribute among others that can take the change
            # This can get complicated. For now, let's add to the first one that can take it.
            for i, dim in enumerate(dims):
                if error > 0 and dim['weight'] < 100:
                    dim['weight'] += error
                    break
                elif error < 0 and dim['weight'] > 0:
                    dim['weight'] += error # error is negative
                    break
            # Re-check sum one last time and if still off, it's a minor rounding issue typically ignorable for display
            # or log a warning. The impact of +/- 1 on 3-5 items is often minimal.
    
    # Ensure all weights are integers as final step for display
    for dim in dims:
        dim['weight'] = int(round(dim.get('weight',0)))

    st.session_state[editable_dims_key] = dims

def handle_document_upload(uploaded_file, document_parser, prompt_generator):
    """Handle document upload and parsing."""
    if uploaded_file is None or not document_parser or not prompt_generator:
        return
    
    # Parse the document
    with st.spinner("Parsing document..."):
        success, message, parsed_text = document_parser.parse_document(uploaded_file)
        
        if success and parsed_text:
            set_tab_state("tab0", "parsed_text", parsed_text)
            
            # Generate prompt from parsed text
            with st.spinner("Generating search prompt with DeepSeek LLM..."):
                prompt_data = prompt_generator.generate_search_prompt(parsed_text)
                set_tab_state("tab0", "prompt_data", prompt_data)
                generated_prompt = prompt_data.get("prompt", "")
                set_tab_state("tab0", "generated_prompt", generated_prompt)
                
                # Don't automatically execute the search - user should edit the prompt first
                # Still set the query, but don't mark it as executed yet
                set_tab_state("tab0", "query", generated_prompt)
                set_tab_state("tab0", "query_executed", False)
                
                # Reset states for dimension editing for the new JD
                set_tab_state("tab0", "editable_job_dimensions", None)
                set_tab_state("tab0", "jd_weights_editor_initialized", False)
                set_tab_state("tab0", "jd_weights_confirmed", False)
                
                # Success message
                st.success("Job description parsed and prompt generated successfully! Please review and edit the summary below if needed.")
        else:
            st.error(message)

def display_parsed_document():
    """Display the parsed document if available."""
    parsed_text = get_tab_state("tab0", "parsed_text")
    if parsed_text:
        with st.expander("Parsed Document", expanded=False):
            st.text_area("Job Description", parsed_text, height=300)

def display_generated_prompt():
    """Display the generated prompt and allow user to edit it before extraction."""
    prompt_data = get_tab_state("tab0", "prompt_data")
    generated_prompt = get_tab_state("tab0", "generated_prompt")
    query_executed = get_tab_state("tab0", "query_executed")
    
    if prompt_data and generated_prompt:
        # Display generated prompt with AI prefix and increased height
        st.subheader("AI Generated Search Prompt")
        st.markdown("You can edit this prompt to add or remove details before extracting metadata filters.")
        
        # Create an editable text area with the generated prompt
        edited_prompt = st.text_area(
            "Edit Prompt for Semantic Search",
            value=generated_prompt,
            height=250,
            key="edited_prompt_text"
        )
        
        # Only show the apply button if the query hasn't been executed yet
        # or if the prompt has been edited
        if not query_executed or edited_prompt != generated_prompt:
            if st.button("Apply Edits and Extract Filters", type="primary", use_container_width=True):
                # Update the prompt in session state with edited version
                set_tab_state("tab0", "generated_prompt", edited_prompt)
                # Use this as the query for search
                set_tab_state("tab0", "query", edited_prompt)
                # Mark query as ready to execute
                set_tab_state("tab0", "query_executed", True)
                
                # Show success message
                st.success("Edits applied! Proceeding to metadata extraction...")
                # Force a rerun to show the extraction UI
                st.rerun()

def process_job_description_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator, profile_retriever, profile_evaluator):
    """Process the job description query and display results."""
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
    
    # Fixed model choice for job description tab
    model_choice = "cohere"
    
    # Always extract metadata filters
    metadata_filter = None
    extracted_filters = None
    
    # STEP 1: Extract metadata filters and allow user to edit them
    with st.status("Step 1: Processing Job Description for Filters...", expanded=True):
        try:
            st.write("Extracting metadata filters from the job description...")
            filter_result = filter_extractor.process_query(query, strict_mode=False)
            metadata_filter = filter_result["pinecone_filter"]
            extracted_filters = filter_result["extracted_filters"]
            
            st.write("Please review and confirm the extracted filters below.")
            # Pass extracted filters to the filter editor - removed tab_id as it's not an accepted argument
            modified_filters = render_filter_editor(extracted_filters)
            
            if modified_filters is None:
                st.info("Please review and adjust filters above, then click 'Confirm Filters & Proceed' to continue.")
                return # Stop if filters not confirmed
            
            # User has confirmed, build new Pinecone filter with modified filters
            metadata_filter = filter_extractor.build_pinecone_filter(modified_filters, strict_mode=False)
            st.success("Filters confirmed!")
                
        except Exception as e:
            st.error(f"Error extracting metadata filters: {str(e)}")
            logger.error(f"Error during filter extraction: {str(e)}")
            return
    add_section_separator()

    # STEP 2: Extract Job Dimensions and Allow Weight Customization
    jd_weights_editor_initialized = get_tab_state("tab0", "jd_weights_editor_initialized")
    jd_weights_confirmed = get_tab_state("tab0", "jd_weights_confirmed")
    editable_job_dimensions = get_tab_state("tab0", "editable_job_dimensions")

    if not jd_weights_editor_initialized:
        with st.spinner("Extracting key evaluation dimensions from Job Description..."):
            try:
                # raw_jd for dimension extraction context, summarized_jd for primary content
                raw_jd_for_context = get_tab_state("tab0", "parsed_text")
                summarized_jd_for_extraction = query # This is the (potentially edited) summary
                
                # Call extract_job_dimensions and get the result
                extracted_dimensions_data = profile_evaluator.extract_job_dimensions(raw_jd_for_context, summarized_jd_for_extraction)
                
                # Explicitly set the job_dimensions attribute on the profile_evaluator instance
                # This ensures the instance holds the latest extracted dimensions for consistency
                # and for any internal uses within the evaluator before weights are confirmed.
                profile_evaluator.job_dimensions = extracted_dimensions_data
                
                # Now get initial_dimensions from the result we obtained
                initial_dimensions = extracted_dimensions_data.get("dimensions", [])
                
                if not initial_dimensions:
                    st.error("Could not extract evaluation dimensions. Please check the job description or try again.")
                    logger.error("Failed to extract initial job dimensions from returned data.")
                    return

                # Initialize weights to be integers and sum to 100 if not already
                current_total_weight = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if current_total_weight == 0 and initial_dimensions:
                    equal_weight = round(100 / len(initial_dimensions))
                    for dim in initial_dimensions: dim['weight'] = equal_weight
                # Normalize to 100
                current_total_weight = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if current_total_weight != 100 and current_total_weight != 0 and initial_dimensions:
                    for dim in initial_dimensions: dim['weight'] = round((dim.get('weight',0) / current_total_weight) * 100)
                # Final adjustment for sum to 100
                final_sum_check = sum(dim.get('weight', 0) for dim in initial_dimensions)
                if final_sum_check != 100 and initial_dimensions:
                    diff = 100 - final_sum_check
                    initial_dimensions[0]['weight'] = initial_dimensions[0].get('weight',0) + diff
                for dim in initial_dimensions: dim['weight'] = int(dim.get('weight',0))

                set_tab_state("tab0", "editable_job_dimensions", initial_dimensions)
                set_tab_state("tab0", "jd_weights_editor_initialized", True)
                st.rerun() # Rerun to display the editor with initialized dimensions
            except Exception as e:
                st.error(f"Error extracting job dimensions: {str(e)}")
                logger.error(f"Error during job dimension extraction: {str(e)}")
                return

    if not jd_weights_confirmed and jd_weights_editor_initialized and editable_job_dimensions:
        with st.container():
            st.subheader("Step 2: Customize Evaluation Weights")
            st.markdown("Review and adjust the weights for each evaluation dimension. The total must sum to 100%.")
            
            total_current_weight = 0
            for dim in editable_job_dimensions:
                dim_id = dim.get('id', 'unknown_id')
                dim_name = dim.get('name', 'Unknown Dimension')
                dim_weight = int(dim.get('weight', 0)) # Ensure integer for number_input
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
                        key=get_tab_state_key("tab0", f"dim_weight_{dim_id}"),
                        on_change=adjust_job_dimension_weights,
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

            if st.button("Confirm Weights and Proceed to Evaluation", type="primary", use_container_width=True, key=get_tab_state_key("tab0", "confirm_weights_button")):
                # Final check and ensure profile_evaluator gets updated dimensions
                final_dims_to_use = get_tab_state("tab0", "editable_job_dimensions")
                if sum(d.get('weight',0) for d in final_dims_to_use) != 100:
                    st.error("Cannot proceed. Weights must sum to 100%.")
                else:
                    profile_evaluator.job_dimensions = {"dimensions": final_dims_to_use}
                    set_tab_state("tab0", "jd_weights_confirmed", True)
                    logger.info(f"Job dimension weights confirmed by user: {final_dims_to_use}")
                    st.rerun()
            return # Stop further processing until weights are confirmed
    
    # Proceed with search and evaluation only if filters and weights are confirmed
    if not (get_tab_state("tab0", "query_executed") and jd_weights_confirmed):
         # This case should be handled by returns above, but as a safeguard.
         # query_executed might be true from prompt editing, but weights not confirmed yet.
         if not jd_weights_confirmed and get_tab_state("tab0", "jd_weights_editor_initialized"):
            # If editor is initialized but weights not confirmed, we are in the editing step.
            # The return above inside the editing UI block should catch this.
            logger.debug("Waiting for weight confirmation.")
         else:
            # This means the prompt wasn't even confirmed, or some other state issue.
            logger.debug("Query or weights not confirmed. Halting before search.")
         return

    # Display the confirmed dimensions as a reference
    if jd_weights_confirmed and editable_job_dimensions:
        display_reference_dimensions(editable_job_dimensions)
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
            raw_jd = get_tab_state("tab0", "parsed_text")
            summarized_jd = get_tab_state("tab0", "generated_prompt")
            
            # Use the passed-in profile_evaluator if available
            if profile_evaluator:
                evaluation_results = profile_evaluator.evaluate_profiles(
                    processed_profiles=processed_profiles,
                    raw_job_description=raw_jd,
                    summarized_job_description=summarized_jd,
                    batch_size=getattr(profile_evaluator, 'batch_size', 6)
                )
            else:
                # Fall back to creating a new one if not available
                logger.info("Creating profile evaluator instance since none was passed")
                profile_evaluator = IndividualProfileEvaluator()
                evaluation_results = profile_evaluator.evaluate_profiles(
                    processed_profiles=processed_profiles,
                    raw_job_description=raw_jd,
                    summarized_job_description=summarized_jd
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
                # Use the edited/summarized JD here as well
                llm_ranking_results = llm_ranker.rank_profiles_job_description(
                    processed_profiles=processed_profiles,
                    raw_job_description=raw_jd,
                    summarized_job_description=summarized_jd
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

def render_job_description_tab(
    document_parser,
    prompt_generator,
    filter_extractor,
    embedders,
    retrieve_documents,
    cohere_reranker,
    profile_aggregator,
    profile_retriever,
    profile_evaluator
):
    """Render the job description tab content."""
    # Initialize tab state if not already initialized
    initialize_job_description_state()
    
    # Tab header
    st.header("Upload Job Description")
    st.markdown("""
    Upload a job description document (PDF, DOC, DOCX) to automatically extract requirements and generate a prompt for semantic candidate search.
    """)
    
    # File uploader for job description document
    uploaded_file = st.file_uploader("Upload Job Description", type=["pdf", "doc", "docx"])
    
    # Process uploaded file button
    if uploaded_file is not None and document_parser and prompt_generator:
        if st.button("Parse Document", type="primary", use_container_width=True, key="parse_doc_button"):
            handle_document_upload(uploaded_file, document_parser, prompt_generator)
    
    # Display parsed document if available
    display_parsed_document()
    
    # Display generated prompt data if available and allow for editing
    display_generated_prompt()
    
    # Get sidebar configuration without rendering UI elements
    settings = create_sidebar_configuration("tab0")
    
    # Only execute the search if the tab0 query has been submitted
    if get_tab_state("tab0", "query_executed"):
        query = get_tab_state("tab0", "query")
        process_job_description_query(
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
