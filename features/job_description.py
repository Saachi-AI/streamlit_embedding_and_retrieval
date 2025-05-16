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
    display_ranked_candidates
)
from core.state_management import initialize_tab_state, get_tab_state, set_tab_state
from llm_profile_ranking import LLMProfileRanker
from profile_evaluator import IndividualProfileEvaluator
from profile_rank_processor import ProfileRankProcessor
from core.filter_editor_components import render_filter_editor

# Configure logging
logger = logging.getLogger(__name__)

def initialize_job_description_state():
    """Initialize job description tab-specific state variables."""
    # Use hardcoded defaults instead of environment variables
    defaults = {
        "semantic_top_k": 15,
        "rerank_top_k": 10,
        "enable_metadata_filtering": True,
        "query_executed": False,
        "parsed_text": None,
        "generated_prompt": None,
        "prompt_data": None,
        "query": None
    }
    initialize_tab_state("tab0", defaults)

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
    
    # Get values directly from session state (highest priority)
    semantic_top_k = st.session_state.get("tab0_semantic_top_k", 15)  # Use our new default value
    rerank_top_k = st.session_state.get("tab0_rerank_top_k", 10)  # Use our new default value
    
    # Fixed model choice for job description tab
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
