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
                
                # Automatically set the query and trigger search
                set_tab_state("tab0", "query", generated_prompt)
                set_tab_state("tab0", "query_executed", True)
                
                # Success message
                st.success("Job description parsed and prompt generated successfully!")
        else:
            st.error(message)

def display_parsed_document():
    """Display the parsed document if available."""
    parsed_text = get_tab_state("tab0", "parsed_text")
    if parsed_text:
        with st.expander("Parsed Document", expanded=False):
            st.text_area("Job Description", parsed_text, height=300)

def display_generated_prompt():
    """Display the generated prompt and prompt data if available."""
    prompt_data = get_tab_state("tab0", "prompt_data")
    generated_prompt = get_tab_state("tab0", "generated_prompt")
    
    if prompt_data and generated_prompt:
        # Display generated prompt with AI prefix and increased height
        st.subheader("AI Generated Search Prompt")
        st.text_area("Prompt for Semantic Search", generated_prompt, height=250)

def process_job_description_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator, profile_retriever):
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
            
            # Display the extracted filters
            # if extracted_filters:
            #     st.subheader("Extracted Filters")
            #     st.json(extracted_filters)
            
            # Show filter editor and wait for user confirmation
            
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
                display_detailed_results(results, tab_prefix="tab0_")
                
                # Add spacing between sections
                add_section_separator()
                
                # Perform reranking - use session state value directly
                with st.spinner("Reranking results with Cohere..."):
                    # Pass rerank_top_k directly
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
                    display_reranked_detailed_results(reranked_results, tab_prefix="tab0_")
                    
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
                            
                        # Store processed profiles for later use
                        st.session_state.processed_profiles = processed_profiles
                        
                    except Exception as e:
                        st.error(f"Error in profile retrieval and preprocessing: {str(e)}")
                        logger.error(f"Error in profile retrieval and preprocessing: {str(e)}")
                        return
                    
                    # Display profile-level results
                    if profile_scores:
                        # Display profile results
                        display_profile_results(profile_scores, profile_aggregator, tab_prefix="tab0_")
                        
                        # Display profile summary
                        display_profile_summary(profile_scores)
                    else:
                        st.warning("No profiles could be aggregated from the reranked results.")
                    
                    # Display profile retrieval and preprocessing results for debugging
                    display_profile_retrieval_and_preprocessing(profile_data, processed_profiles)
                    
                    # Call LLM for profile ranking
                    try:
                        raw_jd = get_tab_state("tab0", "parsed_text")
                        summarized_jd = get_tab_state("tab0", "generated_prompt")
                        
                        llm_ranker = LLMProfileRanker()
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
                            
                            # Display ranked candidates
                            add_section_separator()
                            display_ranked_candidates(processed_candidates)
                        
                        logger.info("LLM Profile Ranking completed")
                    except Exception as e:
                        logger.error(f"Error during LLM profile ranking: {str(e)}")
                
                else:
                    st.warning("Reranking failed. Displaying original results.")
                    
                    # Display fallback results
                    display_fallback_results_header(min(rerank_top_k, len(results)))
                    display_fallback_results(results, rerank_top_k, tab_prefix="tab0_")
            else:
                st.info("No results found. Try adjusting your query or filters.")
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.info(f"Make sure you've embedded documents with this model first. Run `python embedders/embed.py --model {model_choice}`")

def render_job_description_tab(
    document_parser,
    prompt_generator,
    filter_extractor,
    embedders,
    retrieve_documents,
    cohere_reranker,
    profile_aggregator,
    profile_retriever
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
    
    # Display generated prompt data if available
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
            profile_retriever
        )
