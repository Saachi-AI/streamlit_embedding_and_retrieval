import streamlit as st
import os
import json
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

def initialize_job_description_state():
    """Initialize job description tab-specific state variables."""
    defaults = {
        "semantic_top_k": int(os.getenv("SEMANTIC_TOP_K", 10)),
        "rerank_top_k": min(int(os.getenv("RERANK_TOP_K", 5)), int(os.getenv("SEMANTIC_TOP_K", 10))),
        "top_k_profiles": int(os.getenv("TOP_K_PROFILES", 5)),
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
                st.success("Job description parsed and prompt generated successfully! Searching for candidates...")
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
        # Display generated prompt
        st.subheader("Generated Search Prompt")
        st.text_area("Prompt for Semantic Search", generated_prompt, height=150)
        
        # Display all information in a single dropdown
        with st.expander("Prompt Generation Details", expanded=True):
            # Format the entire prompt_data as formatted JSON
            formatted_json = json.dumps(prompt_data, indent=2)
            st.code(formatted_json, language="json")
            
            st.markdown("### Prompt Generation Analysis")
            
            # Display metadata
            if "metadata" in prompt_data:
                st.markdown("#### Metadata Considered")
                metadata = prompt_data["metadata"]
                
                for key, value in metadata.items():
                    if key == "languages":
                        st.markdown(f"**Languages**: {json.dumps(value, indent=2)}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
            
            # If there are any extracted skills
            if "extracted_skills" in prompt_data and prompt_data["extracted_skills"]:
                st.markdown("#### Skills Extracted")
                st.json(prompt_data["extracted_skills"])
            
            # If there is experience information
            if "extracted_experience" in prompt_data and prompt_data["extracted_experience"]:
                st.markdown("#### Experience Requirements")
                st.json(prompt_data["extracted_experience"])

def process_job_description_query(query, settings, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator):
    """Process the job description query and display results."""
    if not query:
        return
    
    # Unpack settings
    semantic_top_k = settings["semantic_top_k"] 
    rerank_top_k = settings["rerank_top_k"]
    top_k_profiles = settings["top_k_profiles"]
    enable_metadata_filtering = settings["enable_metadata_filtering"]
    
    # Fixed model choice for job description tab
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
                display_detailed_results(results, tab_prefix="tab0_")
                
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
                    display_reranked_detailed_results(reranked_results, tab_prefix="tab0_")
                    
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
                        display_profile_results(profile_scores, profile_aggregator, tab_prefix="tab0_")
                        
                        # Display profile summary
                        display_profile_summary(profile_scores)
                    else:
                        st.warning("No profiles could be aggregated from the reranked results.")
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

def render_job_description_tab(document_parser, prompt_generator, filter_extractor, embedders, retrieve_documents, cohere_reranker, profile_aggregator):
    """Render the job description tab content."""
    # Initialize tab state if not already initialized
    initialize_job_description_state()
    
    # Set the active tab in session state
    if "active_tab_index" in st.session_state:
        st.session_state.active_tab_index = 0
    
    # Tab header
    st.header("Upload Job Description")
    st.markdown("""
    Upload a job description document (PDF, DOC, DOCX) to automatically extract requirements and generate a prompt for semantic candidate search.
    """)
    
    # File uploader for job description document
    uploaded_file = st.file_uploader("Upload Job Description", type=["pdf", "doc", "docx"])
    
    # Process uploaded file button
    if uploaded_file is not None and document_parser and prompt_generator:
        if st.button("Parse Document", key="parse_doc_button"):
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
            profile_aggregator
        )
