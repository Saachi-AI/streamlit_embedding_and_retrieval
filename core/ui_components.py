import streamlit as st
import json
import os

def display_retrieval_stats(total_chunks, filtered_size=None, results_count=None, metadata_filter=None, filtered_out=None):
    """Display statistics about the retrieval process."""
    if metadata_filter and filtered_size is not None:
        # Display statistics about the filtering process
        st.markdown(f"""
        <div style="background-color: #37474F; color: white; padding: 10px; border-radius: 5px; margin: 15px 0;">
            <div style="font-weight: bold; margin-bottom: 5px;">Vector Retrieval Pipeline:</div>
            <ul style="margin: 0; padding-left: 20px;">
                <li>Total corpus size: {total_chunks} vectors</li>
                <li>After metadata filtering: {filtered_size} vectors (filtered out {filtered_out if isinstance(filtered_out, str) else total_chunks - filtered_size} vectors)</li>
                <li>Retrieved results: <strong>{results_count}</strong> vectors</li>
            </ul>
            <p style="margin-top: 8px; font-size: 0.9em;">
                Metadata filters are applied first at the database level, then semantic search retrieves the most relevant matches.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the exact Pinecone filter syntax
        with st.expander("Show Pinecone Filter Syntax", expanded=False):
            st.markdown("### Pinecone Filter Query")
            st.code(json.dumps(metadata_filter, indent=2), language="json")
        
        st.subheader(f"Retrieved {results_count} chunks (filtered from {total_chunks} total vectors)")
    else:
        # Make results_count more prominent
        st.subheader(f"Retrieved {results_count} chunks (from {total_chunks} total vectors)")

def display_initial_results_summary(results):
    """Display a summary table of initial retrieval results."""
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

def display_detailed_results(results, tab_prefix=""):
    """Display detailed results for each document in tabs."""
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
                key=f"{tab_prefix}initial_result_{i}"
            )

def display_reranked_results_summary(reranked_results):
    """Display a summary table of reranked results."""
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

def display_reranked_detailed_results(reranked_results, tab_prefix=""):
    """Display detailed reranked results in tabs."""
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
                key=f"{tab_prefix}reranked_result_{i}"
            )

def display_profile_results(profile_scores, profile_aggregator, tab_prefix=""):
    """Display profile-level results."""
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
                
                # Create a simpler layout with columns instead of complex HTML
                cols = st.columns([0.2, 0.5, 0.3])
                
                # Score column with color based on threshold
                bg_color = "#689F38" if chunk_score >= profile_aggregator.threshold else "#9E9E9E" 
                cols[0].markdown(f"<div style='background-color: {bg_color}; color: white; padding: 5px 10px; border-radius: 15px; font-weight: bold; text-align: center;'>{chunk_score_percentage:.2f}%</div>", unsafe_allow_html=True)
                
                # Section info column
                cols[1].markdown(f"<div style='padding: 5px 0px;'>Chunk #{j+1} | Section: {section}</div>", unsafe_allow_html=True)
                
                # Threshold badge column (only if above threshold)
                if chunk_score >= profile_aggregator.threshold:
                    cols[2].markdown("<div style='background-color: #689F38; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; display: inline-block;'>Above threshold</div>", unsafe_allow_html=True)
                
                # Display content in a scrollable text area for better readability
                st.text_area(
                    label="",
                    value=doc.page_content[:1000] + ('...' if len(doc.page_content) > 1000 else ''),
                    height=150,
                    disabled=True,
                    key=f"{tab_prefix}profile_{profile_score.profile_id}_chunk_{j}"
                )

def display_profile_summary(profile_scores):
    """Display a summary table of profile scores."""
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

def create_tab_specific_sidebar(active_tab_index):
    """
    Create a sidebar that only shows controls for the currently active tab.
    
    Args:
        active_tab_index: The index of the currently active tab (0 for Job Description, 1 for Custom Search)
    
    Returns:
        dict: Configuration settings for the active tab
    """
    # Ensure we're using the most up-to-date active tab index from session state
    # This helps ensure consistency when the sidebar is rendered
    if "active_tab_index" in st.session_state:
        active_tab_index = st.session_state.active_tab_index
    
    # Map tab index to tab name for session state keys
    tab_name = f"tab{active_tab_index}"
    tab_title = "Upload Job Description" if active_tab_index == 0 else "Custom Search"
    
    # Add a title to the sidebar
    st.sidebar.title("Settings")
    
    # Show the fixed embedding model
    st.sidebar.markdown("**Embedding Model:** Cohere")
    
    # Get current values from session state with hardcoded defaults
    semantic_top_k_key = f"{tab_name}_semantic_top_k"
    rerank_top_k_key = f"{tab_name}_rerank_top_k"
    
    # Initialize session state values if they don't exist yet
    if semantic_top_k_key not in st.session_state:
        st.session_state[semantic_top_k_key] = 100
    if rerank_top_k_key not in st.session_state:
        st.session_state[rerank_top_k_key] = 90
    
    # Read current values from session state
    semantic_top_k = st.session_state[semantic_top_k_key]
    rerank_top_k = st.session_state[rerank_top_k_key]
    
    # Ensure rerank_top_k <= semantic_top_k
    rerank_top_k = min(rerank_top_k, semantic_top_k)
    if rerank_top_k != st.session_state[rerank_top_k_key]:
        st.session_state[rerank_top_k_key] = rerank_top_k
    
    # Add a header for the active tab
    st.sidebar.markdown(f"**Settings for {tab_title}**")
    
    # Define callback functions to update session state
    def on_semantic_change():
        # Also update rerank if needed to maintain constraint
        if st.session_state[f"{tab_name}_semantic_slider"] < st.session_state[rerank_top_k_key]:
            st.session_state[rerank_top_k_key] = st.session_state[f"{tab_name}_semantic_slider"]
        st.session_state[semantic_top_k_key] = st.session_state[f"{tab_name}_semantic_slider"]
    
    def on_rerank_change():
        st.session_state[rerank_top_k_key] = st.session_state[f"{tab_name}_rerank_slider"]
    
    # Number of retrieval chunks with different widget key and callback
    st.sidebar.slider(
        "Number of Retrieval Chunks",
        min_value=1,
        max_value=1000,
        value=semantic_top_k,
        key=f"{tab_name}_semantic_slider",  # Different key for the widget
        on_change=on_semantic_change
    )
    
    # Number of reranked chunks with different widget key and callback
    st.sidebar.slider(
        "Number of Reranked Chunks",
        min_value=1,
        max_value=semantic_top_k,  # Use current semantic_top_k as max
        value=rerank_top_k,
        key=f"{tab_name}_rerank_slider",  # Different key for the widget
        on_change=on_rerank_change
    )
    
    # Return configuration using session state values
    return {
        "semantic_top_k": st.session_state[semantic_top_k_key],
        "rerank_top_k": st.session_state[rerank_top_k_key]
    }

def create_sidebar_configuration(tab_name):
    """
    Get the sidebar configuration for a specific tab without rendering UI elements.
    This function is used by feature modules to get their configuration.
    """
    # Use session state values with hardcoded defaults instead of environment variables
    semantic_top_k = st.session_state.get(f"{tab_name}_semantic_top_k", 100)
    rerank_top_k = st.session_state.get(f"{tab_name}_rerank_top_k", 90)
    
    return {
        "semantic_top_k": semantic_top_k,
        "rerank_top_k": rerank_top_k
    }

def add_section_separator():
    """Add a visual separator between sections."""
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #555;'>", unsafe_allow_html=True)

def display_reranked_results_header(reranked_results_count):
    """Display a header for reranked results."""
    st.markdown(f"""
    <div style="background-color: #1E3A5F; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h2 style="margin: 0; font-size: 1.5em;">Top {reranked_results_count} Reranked Results</h2>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Reranked using Cohere's rerank-english-v3.0 model</p>
    </div>
    """, unsafe_allow_html=True)

def display_fallback_results_header(fallback_count):
    """Display a header for fallback results when reranking fails."""
    st.markdown(f"""
    <div style="background-color: #6D4C41; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h2 style="margin: 0; font-size: 1.5em;">Top {fallback_count} Results (Fallback)</h2>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Using original retrieval results as fallback because reranking failed</p>
    </div>
    """, unsafe_allow_html=True)

def display_fallback_results(results, rerank_top_k, tab_prefix=""):
    """Display fallback results when reranking fails."""
    # Create a summary table for fallback results
    st.markdown("<div style='margin-bottom: 15px;'>Showing top results from initial retrieval (reranking failed):</div>", unsafe_allow_html=True)
    
    # Create columns for the table header
    cols = st.columns([0.15, 0.15, 0.25, 0.45])
    cols[0].markdown("<div style='font-weight: bold;'>Rank</div>", unsafe_allow_html=True)
    cols[1].markdown("<div style='font-weight: bold;'>Score</div>", unsafe_allow_html=True)
    cols[2].markdown("<div style='font-weight: bold;'>Profile ID</div>", unsafe_allow_html=True)
    cols[3].markdown("<div style='font-weight: bold;'>Section</div>", unsafe_allow_html=True)
    
    # Display summary of each fallback result
    limited_results = results[:rerank_top_k]
    for i, (doc, score) in enumerate(limited_results):
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
    fallback_tabs = st.tabs([f"Result {i+1}" for i in range(len(limited_results))])
    
    # Display each fallback result in a tab
    for i, (tab, (doc, score)) in enumerate(zip(fallback_tabs, limited_results)):
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
                key=f"{tab_prefix}fallback_result_{i}"
            )

def display_profile_retrieval_and_preprocessing(profile_data, processed_profiles):
    """
    Display profile retrieval and preprocessing results for debugging purposes.
    
    Args:
        profile_data: Raw profile data from ProfileRetriever (dictionary)
        processed_profiles: Processed profile data from preprocess_profiles (list)
    """
    # Add some spacing before this section
    add_section_separator()
    
    # Profile Retrieval Results
    st.markdown(f"""
    <div style="background-color: #5D4037; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h2 style="margin: 0; font-size: 1.5em;">Profile Retrieval Results</h2>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Raw profile data retrieved from DynamoDB</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show profile retrieval count
    st.success(f"✅ Successfully retrieved raw data for {len(profile_data)} profiles")
    
    # Show raw profile data in an expander
    with st.expander("View Raw Profile Data", expanded=False):
        st.json(profile_data)

    # Add some spacing before this section
    add_section_separator()

    # Add some separation between sections
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # Profile Preprocessing Results
    st.markdown(f"""
    <div style="background-color: #00695C; color: white; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h2 style="margin: 0; font-size: 1.5em;">Profile Preprocessing Results</h2>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Processed profile data ready for LLM processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show processed profile count
    st.success(f"✅ Successfully processed {len(processed_profiles)} profiles")
    
    # Show processed profile data in an expander
    with st.expander("View Processed Profile Data", expanded=False):
        st.json(processed_profiles)

def display_dimension_scores(dimension_scores):
    """
    Display dimension scores with progress bars and reasoning.
    
    Args:
        dimension_scores: List of dimension score objects with name, score, reasoning, and color
    """
    if not dimension_scores:
        return
    
    st.markdown("#### 📊 Evaluation by Dimensions:")
    
    for dimension in dimension_scores:
        # Get dimension details
        name = dimension.get('name', 'Unknown Dimension')
        score = dimension.get('score', 0)
        reasoning = dimension.get('reasoning', '')
        color = dimension.get('color', 'gray')
        
        # Display dimension name
        st.markdown(f"<strong>{name}</strong>", unsafe_allow_html=True)
        
        # Display score as progress bar
        progress_html = f"""
        <div style="margin-bottom: 5px;">
            <div style="background-color: #f0f0f0; border-radius: 5px; height: 12px; width: 100%;">
                <div style="background-color: {color}; border-radius: 5px; height: 12px; width: {score}%;"></div>
            </div>
            <div style="text-align: right; font-size: 12px; color: {color}; font-weight: bold;">
                {score}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        # Display reasoning without using an expander
        if reasoning:
            st.markdown(f"""
            <details>
                <summary style="cursor: pointer; color: #4a86e8; font-size: 14px;">Show reasoning</summary>
                <div style="margin-top: 8px; margin-left: 20px; font-size: 14px; color: #555;">
                    {reasoning}
                </div>
            </details>
            """, unsafe_allow_html=True)
        
        # Add spacing between dimensions
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

def display_ranked_candidates(processed_candidates):
    """
    Display ranked candidates in expandable sections.
    
    Args:
        processed_candidates: List of processed candidate objects from ProfileRankProcessor
    """
    st.header("Here are the Top Profiles:")
    
    if not processed_candidates:
        st.info("No ranked candidates to display")
        return
    
    # Display each candidate in an expandable section
    for candidate in processed_candidates:
        # Create expander title with rank, display name
        # Check if we have the new format (dimension-based) or old format
        if 'dimension_scores' in candidate:
            # New format - use match category
            match_category = candidate.get('match_category', 'Unknown Match')
            expander_title = f"{candidate['rank_display']} — {candidate['display_name']} - {match_category}"
        else:
            # Old format - use short phrase
            expander_title = f"{candidate['rank_display']} — {candidate['display_name']} - {candidate.get('short_phrase', '')}"
        
        # Create expandable section
        with st.expander(expander_title, expanded=candidate['rank'] == 1):  # Auto-expand first result
            # Display LinkedIn enrichment info if available
            if candidate.get('linkedin_fetched_at'):
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 25px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
                         alt="LinkedIn icon"
                         style="height: 16px; width: 16px; vertical-align: middle;" /> 
                    enriched on <span style='color: #FFCC80;'>{candidate['linkedin_fetched_at']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a two-column layout for profile info and picture
            cols = st.columns([0.60, 0.30])
            
            with cols[0]:  # Left column for text information
                # Display profile header with employment details
                employment_line = f"<h4 style='margin-bottom: 0;'>👤 {candidate['display_name']}</h4>"
                
                # Build the position and company line with colored HTML
                position_html = ""
                company_html = ""
                period_html = ""
                
                if candidate.get('current_position'):
                    position_html = f"<span style='color: #81D4FA;'>{candidate['current_position']}</span>"
                
                if candidate.get('current_company'):
                    if position_html:
                        company_html = f" at <span style='color: #FFCC80;'>{candidate['current_company']}</span>"
                    else:
                        company_html = f"<span style='color: #FFCC80;'>{candidate['current_company']}</span>"
                
                # Add period if available
                if candidate.get('employment_period'):
                    period_html = f" <span style='color: #B0BEC5;'>({candidate['employment_period']})</span>"
                
                # Display employment information if we have any
                if position_html or company_html:
                    employment_html = f"{employment_line}<div style='margin-top: 3px;'>{position_html}{company_html}{period_html}</div>"
                    st.markdown(employment_html, unsafe_allow_html=True)
                else:
                    st.markdown(employment_line, unsafe_allow_html=True)
                
                # Add small spacing instead of a full line break
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                # Check if we have the new format (dimension-based) or old format
                if 'dimension_scores' in candidate:
                    # New format - dimension-based evaluation
                    
                    # Display match percentage if available
                    if 'match_percentage' in candidate:
                        # Determine color based on match percentage
                        if candidate['match_percentage'] >= 85:
                            match_color = "green"
                        elif candidate['match_percentage'] >= 70:
                            match_color = "lightgreen"
                        elif candidate['match_percentage'] >= 50:
                            match_color = "orange"
                        else:
                            match_color = "red"
                            
                        st.markdown(
                            f"#### ⭐ Overall Match: <span style='color: {match_color};'>{candidate['match_percentage']}%</span> "
                            f"(<span style='color: {match_color};'>{candidate['match_category']}</span>)",
                            unsafe_allow_html=True
                        )
                    
                    # Display match reasoning if available without using expander
                    if candidate.get('match_reasoning'):
                        st.markdown(f"""
                        <details>
                            <summary style="cursor: pointer; color: #4a86e8; font-size: 14px;">Overall Match Reasoning</summary>
                            <div style="margin-top: 8px; margin-left: 20px; font-size: 14px; color: #555;">
                                {candidate['match_reasoning']}
                            </div>
                        </details>
                        """, unsafe_allow_html=True)
                    
                    # Display dimension scores
                    display_dimension_scores(candidate.get('dimension_scores', []))
                    
                    # Display key strengths if available
                    if candidate.get('key_strengths') and len(candidate['key_strengths']) > 0:
                        st.markdown("#### 💪 Key Strengths:")
                        for i, strength in enumerate(candidate['key_strengths'], 1):
                            st.markdown(f"<div style='margin-left: 20px;'>{i}. {strength}</div>", unsafe_allow_html=True)
                        
                        # Add spacing after strengths section
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    
                    # Display key gaps if available
                    if candidate.get('key_gaps') and len(candidate['key_gaps']) > 0:
                        st.markdown("#### 🚧 Areas for Development:")
                        for i, gap in enumerate(candidate['key_gaps'], 1):
                            st.markdown(f"<div style='margin-left: 20px;'>{i}. {gap}</div>", unsafe_allow_html=True)
                        
                        # Add spacing after gaps section
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    
                    # Display overqualification warning if applicable
                    if candidate.get('is_overqualified'):
                        st.warning(f"⚠️ Overqualified: {candidate.get('overqualification_reasoning', '')}")
                else:
                    # Old format - skills-based evaluation
                    
                    # Display match score if available
                    if candidate.get('match_score'):
                        st.markdown(f"#### ⭐ Match Score: {candidate['match_score']}%")
                    
                    # Display reasons why this candidate is a good fit
                    if candidate.get('why_good_fit') and len(candidate['why_good_fit']) > 0:
                        st.markdown("#### 🤖 Why this candidate?")
                        for reason in candidate['why_good_fit']:
                            st.markdown(f"<div style='margin-left: 20px;'>{reason}</div>", unsafe_allow_html=True)
                        
                        # Add extra spacing after "Why this candidate?" section
                        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    
                    # Display Skills Match if available
                    if candidate.get('skills_match') and len(candidate['skills_match']) > 0:
                        st.markdown("#### 🧠 Skills Match for this Job:")
                        for i, skill_info in enumerate(candidate['skills_match'], 1):
                            st.markdown(
                                f"<div style='margin-left: 20px;'>{i}. {skill_info['skill']} - {skill_info['status_display']}</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Add spacing after skills match section
                        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                
                # Display Experience if available
                if candidate.get('years_of_experience') and candidate['years_of_experience'] != 0:
                    st.markdown(f"###### 💼 Experience: {candidate['years_of_experience']} years")
                
                # Display Gender if available and not "not mentioned"
                if candidate.get('gender') and candidate['gender'].lower() != "not mentioned":
                    st.markdown(f"###### 🚻 Gender: {candidate['gender']}")
                
                # Display Type if available
                if candidate.get('type') is not None:
                    type_value = candidate['type']
                    color_style = "style='color: #FFCC80;'" if type_value == "Candidate" else ""
                    st.markdown(f"###### 🤝 Type: <span {color_style}>{type_value}</span>", unsafe_allow_html=True)
                elif candidate.get('is_candidate') is not None:
                    is_candidate = candidate['is_candidate']
                    type_value = "Candidate" if is_candidate else "Unknown"
                    color_style = "style='color: #FFCC80;'" if is_candidate else ""
                    st.markdown(f"###### 🤝 Type: <span {color_style}>{type_value}</span>", unsafe_allow_html=True)
                
                # Display Languages
                if candidate.get('languages') and len(candidate['languages']) > 0:
                    st.markdown("###### 🌐 Languages spoken")
                    for language, proficiency in candidate['languages'].items():
                        st.markdown(f"<div style='margin-left: 20px;'>• <strong>{language}</strong> - {proficiency}</div>", unsafe_allow_html=True)
                    
                    # Add spacing after languages
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                
                # Display Previously Placed information
                if candidate.get('previous_placements'):
                    placements = candidate['previous_placements']
                    if len(placements) == 1:
                        # Single placement - display inline
                        placement = placements[0]
                        st.markdown(
                            f"###### 🎖️ Previously Placed: <span style='font-weight: normal;'>We placed the candidate in </span>"
                            f"<span style='color: #81D4FA;'>{placement['company_name']}</span> "
                            f"<span style='font-weight: normal;'>on </span>"
                            f"<span style='color: #B0BEC5;'>{placement['date']}</span> "
                            f"<span style='font-weight: normal;'>by </span>"
                            f"<span style='color: #FFCC80;'>{placement['person']}</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        # Multiple placements - display as list
                        st.markdown("###### 🎖️ Previously Placed:")
                        for i, placement in enumerate(placements, 1):
                            st.markdown(
                                f"<div style='margin-left: 20px;'>{i}. We placed the candidate in "
                                f"<span style='color: #81D4FA;'>{placement['company_name']}</span> on "
                                f"<span style='color: #B0BEC5;'>{placement['date']}</span> by "
                                f"<span style='color: #FFCC80;'>{placement['person']}</span></div>",
                                unsafe_allow_html=True
                            )
                        
                        # Add spacing after multiple placements list
                        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                
                # Display Contradictions or Warnings if available
                if candidate.get('contradictions'):
                    st.markdown("###### ⚠️ Contradictions or Warnings:")
                    for i, warning in enumerate(candidate['contradictions'], 1):
                        st.markdown(f"<div style='margin-left: 20px;'>{i}. {warning}</div>", unsafe_allow_html=True)
                    
                    # Add spacing after warnings
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                
                # Display Profile Created By
                if candidate.get('profile_created'):
                    st.markdown(
                        f"###### 🧑‍💼 Profile Created By : <span style='color: #FFCC80;'>{candidate['profile_created']['name']}</span> "
                        f"on <span style='color: #B0BEC5;'>{candidate['profile_created']['date']}</span>",
                        unsafe_allow_html=True
                    )
                
                # Display Last Contacted By
                if candidate.get('last_contacted'):
                    st.markdown(
                        f"###### 📞 Last Contacted By : <span style='color: #FFCC80;'>{candidate['last_contacted']['name']}</span> "
                        f"on <span style='color: #B0BEC5;'>{candidate['last_contacted']['date']}</span>",
                        unsafe_allow_html=True
                    )
                
                # Display profile ID for debugging
                st.caption(f"Profile ID: {candidate['profile_id']}")
            
            with cols[1]:  # Right column for profile picture
                if candidate.get('profile_picture_url'):
                    # Use HTML to create a rounded profile picture
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center;">
                        <img src="{candidate['profile_picture_url']}" 
                             style="border-radius: 50%; border: 3px solid #0077B5; width: 150px; height: 150px; object-fit: cover;
                             box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.3) inset;
                             transition: transform 0.3s ease, box-shadow 0.3s ease;"
                             alt="{candidate['display_name']}'s profile picture">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add spacing between picture and buttons
                    st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
                    
                    # LinkedIn button (only if available) - centered
                    if candidate.get('linkedin_url'):
                        st.markdown(
                            f"""<div style="display: flex; justify-content: center;">
                                <a href="{candidate['linkedin_url']}" target="_blank" 
                                   style="text-decoration: none; display: inline-block; width: 150px; text-align: center; 
                                   background-color: #0077B5; color: white; padding: 12px 0; 
                                   border: none; border-radius: 8px; font-weight: 500; letter-spacing: 0.5px;
                                   box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                                   transition: all 0.3s ease;">
                                   View on LinkedIn</a>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    # Add spacing between buttons
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    
                    # Tamago button - centered
                    if candidate.get('tamago_url'):
                        st.markdown(
                            f"""<div style="display: flex; justify-content: center;">
                                <a href="{candidate['tamago_url']}" target="_blank"
                                   style="text-decoration: none; display: inline-block; width: 150px; text-align: center;
                                   background-color: #FF5722; color: white; padding: 12px 0;
                                   border: none; border-radius: 8px; font-weight: 500; letter-spacing: 0.5px;
                                   box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                                   transition: all 0.3s ease;">
                                   View on Tamago</a>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
            # Add a separator between candidates
            st.markdown("<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)
