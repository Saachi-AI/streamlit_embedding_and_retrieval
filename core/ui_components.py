import streamlit as st
import json
import os

def display_retrieval_stats(total_chunks, filtered_size=None, results_count=None, metadata_filter=None, filtered_out=None):
    """
    This function was previously used to display retrieval statistics.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_initial_results_summary(results):
    """
    This function was previously used to display a summary table of initial retrieval results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_detailed_results(results, tab_prefix=""):
    """
    This function was previously used to display detailed results for each document in tabs.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_reranked_results_header(reranked_results_count):
    """
    This function was previously used to display a header for reranked results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_reranked_results_summary(reranked_results):
    """
    This function was previously used to display a summary table of reranked results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_reranked_detailed_results(reranked_results, tab_prefix=""):
    """
    This function was previously used to display detailed reranked results in tabs.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_fallback_results_header(fallback_count):
    """
    This function was previously used to display a header for fallback results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_fallback_results(results, rerank_top_k, tab_prefix=""):
    """
    This function was previously used to display fallback results when reranking fails.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_profile_results(profile_scores, profile_aggregator, tab_prefix=""):
    """
    This function was previously used to display profile-level results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

def display_profile_summary(profile_scores):
    """
    This function was previously used to display a summary of profile scores.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    """
    # Function now returns silently without displaying anything
    return

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
    # st.sidebar.markdown("**Embedding Model:** Cohere")
    
    # Get current values from session state with hardcoded defaults
    semantic_top_k_key = f"{tab_name}_semantic_top_k"
    rerank_top_k_key = f"{tab_name}_rerank_top_k"
    
    # Initialize session state values if they don't exist yet
    if semantic_top_k_key not in st.session_state:
        st.session_state[semantic_top_k_key] = 475
    if rerank_top_k_key not in st.session_state:
        st.session_state[rerank_top_k_key] = 250
    
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
    semantic_top_k = st.session_state.get(f"{tab_name}_semantic_top_k", 475)
    rerank_top_k = st.session_state.get(f"{tab_name}_rerank_top_k", 250)
    
    return {
        "semantic_top_k": semantic_top_k,
        "rerank_top_k": rerank_top_k
    }

def add_section_separator():
    """Add a visual separator between sections."""
    st.markdown("<hr style='margin: 30px 0; border: 0; border-top: 1px solid rgba(107, 114, 128, 0.3);'>", unsafe_allow_html=True)

def display_profile_retrieval_and_preprocessing(profile_data, processed_profiles):
    """
    This function was previously used to display profile retrieval and preprocessing results.
    Now returns silently to hide these technical UI elements while preserving the code flow.
    
    Args:
        profile_data: Raw profile data from ProfileRetriever (dictionary)
        processed_profiles: Processed profile data from preprocess_profiles (list)
    """
    # Function now returns silently without displaying anything
    return

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
                    type_value = "Candidate" if is_candidate else "Lead"
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

def display_reference_dimensions(dimensions, title="Reference Evaluation Weights"):
    """
    Display previously confirmed job dimensions as a read-only reference.
    
    Args:
        dimensions: List of dimension objects with name, weight, description, and key_success_factors
        title: Optional title for the reference panel (default: "Reference Evaluation Weights")
    """
    if not dimensions:
        return
    
    # Create a container with a subtle background
    with st.container():
        st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.9em;'>These weights are being used for candidate evaluation:</div>", unsafe_allow_html=True)
        
        # Display each dimension with its weight
        for dim in dimensions:
            dim_name = dim.get('name', 'Unknown Dimension')
            dim_weight = int(dim.get('weight', 0))
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{dim_name}**")
            with col2:
                st.markdown(f"<div style='text-align: right;'><strong>{dim_weight}%</strong></div>", unsafe_allow_html=True)
            
            # Make details available via expander
            with st.expander(f"Details for {dim_name}", expanded=False):
                st.markdown(f"**Description:** {dim.get('description', 'N/A')}")
                st.markdown("**Key Success Factors:**")
                for factor in dim.get('key_success_factors', []):
                    st.markdown(f"- {factor}")
