import streamlit as st
import json
import os
from profile_rank_processor import ProfileRankProcessor
from core.ui_components import display_ranked_candidates as original_display_ranked_candidates

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return None

def display_ranked_candidates_for_testing(processed_candidates):
    """
    Modified version of display_ranked_candidates that respects the auto_expand property.
    This function wraps the original function from core/ui_components.py to maintain UI consistency
    while adding test-specific behavior.
    """
    # First, adjust the candidates for testing (adding auto_expand property)
    for candidate in processed_candidates:
        # If auto_expand is already set, respect it
        # Otherwise, default to rank-based logic (auto-expand for rank 1)
        if 'auto_expand' in candidate:
            # Use existing auto_expand value
            should_expand = candidate['auto_expand']
            # Override the rank-based expansion with our test-specific value
            candidate['rank'] = 1 if should_expand else 2  # Force rank 1 if should expand
    
    # Call the original function with our modified candidates
    original_display_ranked_candidates(processed_candidates)

def main():
    st.set_page_config(
        page_title="UI Testing Page",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("UI Testing Page")
    st.markdown("""
    This page allows you to test UI components with mock data from JSON files.
    """)
    
    # Settings sidebar
    with st.sidebar:
        st.header("Test Settings")
        
        # Add a reload button
        if st.button("🔄 Reload Page", use_container_width=True):
            st.rerun()
        
        st.divider()
        
        # Toggle sections
        show_stats = st.checkbox("Show Statistics", value=True)
        auto_expand = st.checkbox("Auto-expand First Profile", value=True)
        show_debug = st.checkbox("Show Debug Section", value=False)
        
        st.divider()
        
        # Profile selection
        st.subheader("Filter Profiles")
        profile_filter = st.text_input("Profile ID Filter (leave empty for all)")
    
    # File paths (with option to choose different files)
    col1, col2 = st.columns(2)
    with col1:
        llm_results_path = st.text_input("LLM Ranking Results File", value="test_llm_output.json")
    with col2:
        profile_data_path = st.text_input("Profile Data File", value="profile_retrieved_output.json")
    
    # Load JSON files
    llm_ranking_results = load_json_file(llm_results_path)
    profile_data = load_json_file(profile_data_path)
    
    if llm_ranking_results is None or profile_data is None:
        st.warning("Please ensure both JSON files are available and valid.")
        return
    
    # Clean up data if needed (remove any metadata fields)
    if "_disclaimer" in llm_ranking_results:
        del llm_ranking_results["_disclaimer"]
    
    # Filter profiles if filter is specified
    if profile_filter:
        filtered_llm_results = {}
        for profile_id, data in llm_ranking_results.items():
            if profile_filter in profile_id:
                filtered_llm_results[profile_id] = data
        
        if not filtered_llm_results:
            st.warning(f"No profiles found matching filter: {profile_filter}")
            return
        
        llm_ranking_results = filtered_llm_results
    
    # Process data with existing processor
    processor = ProfileRankProcessor()
    candidates = processor.process_ranked_profiles(llm_ranking_results, profile_data)
    
    # Override auto-expand setting based on checkbox
    for candidate in candidates:
        if not auto_expand:
            candidate['auto_expand'] = False
        else:
            candidate['auto_expand'] = (candidate['rank'] == 1)
    
    # Display statistics
    if show_stats:
        st.subheader("Test Data Statistics")
        st.info(f"Loaded {len(llm_ranking_results)} ranked profiles and {len(profile_data)} profile data entries")
    
    # Display the candidates using our testing-specific function that wraps the original component
    display_ranked_candidates_for_testing(candidates)
    
    # Add debugging section if requested
    if show_debug:
        with st.expander("Debug Raw Data", expanded=True):
            st.subheader("LLM Ranking Results")
            st.json(llm_ranking_results)
            
            st.subheader("First Profile Data Entry")
            if profile_data and len(profile_data) > 0:
                first_key = next(iter(profile_data))
                st.json({first_key: profile_data[first_key]})

if __name__ == "__main__":
    main()
