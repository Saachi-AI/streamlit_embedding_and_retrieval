import streamlit as st

def render_search_parameters(tab_id="", key_prefix="search_params"):
    """
    Render advanced search parameters in an expandable section.
    
    Args:
        tab_id: Tab identifier for unique keys
        key_prefix: Prefix for session state keys
    
    Returns:
        dict: Dictionary containing the current parameter values
    """
    # Initialize default values
    defaults = {
        "top_k_profiles": 30,
        "threshold": 0.70,  # Changed to 0.70 to match PROFILE_SCORE_THRESHOLD default
        "llm_choice": "gemini",
        "batch_size": 20
    }
    
    # Create unique keys for this tab
    if tab_id:
        key_suffix = f"_{tab_id}"
    else:
        key_suffix = ""
    
    # Initialize session state with defaults if not present
    # Only initialize if the key doesn't exist to avoid conflicts with widget defaults
    for param, default_value in defaults.items():
        state_key = f"{key_prefix}_{param}{key_suffix}"
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value
    
    # Create expandable section for search parameters
    with st.expander("⚙️ Search Parameters", expanded=False):
        st.markdown("**Adjust these settings to fine-tune your search:**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Define callback function for top_k number input
            def on_top_k_change():
                st.session_state[f"{key_prefix}_top_k_profiles{key_suffix}"] = st.session_state[f"{key_prefix}_top_k_input{key_suffix}"]
            
            # Top K Profiles
            top_k = st.number_input(
                "Top Profiles",
                min_value=5,
                max_value=50,
                value=st.session_state[f"{key_prefix}_top_k_profiles{key_suffix}"],
                step=1,
                help="Maximum number of profiles to retrieve and evaluate",
                key=f"{key_prefix}_top_k_input{key_suffix}",
                on_change=on_top_k_change
            )
            
            # Define callback function for threshold slider
            def on_threshold_change():
                st.session_state[f"{key_prefix}_threshold{key_suffix}"] = st.session_state[f"{key_prefix}_threshold_input{key_suffix}"]
            
            # Threshold
            threshold = st.slider(
                "Score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[f"{key_prefix}_threshold{key_suffix}"],
                step=0.05,
                help="Minimum reranking score for chunks to contribute bonus points to profile scoring. Higher values = more selective bonus scoring, lower values = more generous bonus scoring.",
                key=f"{key_prefix}_threshold_input{key_suffix}",
                on_change=on_threshold_change
            )
        
        with col2:
            # LLM Selection - Use separate widget key to avoid session state conflicts
            current_llm = st.session_state[f"{key_prefix}_llm_choice{key_suffix}"]
            llm_index = 0 if current_llm == "gemini" else 1
            
            # Define callback for LLM choice
            def on_llm_change():
                st.session_state[f"{key_prefix}_llm_choice{key_suffix}"] = st.session_state[f"{key_prefix}_llm_widget{key_suffix}"]
            
            llm_choice = st.selectbox(
                "LLM Model",
                options=["gemini", "grok"],
                index=llm_index,
                help="Choose the Large Language Model for profile evaluation",
                key=f"{key_prefix}_llm_widget{key_suffix}",  # Use separate widget key
                on_change=on_llm_change
            )
            
            # Batch Size - Use separate widget key to avoid session state conflicts
            batch_options = [5, 10, 15, 20]
            current_batch = st.session_state[f"{key_prefix}_batch_size{key_suffix}"]
            
            # Ensure current_batch is in the options, otherwise default to 20
            if current_batch not in batch_options:
                current_batch = 20
                st.session_state[f"{key_prefix}_batch_size{key_suffix}"] = current_batch
            
            batch_index = batch_options.index(current_batch)
            
            # Define callback for batch size
            def on_batch_change():
                st.session_state[f"{key_prefix}_batch_size{key_suffix}"] = st.session_state[f"{key_prefix}_batch_widget{key_suffix}"]
            
            batch_size = st.selectbox(
                "Batch Size",
                options=batch_options,
                index=batch_index,
                help="Number of profiles to process in each batch",
                key=f"{key_prefix}_batch_widget{key_suffix}",  # Use separate widget key
                on_change=on_batch_change
            )
        
        # Reset to defaults button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Reset to Defaults", key=f"{key_prefix}_reset{key_suffix}", use_container_width=True):
                # Reset all values to defaults - only update the data keys, not widget keys
                for param, default_value in defaults.items():
                    state_key = f"{key_prefix}_{param}{key_suffix}"
                    st.session_state[state_key] = default_value
                
                # Rerun to refresh the widgets with new values
                st.rerun()
    
    # Return current parameter values (no need to manually update session state since we're using direct keys)
    return {
        "top_k_profiles": top_k,
        "threshold": threshold,
        "llm_choice": llm_choice,
        "batch_size": batch_size
    }

def get_search_parameters(tab_id="", key_prefix="search_params"):
    """
    Get current search parameter values from session state.
    
    Args:
        tab_id: Tab identifier for unique keys
        key_prefix: Prefix for session state keys
    
    Returns:
        dict: Dictionary containing the current parameter values
    """
    # Default values
    defaults = {
        "top_k_profiles": 30,
        "threshold": 0.70,  # Changed to 0.70 to match PROFILE_SCORE_THRESHOLD default
        "llm_choice": "gemini",
        "batch_size": 20
    }
    
    # Create unique keys for this tab
    if tab_id:
        key_suffix = f"_{tab_id}"
    else:
        key_suffix = ""
    
    # Get values from session state, fallback to defaults
    return {
        "top_k_profiles": st.session_state.get(f"{key_prefix}_top_k_profiles{key_suffix}", defaults["top_k_profiles"]),
        "threshold": st.session_state.get(f"{key_prefix}_threshold{key_suffix}", defaults["threshold"]),
        "llm_choice": st.session_state.get(f"{key_prefix}_llm_choice{key_suffix}", defaults["llm_choice"]),
        "batch_size": st.session_state.get(f"{key_prefix}_batch_size{key_suffix}", defaults["batch_size"])
    } 