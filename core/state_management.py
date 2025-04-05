import streamlit as st

def initialize_tab_state(tab_name, defaults):
    """
    Initialize tab-specific state variables with default values.
    
    Args:
        tab_name: String prefix for the state variables
        defaults: Dictionary of {key: default_value} pairs
    """
    for key, default_value in defaults.items():
        state_key = f"{tab_name}_{key}"
        # Only set if not already in session state (preserves UI settings)
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

def get_tab_state(tab_name, key):
    """
    Get a tab-specific state variable.
    
    Args:
        tab_name: String prefix for the state variable
        key: Key name within the tab's state
        
    Returns:
        Value of the state variable or None if not found
    """
    state_key = f"{tab_name}_{key}"
    return st.session_state.get(state_key)

def set_tab_state(tab_name, key, value):
    """
    Set a tab-specific state variable.
    
    Args:
        tab_name: String prefix for the state variable
        key: Key name within the tab's state
        value: Value to set
    """
    state_key = f"{tab_name}_{key}"
    st.session_state[state_key] = value

def get_or_create_key(key, default_value=None):
    """
    Get a session state key or create it with a default value if it doesn't exist.
    
    Args:
        key: The key to get or create
        default_value: Default value if key doesn't exist
        
    Returns:
        The value of the key
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]
