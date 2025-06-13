import streamlit as st
import os

def update_profile_evaluator_settings(profile_evaluator, search_params):
    """
    Update profile evaluator settings based on search parameters.
    
    Args:
        profile_evaluator: The profile evaluator instance
        search_params: Dictionary containing search parameters
    
    Returns:
        bool: True if successfully updated, False if API key missing
    """
    if not profile_evaluator:
        return False
        
    # Update batch size
    if hasattr(profile_evaluator, 'batch_size'):
        profile_evaluator.batch_size = search_params["batch_size"]
    
    # Update LLM choice and API key if needed
    current_llm = getattr(profile_evaluator, 'llm_choice', 'gemini')
    new_llm = search_params["llm_choice"]
    
    if current_llm != new_llm:
        # LLM choice has changed, need to update API key
        if new_llm == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("GOOGLE_API_KEY not found in environment variables")
                st.warning("Please add your Google API key to your .env file or Streamlit secrets")
                return False
        else:  # grok
            api_key = os.environ.get("XAI_API_KEY")
            if not api_key:
                st.error("XAI_API_KEY not found in environment variables") 
                st.warning("Please add your X AI API key to your .env file or Streamlit secrets")
                return False
        
        # Update the evaluator
        if hasattr(profile_evaluator, 'llm_choice'):
            profile_evaluator.llm_choice = new_llm
        if hasattr(profile_evaluator, 'api_key'):
            profile_evaluator.api_key = api_key
        
        # Show info about the change
        st.info(f"LLM switched to: {new_llm.capitalize()}")
    
    return True

def get_api_key_for_llm(llm_choice):
    """
    Get the appropriate API key for the specified LLM.
    
    Args:
        llm_choice: Either 'gemini' or 'grok'
    
    Returns:
        str: API key or None if not found
    """
    if llm_choice == "gemini":
        return os.environ.get("GOOGLE_API_KEY")
    else:  # grok
        return os.environ.get("XAI_API_KEY")
        
def validate_llm_setup(llm_choice):
    """
    Validate that the required API key is available for the specified LLM.
    
    Args:
        llm_choice: Either 'gemini' or 'grok'
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    api_key = get_api_key_for_llm(llm_choice)
    return api_key is not None and api_key.strip() != "" 