import streamlit as st

def render_filter_editor(extracted_filters):
    """
    Render interactive filter editor and return modified filters.
    
    Args:
        extracted_filters: Dictionary of filters extracted by LLM
        
    Returns:
        Dictionary of modified filters after user interaction or None if not confirmed
    """
    # Initialize session state for filter editor if not exists
    if "filter_editor_initialized" not in st.session_state:
        st.session_state.filter_editor_initialized = True
        st.session_state.filters_confirmed = False
        
        # Extract initial values from extracted_filters
        yoe = extracted_filters.get("min_years_experience", extracted_filters.get("years_of_experience", 0))
        detected_languages = extracted_filters.get("languages", [])
        
        # Initialize session state with extracted values
        st.session_state.gender = "No Preference"
        st.session_state.years_experience = int(yoe) if yoe is not None else 0
        st.session_state.original_jd_yoe = int(yoe) if yoe is not None else 0  # Store original JD value separately
        st.session_state.profile_type = "No Preference"
        
        # Process languages into a consistent format for the UI
        st.session_state.languages = {}
        st.session_state.original_languages = {}  # Store original languages from JD
        
        # Handle case where detected_languages is a list of dictionaries
        if isinstance(detected_languages, list):
            for lang_obj in detected_languages:
                if isinstance(lang_obj, dict) and "Language" in lang_obj and "Proficiency Level" in lang_obj:
                    st.session_state.languages[lang_obj["Language"]] = lang_obj["Proficiency Level"]
                    st.session_state.original_languages[lang_obj["Language"]] = lang_obj["Proficiency Level"]
        # Handle case where detected_languages is already a dictionary
        elif isinstance(detected_languages, dict):
            st.session_state.languages = detected_languages.copy()
            st.session_state.original_languages = detected_languages.copy()
    
    # Display header and info
    st.markdown("## Search Filters")
    st.markdown("Help us fine-tune your search to get the most accurate candidate matches")
    
    
    # Render individual filter components
    render_gender_filter()
    render_yoe_filter()
    render_type_filter()
    render_language_filters()
    
    # Add a confirmation button
    if st.button("Confirm & Proceed", type="primary", use_container_width=True):
        st.session_state.filters_confirmed = True
    
    # If filters are confirmed, construct and return the modified filters
    if st.session_state.filters_confirmed:
        # Construct modified filters dictionary
        modified_filters = {
            "gender": st.session_state.gender if st.session_state.gender != "No Preference" else None,
            "years_of_experience": st.session_state.years_experience,
            "min_years_experience": st.session_state.min_years_experience,
            "max_years_experience": st.session_state.max_years_experience,
            "type": st.session_state.profile_type if st.session_state.profile_type != "No Preference" else None,
            "languages": st.session_state.languages if st.session_state.languages else None
        }
        
        # Remove None values
        modified_filters = {k: v for k, v in modified_filters.items() if v is not None}
        
        return modified_filters
    
    # If not confirmed, return None to indicate waiting
    return None

def render_gender_filter():
    """Render gender preference filter."""
    st.subheader("Gender")
    st.caption("Select a gender if the role specifically requires one")
    
    st.session_state.gender = st.selectbox(
        label="Gender",
        options=["No Preference", "Male", "Female"],
        index=["No Preference", "Male", "Female"].index(st.session_state.gender),
        key="gender_selectbox"
    )

# Callback function for slider changes
def update_yoe_values():
    # Get the current slider values
    min_val, max_val = st.session_state.yoe_range_slider
    
    # Update session state values
    st.session_state.min_years_experience = min_val
    st.session_state.max_years_experience = max_val
    
    # For backward compatibility, keep the years_experience field (set to min value)
    st.session_state.years_experience = min_val

def render_yoe_filter():
    """Render years of experience filter."""
    st.subheader("Years of Experience")
    
    # Initialize min_years_experience and max_years_experience if not already in session state
    if "min_years_experience" not in st.session_state:
        st.session_state.min_years_experience = st.session_state.years_experience
        st.session_state.max_years_experience = min(st.session_state.years_experience + 5, 50)
    
    # Make sure original_jd_yoe is set
    if "original_jd_yoe" not in st.session_state:
        st.session_state.original_jd_yoe = st.session_state.years_experience
    
    # Display detected YOE from JD (using the original value, not the current slider value)
    st.markdown(f"""<div style="text-shadow: 0 0 5px rgba(255,255,255,0.3); font-size: 1.05em;">
                <span style="color: white;">✨ Detected from JD: </span>
                <span style="color: #FFCC80;">{st.session_state.original_jd_yoe} YOE</span>
                <span style="color: white;">. Adjust if needed</span>
                </div>""", 
                unsafe_allow_html=True)
    
    # Caption for the range slider
    st.caption("Select the range of years of experience for candidate search.")
    
    # Use a range slider with two handles and on_change callback
    st.slider(
        "Experience Range (years)",
        min_value=0,
        max_value=50,
        value=(st.session_state.min_years_experience, st.session_state.max_years_experience),
        step=1,
        key="yoe_range_slider",
        on_change=update_yoe_values
    )
    
    # Display the selected range in a user-friendly format
    st.markdown(f"""<div style="margin-top: 10px; margin-bottom: 15px;">
                <span>Greater than </span>
                <strong>{st.session_state.min_years_experience} years</strong>
                <span>, Less than </span>
                <strong>{st.session_state.max_years_experience} years</strong>
                </div>""",
                unsafe_allow_html=True)

def render_type_filter():
    """Render profile type filter."""
    st.subheader("Profile Type")
    st.caption("Candidates are actively seeking positions. Leads are potential matches not actively job hunting.")
    
    st.session_state.profile_type = st.selectbox(
        label="Profile Type",
        options=["No Preference", "Candidate", "Lead"],
        index=["No Preference", "Candidate", "Lead"].index(st.session_state.profile_type),
        key="type_selectbox"
    )

# Callback function for language proficiency changes
def update_language_proficiency():
    # Get the widget key that triggered the callback
    triggered_key = st.session_state.last_triggered_element
    
    # Extract language name from the key (format: "prof_i_Language")
    if triggered_key and triggered_key.startswith("prof_"):
        parts = triggered_key.split("_", 2)
        if len(parts) >= 3:
            language = parts[2]
            # Update the language proficiency in session state
            if language in st.session_state.languages:
                st.session_state.languages[language] = st.session_state[triggered_key]

def render_language_filters():
    """Render language and proficiency filters for detected languages only."""
    st.subheader("Required Languages")
    
    # Initialize original_languages if it doesn't exist
    if "original_languages" not in st.session_state:
        st.session_state.original_languages = st.session_state.languages.copy() if hasattr(st.session_state, "languages") else {}
    
    # Display detected languages info using the original values
    if st.session_state.original_languages:
        detected_str = ", ".join([f"{lang} ({prof})" for lang, prof in st.session_state.original_languages.items()])
        st.markdown(f"""<div style="text-shadow: 0 0 5px rgba(255,255,255,0.3); font-size: 1.05em;">
                    <span style="color: white;">✨ Detected from JD: </span>
                    <span style="color: #FFCC80;">{detected_str}</span>
                    <span style="color: white;">. Adjust proficiency if needed</span>
                    </div>""", 
                    unsafe_allow_html=True)
    
    st.caption("Each proficiency level includes all higher levels automatically.")
    
    # Define standardized proficiency levels
    proficiency_levels = [
        "Elementary Proficiency",
        "Limited Working Proficiency",
        "Professional Working Proficiency",
        "Full Professional Proficiency",
        "Native or Bilingual Proficiency"
    ]
    
    # If no languages detected, show a message
    if not st.session_state.languages:
        st.warning("No languages were detected in the job description.")
        return
    
    # Store last triggered element for callback
    if "last_triggered_element" not in st.session_state:
        st.session_state.last_triggered_element = None
    
    # Render a row for each detected language
    for i, (language, proficiency) in enumerate(list(st.session_state.languages.items())):
        cols = st.columns([1, 2])
        
        with cols[0]:
            # Display language name (non-editable)
            st.markdown(f"**{language}**")
        
        with cols[1]:
            # Find best match for current proficiency in our standardized levels
            current_proficiency = proficiency
            best_match_index = 2  # Default to Professional Working Proficiency
            
            for j, level in enumerate(proficiency_levels):
                if isinstance(current_proficiency, str) and level.lower() == current_proficiency.lower():
                    best_match_index = j
                    break
            
            # Create a unique key for this selectbox
            selectbox_key = f"prof_{i}_{language}"
            
            # Render proficiency selection for this language with callback
            new_proficiency = st.selectbox(
                label=f"Proficiency for {language}",
                options=proficiency_levels,
                index=best_match_index,
                key=selectbox_key,
                on_change=lambda: setattr(st.session_state, "last_triggered_element", selectbox_key)
            )
            
            # Update the proficiency in session state
            st.session_state.languages[language] = new_proficiency

def convert_to_pinecone_filter(modified_filters):
    """
    Convert user-modified filters to Pinecone filter format.
    
    Args:
        modified_filters: Dictionary of user-modified filters
        
    Returns:
        Dictionary in Pinecone filter syntax
    """
    pinecone_filter = {}
    
    # Handle Gender filter (if present)
    if "gender" in modified_filters and modified_filters["gender"] != "No Preference":
        gender_value = modified_filters["gender"].lower()
        # Include both specified gender and "not_mentioned"
        pinecone_filter["$or"] = [
            {"gender": gender_value},
            {"gender": "not_mentioned"}
        ]
    
    # Handle Years of Experience filter with range (min and max)
    if "years_of_experience" in modified_filters:
        # We need to create a filter with min and max values
        min_years = modified_filters.get("min_years_experience", modified_filters["years_of_experience"])
        max_years = modified_filters.get("max_years_experience", min_years + 5)
        
        # Create experience range filter
        exp_filter = {"$and": []}
        
        # Add minimum years filter (greater than or equal to min_years)
        if min_years > 0:
            exp_filter["$and"].append({"years_of_experience": {"$gte": min_years}})
        
        # Add maximum years filter (less than or equal to max_years)
        if max_years < 50:  # Only add the max filter if it's less than the max possible value
            exp_filter["$and"].append({"years_of_experience": {"$lte": max_years}})
        
        # Add experience filter to pinecone_filter
        if exp_filter["$and"]:
            if "$and" not in pinecone_filter:
                pinecone_filter["$and"] = []
            pinecone_filter["$and"].append(exp_filter)
    
    # Handle Type filter (if present)
    if "type" in modified_filters and modified_filters["type"] != "No Preference":
        is_candidate = modified_filters["type"] == "Candidate"
        pinecone_filter["is_candidate"] = is_candidate
    
    # Handle Languages filter
    if "languages" in modified_filters and modified_filters["languages"]:
        language_filters = []
        
        # Threshold mapping - each level includes all higher levels
        threshold_mapping = {
            "Elementary Proficiency": ["Elementary Proficiency", "Limited Working Proficiency", 
                                      "Professional Working Proficiency", "Full Professional Proficiency", 
                                      "Native or Bilingual Proficiency"],
            "Limited Working Proficiency": ["Limited Working Proficiency", "Professional Working Proficiency", 
                                          "Full Professional Proficiency", "Native or Bilingual Proficiency"],
            "Professional Working Proficiency": ["Professional Working Proficiency", "Full Professional Proficiency", 
                                               "Native or Bilingual Proficiency"],
            "Full Professional Proficiency": ["Full Professional Proficiency", "Native or Bilingual Proficiency"],
            "Native or Bilingual Proficiency": ["Native or Bilingual Proficiency"]
        }
        
        for language, proficiency in modified_filters["languages"].items():
            # Get threshold levels - use exact case matching since we control the options now
            included_levels = threshold_mapping.get(proficiency, [])
            
            # Create filter for this language
            language_filter = {
                language: {
                    "$in": included_levels
                }
            }
            
            language_filters.append(language_filter)
        
        # If we have language filters, add them with $and operator
        if language_filters:
            if "$and" not in pinecone_filter:
                pinecone_filter["$and"] = []
            
            pinecone_filter["$and"].extend(language_filters)
    
    return pinecone_filter 