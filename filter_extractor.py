import time
import json
import logging
import traceback
import os
import re
from typing import Dict, Any, List, Optional, Union
import streamlit as st
from openai import OpenAI

# Set up file-based logging
debug_log_file = "filter_debug.log"
with open(debug_log_file, "w") as f:
    f.write("===== DEBUGGING SESSION STARTED =====\n")

def debug_log(message):
    """Write debug messages to both console and log file"""
    print(message)
    with open(debug_log_file, "a") as f:
        f.write(f"{message}\n")

# Set up logging
logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger("FilterExtractor")

# For Groq integration - kept for backward compatibility
from langchain_groq import ChatGroq

class FilterExtractor:
    """
    Extracts metadata filters from natural language queries using DeepSeek R1 Distill Llama 70B via Groq.
    """
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key"""
        debug_log("FilterExtractor initialized")
        # Keep the LangChain ChatGroq for backward compatibility
        self.llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",  # DeepSeek R1 Distill Llama 70B
            api_key=api_key,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=1024
        )
        
        # Initialize the OpenAI client with Groq base URL
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
        # Define the JSON example for the prompt
        self.json_example = """
{
  "years_of_experience": 5,
  "languages": [
    {
      "Language": "English",
      "Proficiency Level": "Professional working proficiency"
    },
    {
      "Language": "Japanese",
      "Proficiency Level": "Native or Bilingual proficiency"
    }
  ]
}
"""
        
        # Define the system prompt
        self.system_prompt = """
You are a metadata extraction assistant. Extract the following fields from the user's candidate search query:

- years_of_experience: Extract numeric value if the user mentions years of experience
- languages: Extract any natural/human language requirements (like English, Japanese, Spanish) with their proficiency levels

Rules:
1. Leave a field empty if not explicitly mentioned in the query
2. For experience, extract the minimum years as a number
3. For language proficiency levels, map to standard values:
   - "native-level", "mother tongue", "native speaker" -> "Native or Bilingual proficiency"
   - "business-level", "professional", "fluent" -> "Professional working proficiency"
   - "conversational", "intermediate" -> "Limited working proficiency"
   - "basic", "elementary", "beginner" -> "Elementary Proficiency"
4. For Japanese language specifically, also map JLPT certification levels:
   - "N1" -> "Native or Bilingual proficiency"
   - "N2" -> "Full professional proficiency" 
   - "N3" -> "Professional working proficiency"
   - "N4" -> "Limited working proficiency"
   - "N5" -> "Elementary Proficiency"
5. ALWAYS return your response in EXACTLY this JSON format:
""" + self.json_example + """
6. IMPORTANT: For languages, always use an array of objects as shown above, never a dictionary or other format.
7. If proficiency level is not specified for a language, assume "Professional working proficiency"
8. Only include human/natural languages like English, French, Japanese, etc. Do not include programming languages.
9. ALWAYS include English and Japanese in the languages array, even if not mentioned in the query:
   - If English is not mentioned in the query, add it with "Limited working proficiency" (Conversational)
   - If Japanese is not mentioned in the query, add it with "Limited working proficiency" (Conversational)
   - If English or Japanese are explicitly mentioned, use their specified proficiency levels instead
   - Any other languages mentioned should be included alongside English and Japanese

CRITICAL: Return a JSON object with only the fields that were EXPLICITLY mentioned in the query. 
DO NOT include fields that are not mentioned.
DO NOT provide any explanations outside of the JSON.
ONLY return the JSON object, nothing else.
"""
        
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from natural language query using LLM"""
        # Wrap the entire method in try-except to catch any exceptions
        try:
            debug_log("\n\n===== EXTRACT FILTERS CALLED =====")
            debug_log(f"Query: {query}")
            
            # Format the user prompt
            user_prompt = f"""
USER QUERY: "{query}"

Extract metadata from this query and return ONLY a JSON object, with NO additional text.
"""
            
            debug_log("System and user prompts defined")
            
            # Get response from LLM using OpenAI client with Groq API
            try:
                debug_log("Calling Groq API with OpenAI client...")
                response = self.client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                
                # Extract LLM response
                llm_response = response.choices[0].message.content
                debug_log(f"Raw LLM response: {llm_response}")
                
                # Extract JSON from the response using regex
                try:
                    # Remove thinking part if present
                    cleaned_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
                    debug_log(f"Cleaned response (after removing thinking): {cleaned_response}")
                    
                    # Try to find JSON in code blocks
                    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned_response)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        debug_log(f"Extracted JSON from code block: {json_str}")
                    else:
                        # If no code blocks, try to find JSON using braces
                        start_idx = cleaned_response.find('{')
                        end_idx = cleaned_response.rfind('}') + 1
                        
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = cleaned_response[start_idx:end_idx]
                            debug_log(f"Extracted JSON using braces: {json_str}")
                        else:
                            # If no JSON found, use the entire cleaned response
                            json_str = cleaned_response.strip()
                            debug_log(f"Using entire cleaned response as JSON: {json_str}")
                    
                    # Parse the JSON
                    extracted_filters = json.loads(json_str)
                    debug_log(f"Parsed JSON object: {extracted_filters}")
                    
                    # Post-process languages if they're in dictionary format
                    if "languages" in extracted_filters and isinstance(extracted_filters["languages"], dict):
                        debug_log("Languages in dictionary format, converting to array...")
                        languages_dict = extracted_filters["languages"]
                        languages_array = []
                        for language, level in languages_dict.items():
                            languages_array.append({
                                "Language": language,
                                "Proficiency Level": level
                            })
                        extracted_filters["languages"] = languages_array
                        debug_log(f"Converted languages to array: {extracted_filters['languages']}")
                    
                    # Post-process to map Japanese language levels if needed
                    if "languages" in extracted_filters and isinstance(extracted_filters["languages"], list):
                        debug_log("Processing languages in array format...")
                        for lang_obj in extracted_filters["languages"]:
                            if lang_obj.get("Language") == "Japanese":
                                japanese_level = lang_obj.get("Proficiency Level", "")
                                # Check if the level contains N1-N5 notation but wasn't properly mapped
                                if "N1" in japanese_level:
                                    lang_obj["Proficiency Level"] = "Native or Bilingual proficiency"
                                elif "N2" in japanese_level:
                                    lang_obj["Proficiency Level"] = "Full professional proficiency"
                                elif "N3" in japanese_level:
                                    lang_obj["Proficiency Level"] = "Professional working proficiency"
                                elif "N4" in japanese_level:
                                    lang_obj["Proficiency Level"] = "Limited working proficiency"
                                elif "N5" in japanese_level:
                                    lang_obj["Proficiency Level"] = "Elementary Proficiency"
                    
                    debug_log(f"Final extracted filters: {extracted_filters}")
                    debug_log("===== EXTRACT FILTERS COMPLETED =====\n")
                    return extracted_filters
                except json.JSONDecodeError as e:
                    debug_log(f"ERROR: Failed to parse JSON from LLM response: {e}")
                    debug_log(f"Problematic JSON string: {json_str if 'json_str' in locals() else 'Not extracted'}")
                    st.warning(f"Failed to parse JSON from LLM response: {e}")
                    debug_log("===== EXTRACT FILTERS COMPLETED WITH EMPTY RESULT =====\n")
                    return {}
            except Exception as e:
                debug_log(f"ERROR: Error calling Groq API: {str(e)}")
                st.error(f"Error calling Groq API: {str(e)}")
                debug_log("===== EXTRACT FILTERS COMPLETED WITH EMPTY RESULT =====\n")
                return {}
        except Exception as e:
            debug_log(f"CRITICAL ERROR: Unexpected exception in extract_filters: {str(e)}")
            debug_log(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error extracting metadata filters: {str(e)}")
            return {}
            
    def build_pinecone_filter(self, extracted_filters: Dict[str, Any], strict_mode: bool = False) -> Dict[str, Any]:
        """
        Convert extracted filters into Pinecone filter format
        
        Args:
            extracted_filters: Dict of filters extracted by the LLM
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
            
        Returns:
            Dict in Pinecone filter format
        """
        try:
            debug_log(f"Building Pinecone filter from extracted filters: {json.dumps(extracted_filters)}")
            
            pinecone_filter = {"$and": []}
            
            # Create the skills exclusion filter using $ne (not equal) operator
            skills_exclusion_filter = {"section": {"$ne": "skills"}}
            pinecone_filter["$and"].append(skills_exclusion_filter)
            
            # last_contacted filter for the last 8 years
            try:
                current_time = int(time.time())
                eight_years_ago = current_time - (8 * 365 * 24 * 60 * 60)  # 8 years in seconds
                debug_log(f"Adding last_contacted filter: timestamps >= {eight_years_ago}")
                
                last_contacted_filter = {"last_contacted": {"$gte": eight_years_ago}}
                pinecone_filter["$and"].append(last_contacted_filter)
            except Exception as e:
                debug_log(f"Error adding last_contacted filter: {str(e)}")
            
            # Process gender filter
            if "gender" in extracted_filters:
                try:
                    gender_value = extracted_filters["gender"].lower()
                    debug_log(f"Processing gender filter: {gender_value}")
                    gender_filter = {"$or": []}
                    
                    if gender_value == "male":
                        gender_filter["$or"].append({"gender": {"$eq": "male"}})
                        # Also include "not_mentioned"
                        gender_filter["$or"].append({"gender": {"$eq": "not mentioned"}})
                            
                    elif gender_value == "female":
                        gender_filter["$or"].append({"gender": {"$eq": "female"}})
                        # Also include "not_mentioned"
                        gender_filter["$or"].append({"gender": {"$eq": "not mentioned"}})
                    
                    if gender_filter["$or"]:
                        pinecone_filter["$and"].append(gender_filter)
                except Exception as e:
                    debug_log(f"Error processing gender filter: {str(e)}")
            
            # Process years of experience filter
            if "years_of_experience" in extracted_filters:
                try:
                    # Try to convert to int, default to 0 if not possible
                    debug_log(f"Processing years_of_experience: {extracted_filters['years_of_experience']}")
                    min_years = int(extracted_filters["years_of_experience"])
                    
                    # Check if we have min and max years experience
                    if "min_years_experience" in extracted_filters and "max_years_experience" in extracted_filters:
                        debug_log(f"Processing min/max years experience range: {extracted_filters['min_years_experience']} to {extracted_filters['max_years_experience']}")
                        min_years = int(extracted_filters["min_years_experience"])
                        max_years = int(extracted_filters["max_years_experience"])
                        
                        # Create a range filter with min and max
                        if min_years > 0 or max_years < 50:
                            exp_filter = {"$and": []}
                            
                            # Add minimum years filter
                            if min_years > 0:
                                exp_filter["$and"].append({"years_of_experience": {"$gte": min_years}})
                            
                            # Add maximum years filter
                            if max_years < 50:
                                exp_filter["$and"].append({"years_of_experience": {"$lte": max_years}})
                            
                            # Include profiles with unknown years (0) if needed
                            if not strict_mode:
                                exp_filter = {"$or": [exp_filter, {"years_of_experience": {"$eq": 0}}]}
                            
                            pinecone_filter["$and"].append(exp_filter)
                    # Traditional single value for minimum years
                    elif min_years > 0:
                        # Include both profiles with minimum years AND profiles with unknown years (0)
                        exp_filter = {"$or": [
                            {"years_of_experience": {"$gte": min_years}},
                            {"years_of_experience": {"$eq": 0}}  # Include profiles with unknown years
                        ]}
                        pinecone_filter["$and"].append(exp_filter)
                except (ValueError, TypeError) as e:
                    debug_log(f"Error processing years_of_experience: {str(e)}")
                    min_years = 0
            
            # Process type filter (for Candidate/Lead distinction)
            if "type" in extracted_filters:
                try:
                    type_value = extracted_filters["type"]
                    debug_log(f"Processing type filter: {type_value}")
                    
                    if type_value == "Candidate":
                        # For Candidate, set is_candidate to true
                        candidate_filter = {"is_candidate": {"$eq": True}}
                        pinecone_filter["$and"].append(candidate_filter)
                    elif type_value == "Lead":
                        # For Lead, set is_candidate to false
                        candidate_filter = {"is_candidate": {"$eq": False}}
                        pinecone_filter["$and"].append(candidate_filter)
                    # If "No Preference" (or any other value), don't add a filter
                except Exception as e:
                    debug_log(f"Error processing type filter: {str(e)}")
                    
            # Process language filters
            if "languages" in extracted_filters:
                debug_log(f"Processing languages: {json.dumps(extracted_filters['languages'])}")
                try:
                    if isinstance(extracted_filters["languages"], dict):
                        # Handle old dictionary format
                        debug_log("Languages in dictionary format")
                        self._process_language_filters(pinecone_filter, extracted_filters["languages"], strict_mode)
                    elif isinstance(extracted_filters["languages"], list):
                        # Handle new array format
                        debug_log("Languages in array format")
                        self._process_language_filters(pinecone_filter, extracted_filters["languages"], strict_mode)
                    else:
                        debug_log(f"Languages in unexpected format: {type(extracted_filters['languages'])}")
                except Exception as e:
                    debug_log(f"Error processing language filters: {str(e)}")
            else:
                debug_log("No languages found in extracted filters")
                
            # If no filters were applied, return only the skills exclusion filter
            if len(pinecone_filter["$and"]) == 1:
                debug_log("Only skills exclusion filter applied")
                return skills_exclusion_filter
            
            debug_log(f"Final Pinecone filter: {json.dumps(pinecone_filter)}")
            return pinecone_filter
        except Exception as e:
            debug_log(f"CRITICAL ERROR: Unexpected exception in build_pinecone_filter: {str(e)}")
            debug_log(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error building Pinecone filter: {str(e)}")
            return {}
        
    def _process_language_filters(self, pinecone_filter, language_filters, strict_mode=False):
        """
        Process language filters and add them to the Pinecone filter
        
        Args:
            pinecone_filter: The filter to add language filters to
            language_filters: List of language objects with Language and Proficiency Level
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
        """
        try:
            debug_log(f"Processing language filters: {json.dumps(language_filters)}")
            # Handle both dictionary format and array format from the image
            if isinstance(language_filters, dict):
                # Convert dictionary to array format for consistency
                debug_log("Converting dictionary language format to array")
                language_array = []
                for language, level in language_filters.items():
                    language_array.append({
                        "Language": language,
                        "Proficiency Level": level
                    })
                language_filters = language_array
            
            # Process each language in the array
            for lang_obj in language_filters:
                # Extract language and level from object
                debug_log(f"Processing language object: {json.dumps(lang_obj)}")
                if isinstance(lang_obj, dict) and "Language" in lang_obj and "Proficiency Level" in lang_obj:
                    language = lang_obj["Language"]
                    level = lang_obj["Proficiency Level"]
                    debug_log(f"Found Language: {language}, Level: {level}")
                else:
                    debug_log(f"Skipping invalid language object: {json.dumps(lang_obj)}")
                    continue
                    
                language_key = language.capitalize()  # Ensure proper capitalization
                level = level.lower() if isinstance(level, str) else ""
                
                # Map proficiency level terms to standard values
                if level in ["native", "native-level", "native or bilingual proficiency", "mother tongue", "native speaker", "n1"]:
                    proficiency_values = ["Native or Bilingual proficiency"]
                elif level in ["business", "business-level", "professional", "fluent", "professional working proficiency", "n3", "full professional proficiency", "n2"]:
                    proficiency_values = [
                        "Professional working proficiency",
                        "Full professional proficiency",
                        "Native or Bilingual proficiency"
                    ]
                elif level in ["conversational", "intermediate", "limited working proficiency", "n4"]:
                    proficiency_values = [
                        "Limited working proficiency",
                        "Professional working proficiency",
                        "Full professional proficiency",
                        "Native or Bilingual proficiency"
                    ]
                elif level in ["basic", "elementary", "beginner", "elementary proficiency", "n5"]:
                    proficiency_values = [
                        "Elementary Proficiency",
                        "Limited working proficiency",
                        "Professional working proficiency",
                        "Full professional proficiency",
                        "Native or Bilingual proficiency"
                    ]
                else:
                    # Default to all levels if unrecognized
                    proficiency_values = [
                        "Elementary Proficiency",
                        "Limited working proficiency",
                        "Professional working proficiency",
                        "Full professional proficiency",
                        "Native or Bilingual proficiency"
                    ]
                
                # Create language filter
                language_filter = {"$or": [
                    {language_key: {"$in": proficiency_values}}
                ]}
                
                # In non-strict mode, also include docs where this language field doesn't exist
                if not strict_mode:
                    language_filter["$or"].append({language_key: {"$exists": False}})
                    
                pinecone_filter["$and"].append(language_filter)
                
                debug_log(f"Added language filter for {language_key}: {json.dumps(language_filter)}")
        except Exception as e:
            debug_log(f"Error in _process_language_filters: {str(e)}")
            debug_log(f"Traceback: {traceback.format_exc()}")
            
    def process_query(self, query: str, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query and return Pinecone filter
        
        Args:
            query: Natural language query from user
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
            
        Returns:
            Dict with pinecone_filter and extracted_filters
        """
        try:
            debug_log(f"\n\n===== PROCESS QUERY STARTED =====")
            debug_log(f"Processing query: {query}")
            debug_log(f"Strict mode: {strict_mode}")
            
            # Step 1: Extract filters
            debug_log("Calling extract_filters...")
            extracted_filters = self.extract_filters(query)
            debug_log(f"extract_filters returned: {json.dumps(extracted_filters)}")
            
            # Step 2: Build Pinecone filter
            debug_log("Calling build_pinecone_filter...")
            pinecone_filter = self.build_pinecone_filter(extracted_filters, strict_mode)
            debug_log(f"build_pinecone_filter returned: {json.dumps(pinecone_filter)}")
            
            debug_log("===== PROCESS QUERY COMPLETED =====\n")
            return {
                "pinecone_filter": pinecone_filter,
                "extracted_filters": extracted_filters,
                "strict_mode": strict_mode
            }
        except Exception as e:
            debug_log(f"CRITICAL ERROR: Unexpected exception in process_query: {str(e)}")
            debug_log(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error extracting metadata filters: {str(e)}")
            return {
                "pinecone_filter": {},
                "extracted_filters": {},
                "strict_mode": strict_mode
            } 