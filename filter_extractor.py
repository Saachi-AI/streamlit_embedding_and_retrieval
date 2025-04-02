import time
import json
from typing import Dict, Any, List, Optional, Union
import streamlit as st

# For Groq integration
from langchain_groq import ChatGroq

class FilterExtractor:
    """
    Extracts metadata filters from natural language queries using DeepSeek R1 Distill Llama 70B via Groq.
    """
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key"""
        self.llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",  # DeepSeek R1 Distill Llama 70B
            api_key=api_key,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=1024
        )
        
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from natural language query using LLM"""
        # Define the prompt template
        prompt = f"""You are a metadata extraction assistant. Extract the following fields from the user's candidate search query:

- gender: Extract if user EXPLICITLY specifies "male" or "female"
- years_of_experience: Extract numeric value if the user mentions years of experience
- last_contacted: Extract time period in years if user mentions when candidates were last contacted
- is_candidate: Extract boolean ONLY if user EXPLICITLY mentions active candidate status
- placed: Extract boolean ONLY if user EXPLICITLY mentions placement status
- languages: Extract any natural/human language requirements (like English, Japanese, Spanish) with their proficiency levels

Rules:
1. Leave a field empty if not explicitly mentioned in the query
2. For gender, only extract if specifically mentioned
3. For experience, extract the minimum years as a number
4. For last_contacted, extract the time period in years
5. For is_candidate and placed, ONLY include these if EXPLICITLY mentioned in the query
6. DO NOT make assumptions about fields that are not mentioned
7. For language proficiency levels, map to standard values:
   - "native-level", "mother tongue", "native speaker" -> "Native or Bilingual proficiency"
   - "business-level", "professional", "fluent" -> "Professional working proficiency"
   - "conversational", "intermediate" -> "Limited working proficiency"
   - "basic", "elementary", "beginner" -> "Elementary Proficiency"
8. For Japanese language specifically, also map JLPT certification levels:
   - "N1" -> "Native or Bilingual proficiency"
   - "N2" -> "Full professional proficiency" 
   - "N3" -> "Professional working proficiency"
   - "N4" -> "Limited working proficiency"
   - "N5" -> "Elementary Proficiency"
9. Return languages as an object with language names as keys and proficiency levels as values
10. If proficiency level is not specified for a language, assume "Professional working proficiency"
11. Only include human/natural languages like English, French, Japanese, etc. Do not include programming languages.

USER QUERY: "{query}"

Return a JSON object with only the fields that were EXPLICITLY mentioned in the query. DO NOT include fields that are not mentioned.
"""
        
        # Get response from LLM
        try:
            response = self.llm.invoke(prompt).content
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    extracted_filters = json.loads(json_str)
                    
                    # Post-process to map Japanese language levels if needed
                    if "languages" in extracted_filters and "Japanese" in extracted_filters["languages"]:
                        japanese_level = extracted_filters["languages"]["Japanese"]
                        # Check if the level contains N1-N5 notation but wasn't properly mapped
                        if "N1" in japanese_level:
                            extracted_filters["languages"]["Japanese"] = "Native or Bilingual proficiency"
                        elif "N2" in japanese_level:
                            extracted_filters["languages"]["Japanese"] = "Full professional proficiency"
                        elif "N3" in japanese_level:
                            extracted_filters["languages"]["Japanese"] = "Professional working proficiency"
                        elif "N4" in japanese_level:
                            extracted_filters["languages"]["Japanese"] = "Limited working proficiency"
                        elif "N5" in japanese_level:
                            extracted_filters["languages"]["Japanese"] = "Elementary Proficiency"
                    
                    return extracted_filters
                else:
                    st.warning("LLM response did not contain valid JSON. Using empty filter.")
                    return {}
            except json.JSONDecodeError as e:
                st.warning(f"Failed to parse JSON from LLM response: {e}")
                return {}
        except Exception as e:
            st.error(f"Error calling Groq API: {e}")
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
        pinecone_filter = {"$and": []}
        
        # Process gender filter
        if "gender" in extracted_filters:
            gender_value = extracted_filters["gender"].lower()
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
        
        # Process years of experience filter
        if "years_of_experience" in extracted_filters:
            # Try to convert to int, default to 0 if not possible
            try:
                min_years = int(extracted_filters["years_of_experience"])
            except (ValueError, TypeError):
                min_years = 0
                
            if min_years > 0:
                # Include both profiles with minimum years AND profiles with unknown years (0)
                exp_filter = {"$or": [
                    {"years_of_experience": {"$gte": min_years}},
                    {"years_of_experience": {"$eq": 0}}  # Include profiles with unknown years
                ]}
                pinecone_filter["$and"].append(exp_filter)
        
        # Process last_contacted filter
        if "last_contacted" in extracted_filters:
            try:
                years = float(extracted_filters["last_contacted"])
                # Convert years to timestamp (current time - years in seconds)
                current_time = int(time.time())
                cutoff_timestamp = current_time - int(years * 365 * 24 * 3600)
                
                contact_filter = {"$or": [
                    {"last_contacted": {"$gte": cutoff_timestamp}}
                ]}
                # Removed the condition for missing fields for non-language metadata
                pinecone_filter["$and"].append(contact_filter)
            except (ValueError, TypeError):
                pass  # Skip if not a valid number
        
        # Process is_candidate filter
        if "is_candidate" in extracted_filters:
            try:
                is_candidate_value = str(extracted_filters["is_candidate"]).lower() in ['true', 'yes', '1']
                candidate_filter = {"$or": [
                    {"is_candidate": {"$eq": is_candidate_value}}
                ]}
                # Removed the condition for missing fields for non-language metadata
                pinecone_filter["$and"].append(candidate_filter)
            except (ValueError, TypeError):
                pass
        
        # Process placed filter
        if "placed" in extracted_filters:
            try:
                placed_value = str(extracted_filters["placed"]).lower() in ['true', 'yes', '1']
                placed_filter = {"$or": [
                    {"placed": {"$eq": placed_value}}
                ]}
                # Removed the condition for missing fields for non-language metadata
                pinecone_filter["$and"].append(placed_filter)
            except (ValueError, TypeError):
                pass
        
        # Process language filters
        if "languages" in extracted_filters and isinstance(extracted_filters["languages"], dict):
            self._process_language_filters(pinecone_filter, extracted_filters["languages"], strict_mode)
            
        # If no filters were applied, return empty object
        if not pinecone_filter["$and"]:
            return {}
            
        return pinecone_filter
        
    def _process_language_filters(self, pinecone_filter, language_filters, strict_mode=False):
        """
        Process language filters and add them to the Pinecone filter
        
        Args:
            pinecone_filter: The filter to add language filters to
            language_filters: Dict of language to proficiency level
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
        """
        for language, level in language_filters.items():
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
            
    def process_query(self, query: str, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query and return Pinecone filter
        
        Args:
            query: Natural language query from user
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
            
        Returns:
            Dict with pinecone_filter and extracted_filters
        """
        extracted_filters = self.extract_filters(query)
        pinecone_filter = self.build_pinecone_filter(extracted_filters, strict_mode)
        
        return {
            "pinecone_filter": pinecone_filter,
            "extracted_filters": extracted_filters,
            "strict_mode": strict_mode
        } 