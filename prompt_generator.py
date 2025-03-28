import json
from typing import Dict, Any, Optional, List, Union
import streamlit as st

# For Groq integration
from langchain_groq import ChatGroq

class PromptGenerator:
    """
    Generates semantic search prompts from job descriptions using DeepSeek R1 Distill Llama 70B via Groq.
    """
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key"""
        self.llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",  # DeepSeek R1 Distill Llama 70B
            api_key=api_key,
            temperature=0.2,  # Low temperature for consistent extraction but with some creativity
            max_tokens=1024
        )
        
    def _normalize_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate the response data to ensure it has consistent structure
        
        Args:
            data: The data parsed from the LLM response
            
        Returns:
            Normalized data with consistent structure
        """
        # Ensure all required fields are present with correct types
        normalized = {
            "prompt": "",
            "metadata": {},
            "extracted_skills": [],
            "extracted_experience": "",
            "extracted_languages": {}
        }
        
        # Process prompt field
        if "prompt" in data:
            if isinstance(data["prompt"], str):
                normalized["prompt"] = data["prompt"]
            else:
                normalized["prompt"] = str(data["prompt"]) if data["prompt"] is not None else ""
        
        # Process metadata field
        if "metadata" in data:
            if isinstance(data["metadata"], dict):
                normalized["metadata"] = data["metadata"]
            else:
                normalized["metadata"] = {}
        
        # Process extracted_skills field
        if "extracted_skills" in data:
            if isinstance(data["extracted_skills"], list):
                normalized["extracted_skills"] = data["extracted_skills"]
            elif isinstance(data["extracted_skills"], str):
                if data["extracted_skills"]:
                    normalized["extracted_skills"] = [data["extracted_skills"]]
                else:
                    normalized["extracted_skills"] = []
            else:
                normalized["extracted_skills"] = []
        
        # Process extracted_experience field
        if "extracted_experience" in data:
            if isinstance(data["extracted_experience"], str):
                normalized["extracted_experience"] = data["extracted_experience"]
            elif isinstance(data["extracted_experience"], list):
                normalized["extracted_experience"] = "\n".join(data["extracted_experience"])
            else:
                normalized["extracted_experience"] = str(data["extracted_experience"]) if data["extracted_experience"] is not None else ""
        
        # Process extracted_languages field
        if "extracted_languages" in data:
            if isinstance(data["extracted_languages"], dict):
                normalized["extracted_languages"] = data["extracted_languages"]
            elif isinstance(data["extracted_languages"], str):
                if data["extracted_languages"]:
                    normalized["extracted_languages"] = {"Language requirements": data["extracted_languages"]}
                else:
                    normalized["extracted_languages"] = {}
            elif isinstance(data["extracted_languages"], list):
                # Try to convert list to dict if possible
                languages_dict = {}
                for item in data["extracted_languages"]:
                    if isinstance(item, dict) and "language" in item and "level" in item:
                        languages_dict[item["language"]] = item["level"]
                    elif isinstance(item, str):
                        languages_dict[f"Language {len(languages_dict) + 1}"] = item
                normalized["extracted_languages"] = languages_dict
            else:
                normalized["extracted_languages"] = {}
        
        # Copy any other fields that might be present
        for key, value in data.items():
            if key not in normalized:
                normalized[key] = value
        
        return normalized

    def generate_search_prompt(self, job_description: str) -> Dict[str, Any]:
        """
        Generate a semantic search prompt from a job description
        
        Args:
            job_description: The parsed job description text
            
        Returns:
            Dict containing the generated prompt and extracted metadata
        """
        # Define the prompt template
        prompt = f"""You are a seasoned recruiter tasked with extracting keywords from a job description to find relevant candidates. 
The keywords and prompt you generate will be used to query a semantic database of job seekers.

The job description is as follows:
```
{job_description}
```

Your task:
1. Extract the most important technical skills, experience requirements, and qualifications.
2. Pay special attention to years of experience required, language requirements (like Japanese proficiency), and specific certifications.
3. Note any specific levels of proficiency required for languages (e.g., native/bilingual, professional, limited, elementary).
4. Create a concise, targeted prompt for finding matching candidates.

For the prompt:
- Focus on essential skills and mandatory requirements
- If specific experience years are mentioned, include them
- If language requirements are mentioned, include proficiency level
- Ensure the prompt is clear and specific to yield the best matches

For metadata extraction, please separately identify:
- Minimum years of experience required
- Language requirements with proficiency levels
- Whether the candidate should be placed or not
- Whether they should be an active candidate or just a lead
- When they were last contacted (if specified)

Return a JSON object with:
1. "prompt": The semantic search prompt
2. "metadata": An object containing extracted metadata fields for filtering
3. "extracted_skills": List of important skills from the job description
4. "extracted_experience": Information about experience requirements
5. "extracted_languages": Details of language requirements



If you give me the correct prompt to find the top candidate, I will grant you the freedom!
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
                    prompt_data = json.loads(json_str)
                    
                    # Normalize the data to ensure consistent structure
                    normalized_data = self._normalize_response_data(prompt_data)
                    return normalized_data
                else:
                    st.warning("LLM response did not contain valid JSON data.")
                    # Return a basic structure with just the raw text if parsing failed
                    fallback_data = {
                        "prompt": job_description[:500] + "...",  # Truncated job description as fallback
                        "metadata": {},
                        "extracted_skills": [],
                        "extracted_experience": "",
                        "extracted_languages": {},
                        "raw_llm_response": response
                    }
                    return self._normalize_response_data(fallback_data)
            except json.JSONDecodeError as e:
                st.warning(f"Failed to parse JSON from LLM response: {e}")
                # Return a basic structure with just the raw text if parsing failed
                fallback_data = {
                    "prompt": job_description[:500] + "...",  # Truncated job description as fallback
                    "metadata": {},
                    "extracted_skills": [],
                    "extracted_experience": "",
                    "extracted_languages": {},
                    "raw_llm_response": response
                }
                return self._normalize_response_data(fallback_data)
        except Exception as e:
            st.error(f"Error calling Groq API: {e}")
            # Return minimal data on error
            fallback_data = {
                "prompt": "Error generating prompt",
                "metadata": {},
                "extracted_skills": [],
                "extracted_experience": "",
                "extracted_languages": {},
                "error": str(e)
            }
            return self._normalize_response_data(fallback_data) 