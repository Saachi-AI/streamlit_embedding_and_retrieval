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
        normalized = {}
        
        if "prompt" in data:
            normalized["prompt"] = data["prompt"]
        
        if "metadata" in data:
            normalized["metadata"] = data["metadata"]
        
        return normalized

    def generate_search_prompt(self, job_description: str) -> Dict[str, Any]:
        """
        Generate a semantic search prompt from a job description
        
        Args:
            job_description: The parsed job description text
            
        Returns:
            Dict containing the generated prompt and extracted metadata
        """
        # Define the prompt template using content from prompt_to_extract_from_JD.md
        prompt = f"""You are an advanced language model acting as a veteran recruiter. Your goal is to analyze a given job description and produce two key outputs: essential metadata for filtering candidates in a vector database and a concise semantic search prompt (200–250 words) for searching in the vector database.

## Essential Metadata Extraction

We only need the following two metadata fields from the job description:

- **min_years_experience**: A numeric value if the job description explicitly mentions years of experience. Leave it empty if not stated.  
- **languages**: An array containing any natural/human language requirements (e.g., English, Japanese, Spanish), along with mapped proficiency levels.

### Metadata Rules

1. Leave a field empty if not explicitly mentioned in the job description.  
2. **Do not** make assumptions about fields that are not mentioned.  
3. For language proficiency levels, map to these standard values:  
   - "native-level," "mother tongue," or "native speaker" → "Native or Bilingual proficiency"  
   - "business-level," "professional," or "fluent" → "Professional working proficiency"  
   - "conversational," "intermediate" → "Limited working proficiency"  
   - "basic," "elementary," or "beginner" → "Elementary Proficiency"  
4. For Japanese specifically, map JLPT certification levels as follows:  
   - "N1" → "Native or Bilingual proficiency"  
   - "N2" → "Full professional proficiency"  
   - "N3" → "Professional working proficiency"  
   - "N4" → "Limited working proficiency"  
   - "N5" → "Elementary Proficiency"  
5. Return the languages as an array of objects, each having one language name and its proficiency level.  
6. If no proficiency level is specified, assume **"Professional working proficiency."**  
7. Only include human/natural languages and exclude programming languages.

## Concise Semantic Search Prompt

1. Identify **core technical** (e.g., software development frameworks, data analysis tools) **or domain-specific** skills, certifications, or specialized knowledge (e.g., clinical procedures, regulatory compliance, energy auditing, finance regulations) that the role requires.
2. Note all explicitly required certifications or licenses (e.g., PMP, AWS Certifications or relevant healthcare, finance, or other industry equivalents).  
3. Prioritize mandatory requirements—focus on the most crucial skills, minimum experience, and language proficiencies.  
4. If the role specifies a certain seniority (e.g., Senior Developer, Mid-level Manager) or a specific functional area (e.g., UI/UX, Data Engineering or healthcare administration, energy operations, financial analysis, etc.,), incorporate these into the prompt.
5. If "preferred" or "nice-to-have" qualifications are mentioned, you may note them separately, but do not overemphasize them.
6. If the job mentions industry-specific regulatory or compliance requirements, include them as mandatory skills.

### Rules for the Prompt

1. It must be **200–250 words** in length.  
2. Exclude irrelevant details (e.g., office address, work hours, salary) unless they directly impact skill or experience requirements.

---

### Job Description
{job_description}

## Required Output Format

Return a valid JSON object in the following format:

{{
  "prompt": "Your semantic search prompt...",
  "metadata": {{
    "min_years_experience": ...,
    "languages": [
      {{
        "Language": "Proficiency Level"
      }}
    ]
  }}
}}
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
                    
                    # Instead of passing the entire response_data to _normalize_response_data
                    # or returning it directly, create a clean version with only desired fields
                    clean_response = {
                        "prompt": prompt_data.get("prompt", ""),
                        "metadata": prompt_data.get("metadata", {})
                    }
                    
                    return clean_response
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