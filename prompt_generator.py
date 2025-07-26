import json
from typing import Dict, Any, Optional, List, Union
import streamlit as st
import logging
import os

# For Groq integration
from langchain_groq import ChatGroq
# For Google Gemini integration
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptGenerator:
    """
    Generates semantic search prompts from job descriptions using either DeepSeek R1 Distill Llama 70B via Groq or Google Gemini.
    """
    
    def __init__(self, api_key: str = None, llm_choice: str = "gemini"):
        """Initialize with API key and LLM choice"""
        self.llm_choice = llm_choice.lower()
        
        if self.llm_choice == "groq":
            # Initialize Groq/DeepSeek
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
            if not self.api_key:
                logger.warning("GROQ_API_KEY not found in environment. Groq LLM will not work.")
            
            self.llm = ChatGroq(
                model_name="deepseek-r1-distill-llama-70b",  # DeepSeek R1 Distill Llama 70B
                api_key=self.api_key,
                temperature=0.2,  # Low temperature for consistent extraction but with some creativity
                max_tokens=1024
            )
            logger.info("Groq/DeepSeek client initialized successfully")
            
        elif self.llm_choice == "gemini":
            # Initialize Gemini
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not found in environment. Gemini LLM will not work.")
            else:
                logger.info(f"Using Google API key (starting with: {self.api_key[:4]}{'*' * 10})")
            
            # Initialize the Gemini client
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini client initialized successfully")
            
        else:
            logger.warning(f"Unknown LLM choice: {llm_choice}. Defaulting to Gemini.")
            self.llm_choice = "gemini"
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not found in environment. Gemini LLM will not work.")
            
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini client initialized successfully (fallback)")
        
    def _call_groq_llm(self, prompt: str) -> str:
        """
        Call Groq LLM API.
        
        Args:
            prompt: The complete prompt to send
            
        Returns:
            LLM response text
        """
        try:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found. Cannot call Groq API.")
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return ""
    
    def _call_gemini_llm(self, prompt: str) -> str:
        """
        Call Gemini LLM API.
        
        Args:
            prompt: The complete prompt to send
            
        Returns:
            LLM response text
        """
        try:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found. Cannot call Gemini API.")
            
            response = self.client.models.generate_content(
                model="models/gemini-2.5-pro",
                contents=prompt,
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return ""

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
        
        return normalized

    def generate_search_prompt(self, job_description: str) -> Dict[str, Any]:
        """
        Generate a semantic search prompt from a job description
        
        Args:
            job_description: The parsed job description text
            
        Returns:
            Dict containing the generated prompt
        """
        # Define the prompt template using content from prompt_to_extract_from_JD.md
        prompt = f"""You are an advanced language model acting as a veteran recruiter. Your goal is to analyze a given job description and produce a concise semantic search prompt (200–250 words) for searching candidates in a vector database.

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
3. Focus on creating a semantic search prompt that will effectively match relevant candidate profiles.

---

### Job Description
{job_description}

## Required Output Format

Return a valid JSON object in the following format:

{{
  "prompt": "Your semantic search prompt..."
}}
"""
        
        # Get response from LLM
        try:
            logger.info(f"Using {self.llm_choice.upper()} LLM for job description summarization")
            
            if self.llm_choice == "groq":
                response_text = self._call_groq_llm(prompt)
            elif self.llm_choice == "gemini":
                response_text = self._call_gemini_llm(prompt)
            else:
                response_text = "" # Fallback if llm_choice is unexpected

            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    prompt_data = json.loads(json_str)
                    
                    # Create a clean response with only the prompt field
                    clean_response = {
                        "prompt": prompt_data.get("prompt", ""),
                    }
                    
                    logger.info(f"Job description summarization completed successfully using {self.llm_choice.upper()}")
                    return clean_response
                else:
                    st.warning("LLM response did not contain valid JSON data.")
                    # Return a basic structure with just the raw text if parsing failed
                    fallback_data = {
                        "prompt": job_description[:500] + "...",  # Truncated job description as fallback
                        "raw_llm_response": response_text
                    }
                    return self._normalize_response_data(fallback_data)
            except json.JSONDecodeError as e:
                st.warning(f"Failed to parse JSON from LLM response: {e}")
                # Return a basic structure with just the raw text if parsing failed
                fallback_data = {
                    "prompt": job_description[:500] + "...",  # Truncated job description as fallback
                    "raw_llm_response": response_text
                }
                return self._normalize_response_data(fallback_data)
        except Exception as e:
            st.error(f"Error calling Groq API: {e}")
            # Return minimal data on error
            fallback_data = {
                "prompt": "Error generating prompt",
                "raw_llm_response": str(e)
            }
            return self._normalize_response_data(fallback_data) 