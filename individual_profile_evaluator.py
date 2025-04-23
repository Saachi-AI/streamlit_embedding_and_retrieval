import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from google import genai  # Changed from OpenAI to Google's genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndividualProfileEvaluator:
    """
    Class to evaluate individual profiles against a job description using LLM.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with Google API key."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found in environment. Individual profile evaluation will not work.")
        
        # Initialize the Google client API
        self.client = genai.Client(api_key=self.api_key)
        
        # System prompt for LLM
        self.system_prompt = """
        You are a world-class recruiter with extensive experience in analyzing candidate profiles for job fit.
        
        Your task is to evaluate a SINGLE candidate profile against a job description and provide:
        1. A percentage match score (0-100%)
        2. A categorical rating ("Excellent Match", "Good Match", "Fair Match", "Poor Match")
        3. Dimensional scores for key aspects of the job fit
        
        ======================
        1. Data You Receive
        ======================
        
        Job Description
        - Explains the key requirements, domain needs, languages, etc.
        - Use this to identify what dimensions are important for this specific role
        
        Candidate Profile
        You'll receive a single candidate profile that may include:
        - profile_id: Internal ID
        - consultant_description: Recruiter's summary or notes about the candidate
        - headline: The candidate's "public headline" (like a LinkedIn tagline)
        - employments: Array of employment history objects 
        - notes: Internal consultant notes
        - skills: Grouped by industry, functional, general, or not grouped
        - education: Degrees or certifications
        - languages: Key-value pairs of language and proficiency levels
        - linkedin_description: Candidate's own summary
        
        =========================
        2. Analysis Instructions
        =========================
        
        1. First, analyze the job description to understand key requirements and dimensions
           - Identify 3-5 key dimensions relevant to this specific role (e.g., Technical Skills, Domain Experience, Leadership)
           - Dimensions should be tailored to the job domain (different for tech vs. marketing vs. finance roles)
        
        2. Evaluate the candidate against these dimensions
           - For each dimension, provide a percentage score (0-100%)
           - Provide brief reasoning for each score
        
        3. Calculate an overall percentage match
           - Weight dimensions appropriately based on the job requirements
           - This should represent how well the candidate matches the job overall
        
        4. Assign a categorical rating:
           - "Excellent Match" (85-100%): Candidate meets or exceeds all key requirements
           - "Good Match" (70-84%): Candidate meets most key requirements
           - "Fair Match" (50-69%): Candidate meets some key requirements but has notable gaps
           - "Poor Match" (0-49%): Candidate has significant gaps in key requirements
        
        5. Identify top strengths and gaps
           - List 2-3 key strengths relative to the job requirements
           - List 1-2 key gaps or areas for development
        
        =========================
        3. Output Format
        =========================
        
        Your response MUST be a JSON object with the following structure:
        
        {
          "profile_id": "profile ID from the input",
          "overall_match": {
            "percentage": 75,
            "category": "Good Match"
          },
          "dimensional_scores": [
            {
              "dimension": "Technical Skills",
              "score": 80,
              "reasoning": "Strong JavaScript and React experience, missing some backend skills"
            },
            {
              "dimension": "Domain Experience",
              "score": 65,
              "reasoning": "Has 3 years in fintech, job requires 5+ years"
            }
            // other dimensions as needed
          ],
          "key_strengths": [
            "5+ years of frontend development experience",
            "Fintech domain knowledge"
          ],
          "key_gaps": [
            "Limited experience with microservices architecture"
          ]
        }
        
        =========================
        4. Important Guidelines
        =========================
        
        1. Be specific and analytical in your reasoning
        2. Base your evaluation only on the information provided
        3. Don't hallucinate or assume additional qualifications
        4. Adapt the dimensional scoring to each specific job role
        5. Provide ONLY the JSON response, no additional text before or after
        """
    
    def evaluate_single_profile(
        self, 
        profile: Dict[str, Any], 
        raw_job_description: str,
        summarized_job_description: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single profile against a job description.
        
        Args:
            profile: Processed profile data dictionary
            raw_job_description: Full job description text
            summarized_job_description: Optional summarized job description
            
        Returns:
            Dictionary containing evaluation results or None if error
        """
        if not profile:
            logger.warning("No profile data provided for evaluation")
            return None
        
        if not raw_job_description:
            logger.warning("No job description provided for evaluation")
            return None
        
        try:
            # Format the job description to include both raw and summarized versions if available
            jd_text = ""
            if summarized_job_description:
                jd_text = f"""
                --- SUMMARIZED JD START ---
                {summarized_job_description}
                --- SUMMARIZED JD END ---
                
                --- DETAILED JD START ---
                {raw_job_description}
                --- DETAILED JD END ---
                """
            else:
                jd_text = f"""
                --- JOB DESCRIPTION START ---
                {raw_job_description}
                --- JOB DESCRIPTION END ---
                """
            
            # Convert profile to JSON string
            profile_json = json.dumps(profile, ensure_ascii=False)
            
            # Format the user prompt
            user_prompt = f"""
            {self.system_prompt}
            
            Evaluate this single candidate profile against the job description.
            Provide your evaluation in the required JSON format with dimensional scores,
            overall match percentage and category, key strengths, and key gaps.
            
            {jd_text}
            
            --- CANDIDATE PROFILE START ---
            {profile_json}
            --- CANDIDATE PROFILE END ---
            """
            
            # Call Google's Gemini API using the Client approach
            logger.info(f"Calling Google Gemini to evaluate profile {profile.get('profile_id', 'unknown')}")
            try:
                import time
                start_time = time.time()
                
                # Use models.generate_content similar to google_api.py
                response = self.client.models.generate_content(
                    model="gemini-2.5-pro-exp-03-25",
                    contents=user_prompt,
                )
                
                end_time = time.time()
                logger.info(f"API call took {end_time - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                raise
            
            # Extract and parse response
            llm_response = response.text
            logger.info("LLM Individual Profile Evaluation response:")
            logger.info(llm_response)
            
            # Extract JSON from the response - keep the same parsing logic
            try:
                # Remove thinking part
                cleaned_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL)
                
                # Try to find JSON in code blocks
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned_response)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # If no code blocks, use the cleaned response
                    json_str = cleaned_response.strip()
                
                # Parse the JSON
                parsed_response = json.loads(json_str)
                return parsed_response
            except Exception as e:
                logger.error(f"Error extracting or parsing JSON from LLM response: {str(e)}")
                logger.debug(f"Original response: {llm_response[:1000]}")
                return llm_response  # Return original response as fallback
            
        except Exception as e:
            logger.error(f"Error in individual profile evaluation: {str(e)}")
            return None
    
    def evaluate_profiles(
        self, 
        processed_profiles: List[Dict[str, Any]], 
        raw_job_description: str,
        summarized_job_description: str = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple profiles against a job description, one by one.
        
        Args:
            processed_profiles: List of processed profile data
            raw_job_description: Full job description text
            summarized_job_description: Optional summarized job description
            
        Returns:
            List of evaluation results sorted by match percentage (highest first)
        """
        if not processed_profiles:
            logger.info("No profiles to evaluate")
            return []
        
        if not raw_job_description:
            logger.info("No job description provided for evaluation")
            return []
        
        evaluation_results = []
        
        # Process each profile individually
        for profile in processed_profiles:
            try:
                logger.info(f"Evaluating profile {profile.get('profile_id', 'unknown')}")
                result = self.evaluate_single_profile(
                    profile=profile,
                    raw_job_description=raw_job_description,
                    summarized_job_description=summarized_job_description
                )
                
                if result:
                    evaluation_results.append(result)
                else:
                    logger.warning(f"No evaluation result for profile {profile.get('profile_id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error evaluating profile {profile.get('profile_id', 'unknown')}: {str(e)}")
        
        # Sort results by match percentage (highest first)
        evaluation_results.sort(
            key=lambda x: x.get("overall_match", {}).get("percentage", 0), 
            reverse=True
        )
        
        return evaluation_results 