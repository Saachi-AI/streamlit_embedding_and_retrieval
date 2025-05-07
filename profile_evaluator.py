import os
import json
import logging
import re
import argparse
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndividualProfileEvaluator:
    """
    Class to evaluate individual profiles using LLM based on job description.
    This class extracts dimensions from job descriptions and provides detailed
    evaluation of each candidate against these dimensions.
    """
    
    def __init__(self, api_key: str = None, llm_choice: str = "grok"):
        """Initialize with X AI API key or Google API key based on llm_choice."""
        self.llm_choice = llm_choice.lower()
        
        if self.llm_choice == "grok":
            self.api_key = api_key or os.environ.get("XAI_API_KEY")
            if not self.api_key:
                logger.warning("XAI_API_KEY not found in environment. Grok LLM evaluation will not work.")
            
            # Initialize the OpenAI client with X AI API base URL
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        elif self.llm_choice == "gemini":
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not found in environment. Gemini LLM evaluation will not work.")
            else:
                # Log that we found the API key (safely)
                logger.info(f"Using Google API key (starting with: {self.api_key[:4]}{'*' * 10})")
            
            # Initialize the Gemini client 
            # Note: The client directly takes the API key, no need for configure method
            self.client = genai.Client(api_key=self.api_key)
        else:
            logger.warning(f"Unknown LLM choice: {llm_choice}. Defaulting to Grok.")
            self.llm_choice = "grok"
            self.api_key = api_key or os.environ.get("XAI_API_KEY")
            if not self.api_key:
                logger.warning("XAI_API_KEY not found in environment. Grok LLM evaluation will not work.")
            
            # Initialize the OpenAI client with X AI API base URL
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        
        # Initialize job dimensions cache
        self.job_dimensions = None
        
        # System prompt for dimension extraction
        self.dimension_extraction_prompt = """
        You are an expert talent-acquisition specialist trained to analyze job descriptions and identify the key dimensions that should be used to evaluate candidates.

        TASK  
        • Read the job description provided in the user message.  
        • Return exactly **3-5** job-specific evaluation dimensions.  
        • ALWAYS include one dimension named “Role Alignment”.

        RULES:
        1. Identify 3-5 job-specific evaluation dimensions based on the job description.
        2. Each dimension must be returned with four fields:  
            • "id" - concise snake_case slug (≤ 30 chars) used as a stable reference  
            • "name" - human-readable title  
            • "weight" - integer 0-100; all weights must sum to 100  
            • "description" - one-sentence explanation of what this dimension evaluates  
            • "key_success_factors" - 2-3 bullet points defining excellence
        3. Use weights to reflect relative importance in the JD (higher weight = more critical).
        4. ALWAYS include "Role Alignment" as one of the dimensions to evaluate career fit.
        5. Focus on extracting dimensions related to:
           - Required skills and technical expertise
           - Domain/industry experience
           - Relevant qualifications or certifications
           - Soft skills or behavioral attributes mentioned
           - Language requirements if specified
        6. Dimensions should be specific enough to meaningfully differentiate candidates
        7. Do not include generic dimensions that apply to all jobs (e.g., "Communication Skills") unless specifically emphasized in the description

        OUTPUT FORMAT:
        Return a JSON object with the following structure:
        {
          "dimensions": [
            {
              "id": "dimension_name",
              "name": "Dimension Name",
              "weight": 20,
              "description": "Clear explanation of what this dimension evaluates",
              "key_success_factors": [
                "Success factor 1",
                "Success factor 2",
                "Success factor 3"
              ]
            },
            // Additional dimensions...
          ]
        }

        IMPORTANT: 
        - Do not include any text outside the JSON structure
        - Ensure the output is valid JSON
        - Make dimensions specific to this particular job
        - ALWAYS include "Role Alignment" as one of the dimensions
        """
        
        # System prompt for profile evaluation
        self.profile_evaluation_prompt = """
        You are an expert talent-acquisition specialist. Your task is to evaluate a candidate profile against predefined job-specific dimensions and deliver a structured JSON assessment.

        EVALUATION RULES:
        1. Each dimension arrives with an "id", "name", and "weight" (0-100), "description" & "key_success_factors".
            - If any weight is absent, assume all dimensions are equally weighted.   
        2. For every dimension:
            - Assign a score from 0-100%.
            - Provide concise reasoning (MAX 60 words) citing concrete evidence from the profile.
            - Ignore buzz-words or generic soft-skill claims unless the profile provides verifiable proof.  
        3. Compute an overall match percentage = weighted average of the dimension scores.  
        4. Categorize the candidate as:
           - "Excellent Match" (85-100%)
           - "Good Match" (70-84%)
           - "Fair Match" (50-69%)
           - "Poor Match" (0-49%)
        5. Identify if the candidate is overqualified for the role and if the candidate is clearly far more senior than required, set `"is_overqualified": true`.
        6. Highlight 2-3 key strengths and 1-2 key gaps

        OVERQUALIFICATION ASSESSMENT:
        - If the profile indicates the candidate is significantly more senior than required (e.g., CEO applying for developer position), flag this with specific reasoning
        - Adjust the overall score downward for significant overqualification, as these candidates are less likely to be satisfied in the role
        - Consider title history, years of experience, and level of previous responsibilities

        OUTPUT FORMAT:
        Return **valid JSON only**, exactly in this schema:
        {
          "profile_id": "candidate's profile ID",
          "dimensions": [
            {
              "id": "dimension_name",
              "name": "Dimension Name",
              "score": 85,
              "reasoning": "Detailed explanation of why this score was assigned, referencing specific aspects of the candidate's profile."
            },
            // Additional dimensions...
          ],
          "overall_match": {
            "percentage": 78,
            "category": "Good Match",
            "reasoning": "Overall assessment of why the candidate received this score and category"
          },
          "is_overqualified": false,  // or true with reasoning if applicable
          "overqualification_reasoning": "",  // Only populated if is_overqualified is true
          "key_strengths": [
            "Strength 1 with specific evidence",
            "Strength 2 with specific evidence",
            "Strength 3 with specific evidence"
          ],
          "key_gaps": [
            "Gap 1 with specific evidence",
            "Gap 2 with specific evidence"
          ]
        }

        IMPORTANT: 
        - Do not include any text outside the JSON structure
        - Ensure the output is valid JSON
        - Provide specific evidence from the profile for each score and assessment
        - Be thorough in your reasoning, explaining exactly why scores were assigned
        - Ensure the overall match percentage is a weighted average of the dimension scores
        """
    
    def _call_grok_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """
        Call Grok LLM API.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        response = self.client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _call_gemini_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """
        Call Gemini LLM API.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        # Combine system prompt and user prompt for Gemini
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate content with parameters directly
        response = self.client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=combined_prompt,
        )
        
        return response.text
    
    def extract_job_dimensions(self, raw_job_description: str, job_description_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract key dimensions from job description for candidate evaluation.
        
        Args:
            raw_job_description: Full job description text
            summarized_job_description: Optional summarized job description
            
        Returns:
            Dictionary containing extracted dimensions
        """
            
        try:
            user_prompt = f"""
            Please analyze the following summarized job description and identify 3-5 key dimensions for candidate evaluation, following the specification in the system prompt.

            --- SUMMARIZED JOB DESCRIPTION ---
            {job_description_prompt}
            --- END SUMMARIZED JOB DESCRIPTION ---
            """

            # Call LLM to extract dimensions
            logger.info(f"Calling {self.llm_choice.upper()} LLM to extract job dimensions")
            
            if self.llm_choice == "grok":
                llm_response = self._call_grok_llm(
                    system_prompt=self.dimension_extraction_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=4096
                )
            else:  # gemini
                llm_response = self._call_gemini_llm(
                    system_prompt=self.dimension_extraction_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=4096
                )
            
            logger.info(f"Job dimension extraction response received from {self.llm_choice.upper()} LLM")
            
            # Extract JSON from the response
            extracted_dimensions = self._extract_json_from_response(llm_response)
            
            # Validate dimensions format
            if "dimensions" not in extracted_dimensions or not isinstance(extracted_dimensions["dimensions"], list):
                logger.error("Invalid format in extracted dimensions response")
                return {"dimensions": []}
                
            # Ensure Role Alignment dimension is included
            has_role_alignment = any(
                dim.get("name", "").lower() == "role alignment" 
                for dim in extracted_dimensions["dimensions"]
            )
            
            if not has_role_alignment:
                logger.warning("Role Alignment dimension not found, adding it manually")
                extracted_dimensions["dimensions"].append({
                    "name": "Role Alignment",
                    "description": "Evaluates how well the candidate's career trajectory, seniority, and aspirations align with this specific role",
                    "key_success_factors": [
                        "Appropriate seniority level for the position",
                        "Career trajectory suggests interest in this type of role",
                        "Not significantly overqualified or underqualified"
                    ]
                })
                
            logger.info(f"Successfully extracted {len(extracted_dimensions['dimensions'])} dimensions from job description")
            return extracted_dimensions
                
        except Exception as e:
            logger.error(f"Error extracting job dimensions: {str(e)}")
            return {"dimensions": []}
    
    def evaluate_profile(
        self,
        profile_data: Dict[str, Any],
        raw_job_description: str,
        dimensions: List[Dict[str, Any]],
        profile_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single profile against extracted job dimensions.
        
        Args:
            profile_data: Processed profile data
            raw_job_description: Full job description
            dimensions: Extracted job dimensions
            profile_id: Profile ID
            
        Returns:
            Dictionary containing detailed evaluation results
        """
        if not profile_data or not dimensions:
            logger.error("Missing profile data or dimensions for evaluation")
            return {}
            
        try:
            # Format dimensions as string for the prompt
            dimensions_str = json.dumps(dimensions, ensure_ascii=False)
            
            # Format profile data as string
            profile_str = json.dumps(profile_data, ensure_ascii=False)
            
            # Format the user prompt
            user_prompt = f"""
            Your task is to evaluate this candidate profile against the job requirements and dimensions.

            --- JOB DESCRIPTION ---
            {raw_job_description}
            --- END JOB DESCRIPTION ---

            --- EVALUATION DIMENSIONS ---
            {dimensions_str}
            --- END EVALUATION DIMENSIONS ---

            --- CANDIDATE PROFILE ---
            {profile_str}
            --- END CANDIDATE PROFILE ---

            Please evaluate this candidate (profile_id: {profile_id}) against each dimension, providing percentage scores, reasoning, and an overall assessment.
            """
            
            # Call LLM for profile evaluation
            logger.info(f"Calling {self.llm_choice.upper()} LLM to evaluate profile {profile_id}")
            
            if self.llm_choice == "grok":
                llm_response = self._call_grok_llm(
                    system_prompt=self.profile_evaluation_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192
                )
            else:  # gemini
                llm_response = self._call_gemini_llm(
                    system_prompt=self.profile_evaluation_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192
                )
            
            logger.info(f"Profile evaluation response received from {self.llm_choice.upper()} LLM for profile {profile_id}")
            
            # Extract JSON from the response
            evaluation_results = self._extract_json_from_response(llm_response)
            
            # Ensure profile_id is included
            if "profile_id" not in evaluation_results:
                evaluation_results["profile_id"] = profile_id
                
            return evaluation_results
                
        except Exception as e:
            logger.error(f"Error evaluating profile {profile_id}: {str(e)}")
            return {"profile_id": profile_id, "error": str(e)}
    
    def evaluate_profiles(
        self,
        processed_profiles: List[Dict[str, Any]],
        raw_job_description: str,
        summarized_job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple profiles against job description.
        
        Args:
            processed_profiles: List of processed profile data
            raw_job_description: Full job description
            summarized_job_description: Optional summarized job description
            
        Returns:
            Dictionary containing evaluation results for all profiles
        """
        if not processed_profiles:
            logger.info("No profiles to evaluate")
            return {"profiles": []}
            
        if not raw_job_description:
            logger.info("Missing job description - skipping evaluation")
            return {"profiles": []}
            
        try:
            # Extract job dimensions if not already cached
            if not self.job_dimensions:
                logger.info("Extracting job dimensions")
                self.job_dimensions = self.extract_job_dimensions(raw_job_description, summarized_job_description)
            
            dimensions = self.job_dimensions.get("dimensions", [])
            if not dimensions:
                logger.error("Failed to extract job dimensions")
                return {"profiles": []}
                
            # Evaluate each profile
            evaluated_profiles = []
            for profile in processed_profiles:
                profile_id = profile.get("profile_id", "unknown")
                logger.info(f"Evaluating profile {profile_id}")
                
                evaluation = self.evaluate_profile(
                    profile_data=profile,
                    raw_job_description=raw_job_description,
                    dimensions=dimensions,
                    profile_id=profile_id
                )
                
                evaluated_profiles.append(evaluation)
                
            # Sort profiles: First by overqualification status, then by match percentage (descending)
            evaluated_profiles.sort(
                key=lambda x: (
                    # Sort overqualified candidates after non-overqualified candidates
                    x.get("is_overqualified", False),
                    # Then by overall match percentage (descending)
                    -x.get("overall_match", {}).get("percentage", 0)
                )
            )
            
            # Add ranks based on sorted order
            for i, profile in enumerate(evaluated_profiles):
                profile["rank"] = i + 1
                
            return {
                "job_dimensions": dimensions,
                "profiles": evaluated_profiles
            }
                
        except Exception as e:
            logger.error(f"Error in profile evaluation: {str(e)}")
            return {"profiles": []}
    
    def evaluate_profiles_custom_query(
        self,
        processed_profiles: List[Dict[str, Any]],
        custom_query: str
    ) -> Dict[str, Any]:
        """
        Evaluate profiles based on custom query.
        
        Args:
            processed_profiles: List of processed profile data
            custom_query: Custom query text
            
        Returns:
            Dictionary containing evaluation results for all profiles
        """
        # Reset dimensions cache for new query
        self.job_dimensions = None
        
        # Use the same evaluation method but with custom query as job description
        return self.evaluate_profiles(processed_profiles, custom_query)
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Remove thinking part
            cleaned_response = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
            
            # Try to find JSON in code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned_response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code blocks, try to find JSON using braces
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = cleaned_response[start_idx:end_idx]
                else:
                    # If no JSON found, use the entire cleaned response
                    json_str = cleaned_response.strip()
            
            # Parse the JSON
            parsed_json = json.loads(json_str)
            return parsed_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            logger.debug(f"Problematic response: {response_text[:1000]}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting JSON from LLM response: {str(e)}")
            return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Profile Evaluator using LLM")
    parser.add_argument(
        "--llm",
        type=str,
        default="grok",
        choices=["grok", "gemini"],
        help="LLM model to use (default: grok)"
    )
    parser.add_argument(
        "--job",
        type=str,
        help="Path to job description file"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        help="Path to profiles file or directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = IndividualProfileEvaluator(llm_choice=args.llm)
    print(f"Initialized profile evaluator with LLM: {args.llm}")
    print("Note: For full functionality, use this class within your application.")
    print("For streamlit integration, pass LLM choice via config or environment variables.") 