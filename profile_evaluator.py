import os
import json
import logging
import re
import argparse
import asyncio
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI, AsyncOpenAI
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
            
            # Initialize both sync and async OpenAI clients with X AI API base URL
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
            logger.info("Grok sync and async clients initialized successfully")
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
            
            # Initialize both sync and async OpenAI clients with X AI API base URL
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
            logger.info("Grok sync and async clients initialized successfully (fallback)")
        
        # Initialize job dimensions cache
        self.job_dimensions = None
        
        # System prompt for dimension extraction
        self.dimension_extraction_prompt = """
        You are an expert talent-acquisition specialist trained to analyze job descriptions and identify the key dimensions that should be used to evaluate candidates.

        TASK  
        • Read the job description provided in the user message.  
        • Return exactly **3-5** job-specific evaluation dimensions.  
        • ALWAYS include one dimension named "Role Alignment".

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
        
        # Default batch size for parallel processing (can be overridden)
        self.batch_size = 6
    
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
        try:
            logger.debug(f"Calling Grok API with system prompt length: {len(system_prompt)}")
            logger.debug(f"Calling Grok API with user prompt length: {len(user_prompt)}")
            
            if not self.api_key:
                raise ValueError("XAI_API_KEY not found. Cannot call Grok API.")
        
            response = self.client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not response or not response.choices or len(response.choices) == 0:
                logger.error("Invalid response structure from Grok API")
                return ""
            
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty content in Grok API response")
                return ""
                
            logger.debug(f"Grok API response length: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"Error calling Grok API: {str(e)}")
            logger.exception("Full Grok API error details:")
            return ""
    
    async def _call_grok_llm_async(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """
        Call Grok LLM API asynchronously.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        try:
            logger.debug(f"Calling Grok API async with system prompt length: {len(system_prompt)}")
            logger.debug(f"Calling Grok API async with user prompt length: {len(user_prompt)}")
            
            if not self.api_key:
                raise ValueError("XAI_API_KEY not found. Cannot call Grok API.")
        
            # Initialize async_client if it doesn't exist (for cached instances)
            if not hasattr(self, 'async_client') or self.async_client is None:
                logger.info("Initializing async_client for cached Grok instance")
                self.async_client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
                logger.info("Async_client initialized successfully for cached instance")
        
            response = await self.async_client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not response or not response.choices or len(response.choices) == 0:
                logger.error("Invalid response structure from Grok API async")
                return ""
            
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty content in Grok API async response")
                return ""
                
            logger.debug(f"Grok API async response length: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"Error calling Grok API async: {str(e)}")
            logger.exception("Full Grok API async error details:")
            return ""
    
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
            model="models/gemini-2.5-pro-preview-03-25",
            contents=combined_prompt,
        )
        
        return response.text
    
    async def _call_gemini_llm_async_wrapper(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
        """
        Call Gemini LLM API with asyncio compatibility.
        This is a wrapper around the synchronous method to make it work with asyncio.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        # Use run_in_executor to run the synchronous method in a thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._call_gemini_llm(system_prompt, user_prompt, temperature, max_tokens)
        )
    
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
            logger.debug(f"Job description prompt length: {len(job_description_prompt) if job_description_prompt else 0}")
            
            if self.llm_choice == "grok":
                # Try with different temperature for Grok
                llm_response = self._call_grok_llm(
                    system_prompt=self.dimension_extraction_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,  # Lower temperature for more consistent JSON
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
            logger.debug(f"Response length: {len(llm_response) if llm_response else 0}")
            
            if not llm_response or len(llm_response.strip()) == 0:
                logger.error("Empty response from LLM")
                return {"dimensions": []}
            
            # Extract JSON from the response
            extracted_dimensions = self._extract_json_from_response(llm_response)
            
            # Validate dimensions format
            if not extracted_dimensions:
                logger.error("Failed to extract any JSON from LLM response")
                return {"dimensions": []}
            
            if "dimensions" not in extracted_dimensions:
                logger.error(f"No 'dimensions' key in extracted JSON. Keys found: {list(extracted_dimensions.keys())}")
                return {"dimensions": []}
                
            if not isinstance(extracted_dimensions["dimensions"], list):
                logger.error(f"'dimensions' is not a list. Type: {type(extracted_dimensions['dimensions'])}")
                return {"dimensions": []}
            
            if len(extracted_dimensions["dimensions"]) == 0:
                logger.error("Empty dimensions list extracted")
                return {"dimensions": []}
                
            # Log extracted dimensions for debugging
            logger.info(f"Extracted {len(extracted_dimensions['dimensions'])} dimensions:")
            for i, dim in enumerate(extracted_dimensions["dimensions"]):
                logger.debug(f"Dimension {i+1}: {dim.get('name', 'NO_NAME')} - {dim.get('description', 'NO_DESC')[:50]}...")
                
            # Ensure Role Alignment dimension is included
            has_role_alignment = any(
                dim.get("name", "").lower() == "role alignment" 
                for dim in extracted_dimensions["dimensions"]
            )
            
            if not has_role_alignment:
                logger.warning("Role Alignment dimension not found, adding it manually")
                extracted_dimensions["dimensions"].append({
                    "id": "role_alignment",
                    "name": "Role Alignment",
                    "weight": 20,  # Add default weight
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
            logger.exception("Full exception details:")
            return {"dimensions": []}
    
    def evaluate_profile(
        self,
        profile_data: Dict[str, Any],
        raw_job_description: str,
        dimensions: List[Dict[str, Any]],
        profile_id: str,
        summarized_job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single profile against extracted job dimensions.
        
        Args:
            profile_data: Processed profile data
            raw_job_description: Full job description
            dimensions: Extracted job dimensions
            profile_id: Profile ID
            summarized_job_description: Optional summarized/edited job description
            
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
            
            # Use summarized job description if available, otherwise fall back to raw
            job_description_to_use = summarized_job_description if summarized_job_description else raw_job_description
            
            # Format the user prompt
            user_prompt = f"""
            Your task is to evaluate this candidate profile against the job requirements and dimensions.

            --- JOB DESCRIPTION ---
            {job_description_to_use}
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

    async def evaluate_profile_async(
        self,
        profile_data: Dict[str, Any],
        raw_job_description: str,
        dimensions: List[Dict[str, Any]],
        profile_id: str,
        summarized_job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single profile against extracted job dimensions asynchronously.
        
        Args:
            profile_data: Processed profile data
            raw_job_description: Full job description
            dimensions: Extracted job dimensions
            profile_id: Profile ID
            summarized_job_description: Optional summarized/edited job description
            
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
            
            # Use summarized job description if available, otherwise fall back to raw
            job_description_to_use = summarized_job_description if summarized_job_description else raw_job_description
            
            # Format the user prompt
            user_prompt = f"""
            Your task is to evaluate this candidate profile against the job requirements and dimensions.

            --- JOB DESCRIPTION ---
            {job_description_to_use}
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
            logger.info(f"Calling {self.llm_choice.upper()} LLM to evaluate profile {profile_id} (async)")
            
            if self.llm_choice == "grok":
                # Use the true async method for Grok
                llm_response = await self._call_grok_llm_async(
                    system_prompt=self.profile_evaluation_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192
                )
            else:  # gemini
                # Use the async wrapper for Gemini
                llm_response = await self._call_gemini_llm_async_wrapper(
                    system_prompt=self.profile_evaluation_prompt,
                    user_prompt=user_prompt,
                    temperature=0.3,
                    max_tokens=8192
                )
            
            logger.info(f"Profile evaluation response received from {self.llm_choice.upper()} LLM for profile {profile_id} (async)")
            
            # Extract JSON from the response
            evaluation_results = self._extract_json_from_response(llm_response)
            
            # Ensure profile_id is included
            if "profile_id" not in evaluation_results:
                evaluation_results["profile_id"] = profile_id
                
            return evaluation_results
                
        except Exception as e:
            logger.error(f"Error evaluating profile {profile_id} asynchronously: {str(e)}")
            return {"profile_id": profile_id, "error": str(e)}
    
    async def evaluate_profiles_parallel(
        self,
        processed_profiles: List[Dict[str, Any]],
        raw_job_description: str,
        dimensions: List[Dict[str, Any]],
        batch_size: int = 6,
        summarized_job_description: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple profiles in parallel using async.
        
        Args:
            processed_profiles: List of processed profile data
            raw_job_description: Full job description
            dimensions: Extracted job dimensions
            batch_size: Number of profiles to evaluate concurrently
            summarized_job_description: Optional summarized/edited job description
            
        Returns:
            List of evaluation results for each profile
        """
        if not processed_profiles:
            logger.info("No profiles to evaluate in parallel")
            return []
            
        logger.info(f"Evaluating {len(processed_profiles)} profiles in parallel with batch size {batch_size}")
        
        # Split profiles into batches
        all_results = []
        for i in range(0, len(processed_profiles), batch_size):
            batch = processed_profiles[i:i+batch_size]
            logger.info(f"Processing batch of {len(batch)} profiles (batch {i//batch_size + 1})")
            
            # Create tasks for each profile in the batch
            tasks = []
            for profile in batch:
                profile_id = profile.get("profile_id", "unknown")
                logger.info(f"Creating async task for profile {profile_id}")
                
                task = self.evaluate_profile_async(
                    profile_data=profile,
                    raw_job_description=raw_job_description,
                    dimensions=dimensions,
                    profile_id=profile_id,
                    summarized_job_description=summarized_job_description
                )
                tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            logger.info(f"Completed batch of {len(batch)} profiles")
        
        return all_results
    
    def evaluate_profiles(
        self,
        processed_profiles: List[Dict[str, Any]],
        raw_job_description: str,
        summarized_job_description: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple profiles against job description.
        
        Args:
            processed_profiles: List of processed profile data
            raw_job_description: Full job description
            summarized_job_description: Optional summarized job description
            batch_size: Optional batch size override (default: uses self.batch_size)
            
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
            # Use provided batch_size or fall back to self.batch_size
            actual_batch_size = batch_size if batch_size is not None else self.batch_size
            
            # Extract job dimensions if not already cached
            if not self.job_dimensions:
                logger.info("Extracting job dimensions")
                # Use summarized job description for extracting dimensions if available
                job_description_for_dimensions = summarized_job_description if summarized_job_description else raw_job_description
                self.job_dimensions = self.extract_job_dimensions(raw_job_description, job_description_for_dimensions)
            
            dimensions = self.job_dimensions.get("dimensions", [])
            if not dimensions:
                logger.error("Failed to extract job dimensions")
                return {"profiles": []}
            
            # Both Grok and Gemini now support parallel evaluation
            logger.info(f"Evaluating {len(processed_profiles)} profiles in parallel with batch size {actual_batch_size} using {self.llm_choice.upper()}")
            # We need to run the async code in an event loop
            evaluated_profiles = asyncio.run(self.evaluate_profiles_parallel(
                processed_profiles=processed_profiles,
                raw_job_description=raw_job_description,
                dimensions=dimensions,
                batch_size=actual_batch_size,
                summarized_job_description=summarized_job_description
            ))
                
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
        custom_query: str,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate profiles based on custom query.
        
        Args:
            processed_profiles: List of processed profile data
            custom_query: Custom query text
            batch_size: Optional batch size override (default: uses self.batch_size)
            
        Returns:
            Dictionary containing evaluation results for all profiles
        """
        # Reset dimensions cache for new query
        self.job_dimensions = None
        
        # Use the same evaluation method but with custom query as job description
        return self.evaluate_profiles(processed_profiles, custom_query, batch_size=batch_size)
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw {self.llm_choice.upper()} response: {response_text[:500]}...")
            
            # Remove thinking part and other common patterns
            cleaned_response = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
            cleaned_response = re.sub(r"<thinking>.*?</thinking>", "", cleaned_response, flags=re.DOTALL)
            
            # Try multiple JSON extraction methods
            json_str = None
            
            # Method 1: Try to find JSON in code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned_response)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.debug("Found JSON in code blocks")
            
            # Method 2: If no code blocks, try to find JSON using braces with better matching
            if not json_str:
                # Find all potential JSON objects
                brace_matches = []
                depth = 0
                start_idx = -1
                
                for i, char in enumerate(cleaned_response):
                    if char == '{':
                        if depth == 0:
                            start_idx = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start_idx >= 0:
                            brace_matches.append(cleaned_response[start_idx:i+1])
                
                # Try the longest match first
                if brace_matches:
                    json_str = max(brace_matches, key=len)
                    logger.debug("Found JSON using brace matching")
            
            # Method 3: Try to clean and extract JSON more aggressively
            if not json_str:
                # Remove common non-JSON prefixes/suffixes
                cleaned = cleaned_response.strip()
                patterns_to_remove = [
                    r"^.*?(?=\{)",  # Remove everything before first {
                    r"\}.*?$",      # Remove everything after last }
                ]
                
                for pattern in patterns_to_remove:
                    cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
                
                if cleaned.startswith('{') and cleaned.endswith('}'):
                    json_str = cleaned
                    logger.debug("Found JSON using aggressive cleaning")
            
            # Final fallback: use the entire cleaned response
            if not json_str:
                json_str = cleaned_response.strip()
                logger.debug("Using entire cleaned response as JSON")
            
            # Log what we're trying to parse
            logger.debug(f"Attempting to parse JSON: {json_str[:200]}...")
            
            # Parse JSON
            result = json.loads(json_str)
            logger.info(f"Successfully parsed JSON from {self.llm_choice.upper()} response")
            return result
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.llm_choice.upper()} response: {str(e)}")
            logger.error(f"Attempted JSON string: {json_str[:500] if json_str else 'None'}...")
            logger.error(f"Full response: {response_text}")
            # Return empty dict on failure
            return {}

def parse_args():
    """Parse command-line arguments for CLI operation."""
    parser = argparse.ArgumentParser(description="Profile Evaluator CLI Tool")
    parser.add_argument("--llm", choices=["grok", "gemini"], default="grok", help="LLM to use for evaluation")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for parallel processing (default: 6)")
    parser.add_argument("--job-description", type=str, required=True, help="Path to job description file")
    parser.add_argument("--profiles", type=str, required=True, help="Path to profiles JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info(f"Starting Profile Evaluator with {args.llm} LLM")
    
    # Initialize evaluator
    evaluator = IndividualProfileEvaluator(llm_choice=args.llm)
    evaluator.batch_size = args.batch_size
    
    # Load job description
    with open(args.job_description, 'r', encoding='utf-8') as f:
        job_description = f.read()
    
    # Load profiles
    with open(args.profiles, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    # Evaluate profiles
    results = evaluator.evaluate_profiles(profiles, job_description, batch_size=args.batch_size)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {args.output}") 