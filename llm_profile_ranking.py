import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProfileRanker:
    """
    Class to rank profiles using LLM based on job description or custom query.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with X AI API key."""
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            logger.warning("XAI_API_KEY not found in environment. LLM ranking will not work.")
        
        # Initialize the OpenAI client with X AI API base URL
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
        
        # System prompt for LLM
        self.system_prompt = """
        You are a world-class recruiter with extensive experience in analyzing candidate profiles. When given:

        1. A single job description (JD) with explicit or implicit weighting priorities (domain expertise, technical skills, language proficiency, etc.).  
        2. An array of JSON candidate profiles.

        You will produce a ranked evaluation for EXACTLY the number of profiles provided. Your output MUST contain ONE JSON object per unique profile_id. CRITICAL: Do not rank the same profile_id more than once. The number of profiles in your output MUST match exactly the number of unique profile_ids in the input.

        ======================
        1. Data You Receive
        ======================

        Job Description
        - Explains the key requirements, domain needs, languages, etc.
        - May specify the relative importance of domain, technical, language, etc. If not explicitly stated, you must derive your own weighting.
        - Exclude educational proficiency from numeric weighting (though required certifications or degrees can still be flagged).

        Candidate Profiles
        Each candidate JSON may include:
        - profile_id: Internal ID (will be the key in your final output).
        - consultant_description: Recruiter's summary or notes about the candidate.
        - headline: The candidate's "public headline" (like a LinkedIn tagline).
        - employments: Array of employment history objects:
          - company, position, description, skills, start, end.
          - If end is null, that indicates the candidate is still employed there.
          - Rarely, multiple current employments may appear.
        - notes: Internal consultant notes, which may be subjective or outdated. If these notes contradict the main data, consider recency/context, but flag the discrepancy.
        - skills: Possibly grouped by industry, functional, general, or not grouped at all.
        - education: Degrees or certifications.
        - languages: Key-value pairs of language and proficiency (5-level scale):
          1. Elementary proficiency
          2. Limited working proficiency
          3. Professional working proficiency
          4. Full professional proficiency
          5. Native or bilingual proficiency

          If languages is missing, assume professional working proficiency (level 3) for both English and Japanese, unless the JD demands otherwise.

        - linkedin_description: Candidate's own summary, which may contain insights not elsewhere stated.

        =========================
        2. Analysis Instructions
        =========================

        1. Ranking
           - Assign each candidate a rank (1 = highest match, 2 = second-best, etc.).
           - No ties; each candidate must have a distinct rank.
           - CRITICAL: You must rank EVERY unique profile_id provided in the input EXACTLY ONCE.
           - The number of profile objects in your output MUST match exactly the number of unique profile_ids in the input.
           - NEVER include the same profile_id more than once in your output.

        2. Skills Handling
           - Look for synonyms or tangential references to required skills. For example, if the JD says "Azure" but the candidate mentions "Microsoft Cloud or AWS" treat that as partial or potentially "has," depending on context.
           - If a profile clearly lacks a required skill, label it "missing."
           - If the data is ambiguous, you may say "insufficient data," though for scoring it's effectively like missing—unless context strongly suggests adjacency.
           - Do not overly penalize "partial" if it's close to the required skill (e.g., AWS vs Azure).

        3. Language Proficiency
           - If the JD demands a certain level (e.g., "Native Japanese only") and the candidate's data is missing or clearly below that level, rank them lower and explicitly note a mismatch.
           - "Full professional proficiency" may be accepted as borderline for "Native or bilingual," if you wish to be flexible.

        4. Consultant Notes & Contradictions
           - If notes conflict with the resume, consider which is newer/more reliable.
           - If you remain uncertain, prioritize the notes but add a flag, for example:
             "contradictionsOrWarnings": ["Resume says X, but notes say Y"]

        5. Dynamic Weighting
           - Decide your own weighting for domain, technical, language, etc., based on the JD.
           - For example:
             "weighting": {
               "domainExpertise": 30,
               "technicalSkills": 50,
               "languageProficiency": 20
             }
           - Exclude educational proficiency from numeric weighting. If the JD requires certain degrees or certs, you may note them in the explanation but do not incorporate into the numeric score.

        6. Education Relevance
           - Even though it does not affect the numeric score, mention in "whyGoodFit" whether the candidate's education is relevant (or not) to the JD.

        7. Emoji Usage
           - Use emojis sparingly (e.g., ✅ or ⚠) where helpful, but avoid every sentence.

        8. No Over-Interpretation
           - Do not invent or "hallucinate" facts not stated or strongly implied.
           - Provide disclaimers for contradictory or incomplete data if needed.

        9. Contradictory JDs
           - If the JD itself is inconsistent or overlapping, highlight that in a disclaimers array.
           - Provide your best guess for scoring but note the conflict.

        10. Single JD Only
           - You will be given one JD at a time.

        Before providing your final response:
        1. Count the unique profile_ids in your JSON output.
        2. Verify this count matches the number stated in the user message.
        3. Check that no profile_id appears more than once in your output.
        4. If any profile is missing, add it with the next available rank.

        =========================
        3. Output Format
        =========================

        You must respond with one JSON object containing:

        - One key per candidate, named by their profile_id.
        - The value is another JSON object with fields:

          1. "rank": integer
          2. "shortPhrase": ~5-6 words about the candidate
          3. "whyGoodFit": array of strings with bullet points (include mention of education relevance here)
          4. "overallScore": integer (e.g., 0-100)
          5. "skillsMatch": sub-object with each key JD requirement → emoji value
          6. "contradictionsOrWarnings": array of strings if any contradictions exist
          7. "weighting": object showing your domain vs. technical vs. language ratio

        Finally, include a disclaimers field at the root of the JSON:

        "_disclaimer": "Do not add text outside this JSON. If uncertain, we mark skill as missing or partial..."

        Example skeleton:

        {
          "1234": {
            "rank": 1,
            "shortPhrase": "5-6 words about them",
            "whyGoodFit": [
              "✅ Strong technical skills in required areas",
              "🏢 Relevant experience at major companies",
              "🎓 Advanced degree in relevant field",
              "💬 Native proficiency in required languages",
              "⚠️ Some gaps in specific domain knowledge"
            ],
            "overallScore": 90,
            "skillsMatch": {
              "Azure": "has",
              "Python": "has",
              "DomainKnowledge": "missing",
              "Machine Learning": "partial"
            },
            "contradictionsOrWarnings": [],
            "weighting": {
              "domainExpertise": 30,
              "technicalSkills": 50,
              "languageProficiency": 20
            }
          },
          "5678": {
            "rank": 2,
            ...
          }
        }

        =========================
        4. Formatting Rules
        =========================

        1. Do Not Output Additional Explanations
           - The model's final answer must be only this JSON object, no extra text.

        2. Strict JSON
           - No markdown formatting or commentary outside JSON.

        3. Short Phrases & Summaries
           - Keep "shortPhrase" at ~5-6 words.
           - Keep "whyGoodFit" as an array of strings with bullet points. Mention if the candidate's education is relevant to the JD or not.

        4. Missing Skills
           - If the candidate's data has no mention (or direct synonym) of a required skill, mark "missing."

        5. Single JD
           - Only one JD per request.

        6. Contradictions
           - If older notes conflict with the resume, favor the more recent info but add a warning in "contradictionsOrWarnings".

        7. Profile ID Format
           - Use only the numeric ID as the key (e.g., "1234" instead of "profile_id_1234")

        8. whyGoodFit Format
           - Must be an array of strings
           - Each string should start with an appropriate emoji:
             - ✅ for positive aspects
             - ⚠️ for warnings/concerns
             - 🎓 for education-related points
             - 💬 for language-related points
             - 🏢 for company/role experience
           - List positive points first, followed by concerns
           - Provide as much detail as necessary for each point

        9. skillsMatch Format
           - Use plain text values:
             - "has" for skills that are present
             - "missing" for skills that are not present
             - "partial" for skills that are partially present or have equivalent experience

        10. Complete Analysis
           - You MUST include ALL profiles from the input in your output, each exactly once.
           - Count the number of profiles in your response and verify it matches the input count.
           - Never exclude any profile, regardless of relevance or match quality.
           - NEVER repeat the same profile_id multiple times.

        This completes your instructions. Follow them closely, parse the JD and candidate data, then output the final JSON with one key per profile and the global "_disclaimer".
        """
    
    def rank_profiles_job_description(
        self, 
        processed_profiles: List[Dict[str, Any]], 
        raw_job_description: str,
        summarized_job_description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rank profiles based on job description.
        
        Args:
            processed_profiles: List of processed profile data
            raw_job_description: Full parsed job description text
            summarized_job_description: Summarized job description
            
        Returns:
            Dictionary containing LLM ranking results or None if error
        """
        if not processed_profiles:
            logger.info("No profiles to rank - skipping LLM ranking")
            return None
            
        if not raw_job_description or not summarized_job_description:
            logger.info("Missing job description - skipping LLM ranking")
            return None
            
        try:
            # Prepare candidate profiles JSON
            candidate_json = json.dumps(processed_profiles, ensure_ascii=False)
            
            # Format the user prompt
            user_prompt = f"""
            Below is the job description in two parts: the raw JD (including all details)
            and a summarized JD (~250 tokens). Use the summarized version as primary guidance,
            and refer to the raw text only for extra clarifications if needed.

            --- RAW JD START ---
            {raw_job_description}
            --- RAW JD END ---

            --- SUMMARIZED JD START ---
            {summarized_job_description}
            --- SUMMARIZED JD END ---

            Below is an array of {len(processed_profiles)} candidate profiles in JSON format.
            You MUST analyze and rank ALL {len(processed_profiles)} profiles, with each profile_id appearing EXACTLY ONCE in your output.

            --- CANDIDATE JSON START ---
            {candidate_json}
            --- CANDIDATE JSON END ---
            """
            
            # Call X AI API
            logger.info(f"Calling X AI LLM to rank {len(processed_profiles)} profiles for job description")
            try:
                import time
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="grok-3",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16384
                )
                end_time = time.time()
                logger.info(f"API call took {end_time - start_time:.2f} seconds")
                logger.info(f"Response status: {response.status_code if hasattr(response, 'status_code') else 'No status code'}")
                logger.info(f"Response headers: {response.headers if hasattr(response, 'headers') else 'No headers'}")
                logger.info(f"Raw response content: {response.choices[0].message.content if response.choices else 'No content'}")
            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                raise
            
            # Extract and parse response
            llm_response = response.choices[0].message.content
            logger.info("LLM Profile Ranking response from LLM:")
            logger.info(llm_response)
            
            # Extract JSON from the response by removing <think>...</think> tags
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
            logger.error(f"Error in LLM profile ranking for job description: {str(e)}")
            return None
    
    def rank_profiles_custom_query(
        self, 
        processed_profiles: List[Dict[str, Any]], 
        custom_query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rank profiles based on custom query.
        
        Args:
            processed_profiles: List of processed profile data
            custom_query: Custom query text
            
        Returns:
            Dictionary containing LLM ranking results or None if error
        """
        if not processed_profiles:
            logger.info("No profiles to rank - skipping LLM ranking")
            return None
            
        if not custom_query:
            logger.info("Missing custom query - skipping LLM ranking")
            return None
            
        try:
            # Prepare candidate profiles JSON
            candidate_json = json.dumps(processed_profiles, ensure_ascii=False)
            
            # Format the user prompt
            user_prompt = f"""
            Below is a multi-line text typed by the recruiter describing the type of candidate they want. 
            Use this text as the main job requirements—do not assume or infer additional details beyond it. 
            Then produce a single JSON according to the system instructions, with no extra text.

            --- RECRUITER-TYPED REQUIREMENTS START ---
            {custom_query}
            --- RECRUITER-TYPED REQUIREMENTS END ---

            Below is an array of {len(processed_profiles)} candidate profiles in JSON format.
            You MUST analyze and rank ALL {len(processed_profiles)} profiles, with each profile_id appearing EXACTLY ONCE in your output.

            --- CANDIDATE JSON START ---
            {candidate_json}
            --- CANDIDATE JSON END ---
            """
            
            # Call X AI API
            logger.info(f"Calling X AI LLM to rank {len(processed_profiles)} profiles for custom query")
            try:
                import time
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="grok-3",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=16384
                )
                end_time = time.time()
                logger.info(f"API call took {end_time - start_time:.2f} seconds")
                logger.info(f"Response status: {response.status_code if hasattr(response, 'status_code') else 'No status code'}")
                logger.info(f"Response headers: {response.headers if hasattr(response, 'headers') else 'No headers'}")
                logger.info(f"Raw response content: {response.choices[0].message.content if response.choices else 'No content'}")
            except Exception as e:
                logger.error(f"Error during API call: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                raise
            
            # Extract and parse response
            llm_response = response.choices[0].message.content
            logger.info("LLM Profile Ranking response from LLM:")
            logger.info(llm_response)
            
            # Extract JSON from the response by removing <think>...</think> tags
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
            logger.error(f"Error in LLM profile ranking for custom query: {str(e)}")
            return None
