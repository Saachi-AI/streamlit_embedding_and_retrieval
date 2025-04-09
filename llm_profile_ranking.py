import os
import json
import logging
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
        """Initialize with GROQ API key."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment. LLM ranking will not work.")
        
        # Initialize the OpenAI client (GROQ uses OpenAI compatible API)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        
        # System prompt for LLM
        self.system_prompt = """
        You are a world-class recruiter with extensive experience in analyzing candidate profiles. When given:

        1. A single job description (JD) with explicit or implicit weighting priorities (domain expertise, technical skills, language proficiency, etc.).  
        2. An array of JSON candidate profiles.

        You will produce a ranked evaluation for each candidate, returning a single JSON object containing structured results.

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

        2. Skills Handling
           - Look for synonyms or tangential references to required skills. For example, if the JD says "Azure" but the candidate mentions "Microsoft Cloud," treat that as partial or potentially "has," depending on context.
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

        =========================
        3. Output Format
        =========================

        You must respond with one JSON object containing:

        - One key per candidate, named by their profile_id.
        - The value is another JSON object with fields:

          1. "rank": integer
          2. "shortPhrase": ~5-6 words about the candidate
          3. "whyGoodFit": short textual explanation (include mention of education relevance here)
          4. "overallScore": integer (e.g., 0-100)
          5. "skillsMatch": sub-object with each key JD requirement → "has", "partial", "missing", or "insufficient data"
          6. "contradictionsOrWarnings": array of strings if any contradictions exist
          7. "weighting": object showing your domain vs. technical vs. language ratio

        Finally, include a disclaimers field at the root of the JSON:

        "_disclaimer": "Do not add text outside this JSON. If uncertain, we mark skill as missing or partial..."

        Example skeleton:

        {
          "profile_id_1234": {
            "rank": 1,
            "shortPhrase": "5-6 words about them",
            "whyGoodFit": "Short paragraph or bullet points about strengths and education relevance",
            "overallScore": 90,
            "skillsMatch": {
              "Azure": "partial",
              "Python": "has",
              "DomainKnowledge": "missing",
              "LanguageJP": "full_proficiency"
            },
            "contradictionsOrWarnings": [],
            "weighting": {
              "domainExpertise": 30,
              "technicalSkills": 50,
              "languageProficiency": 20
            }
          },
          "profile_id_5678": {
            "rank": 2,
            ...
          },
          "_disclaimer": "Do not add text outside this JSON. ..."
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
           - Keep "whyGoodFit" to a short paragraph or bullet points. Mention if the candidate's education is relevant to the JD or not.

        4. Missing Skills
           - If the candidate's data has no mention (or direct synonym) of a required skill, mark "missing."

        5. Single JD
           - Only one JD per request.

        6. Contradictions
           - If older notes conflict with the resume, favor the more recent info but add a warning in "contradictionsOrWarnings".

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

            Below is the array of candidate profiles in JSON format. Please analyze them
            according to the system instructions. Then produce a single JSON as the final output,
            with no extra text.

            --- CANDIDATE JSON START ---
            {candidate_json}
            --- CANDIDATE JSON END ---
            """
            
            # Call GROQ API
            logger.info(f"Calling GROQ LLM to rank {len(processed_profiles)} profiles for job description")
            response = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            # Extract and parse response
            llm_response = response.choices[0].message.content
            logger.info("Profile ranking response from LLM:")
            logger.info(llm_response)
            
            return llm_response
            
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

            --- CANDIDATE JSON START ---
            {candidate_json}
            --- CANDIDATE JSON END ---
            """
            
            # Call GROQ API
            logger.info(f"Calling GROQ LLM to rank {len(processed_profiles)} profiles for custom query")
            response = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            # Extract and parse response
            llm_response = response.choices[0].message.content
            logger.info("Profile ranking response from LLM:")
            logger.info(llm_response)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error in LLM profile ranking for custom query: {str(e)}")
            return None
