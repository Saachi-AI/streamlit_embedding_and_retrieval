import json
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileRankProcessor:
    """
    Class to process LLM ranking results for UI display.
    Transforms the LLM output to a format suitable for Streamlit UI components.
    """
    
    def __init__(self):
        """Initialize the ProfileRankProcessor."""
        self.logger = logging.getLogger(__name__)
    
    def _get_rank_display(self, rank: int) -> str:
        """
        Get the display string for a rank with the appropriate emoji.
        
        Args:
            rank: The numerical rank (1, 2, 3, etc.)
            
        Returns:
            String with rank emoji and number (e.g., "🏅 Rank #1")
        """
        if rank == 1:
            emoji = "🏅"  # Gold medal
        elif rank == 2:
            emoji = "🥈"  # Silver medal
        elif rank == 3:
            emoji = "🥉"  # Bronze medal
        elif rank == 4:
            emoji = "4️⃣"  # Keycap digit four
        elif rank == 5:
            emoji = "5️⃣"  # Keycap digit five
        elif rank == 6:
            emoji = "6️⃣"  # Keycap digit six
        elif rank == 7:
            emoji = "7️⃣"  # Keycap digit seven
        elif rank == 8:
            emoji = "8️⃣"  # Keycap digit eight
        elif rank == 9:
            emoji = "9️⃣"  # Keycap digit nine
        elif rank == 10:
            emoji = "🔟"  # Keycap: 10
        else:
            emoji = str(rank)  # No emoji for ranks 11+
        
        return f"{emoji} Rank #{rank}"
    
    def _get_employment_info(self, profile_data, profile_id):
        """
        Extract employment information from profile data.
        
        Args:
            profile_data: Raw profile data dictionary
            profile_id: The profile ID to look up
            
        Returns:
            Dictionary with position, company_name, and period information
        """
        employment_info = {
            "position": None,
            "company_name": None,
            "period": None
        }
        
        if profile_id not in profile_data:
            return employment_info
        
        # Function to convert date to YYYY/MM format
        def format_date(date_str):
            if not date_str:
                return None
            
            try:
                # Handle LinkedIn format (MM/DD/YYYY)
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        return f"{year}/{month.zfill(2)}"
                # Handle Tamago format (YYYY-MM-DD)
                elif '-' in date_str:
                    parts = date_str.split('-')
                    if len(parts) == 3:
                        year, month, day = parts
                        return f"{year}/{month.zfill(2)}"
                return date_str  # Return as is if format not recognized
            except Exception:
                return date_str  # Return as is if any error occurs
        
        # Function to parse date string to datetime for comparison
        def parse_date_for_sorting(date_str):
            from datetime import datetime
            import re
            
            if not date_str:
                return None
                
            try:
                # Handle LinkedIn format (MM/DD/YYYY)
                if '/' in date_str:
                    match = re.match(r'(\d+)/(\d+)/(\d+)', date_str)
                    if match:
                        month, day, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                
                # Handle Tamago format (YYYY-MM-DD)
                elif '-' in date_str:
                    match = re.match(r'(\d+)-(\d+)-(\d+)', date_str)
                    if match:
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                
                # Return None if format not recognized
                return None
            except Exception as e:
                self.logger.warning(f"Error parsing date {date_str}: {e}")
                return None
        
        # Function to validate date
        def is_valid_experience(experience):
            import datetime
            
            # Skip if start_date is missing
            if not experience.get('start_date') and not experience.get('start'):
                return False
                
            # Get the correct field names based on data source
            start_field = 'start' if 'start' in experience else 'start_date'
            end_field = 'end' if 'end' in experience else 'end_date'
            
            start_date = parse_date_for_sorting(experience.get(start_field))
            end_date = parse_date_for_sorting(experience.get(end_field))
            
            # Skip if start_date couldn't be parsed
            if not start_date:
                return False
                
            # Check if dates make sense
            today = datetime.datetime.now()
            
            # Warning for future start date
            if start_date > today:
                self.logger.warning(f"Future start date detected: {experience.get(start_field)}")
                return False
                
            # Warning for end date before start date
            if end_date and end_date < start_date:
                self.logger.warning(f"End date before start date: {experience.get(start_field)} -> {experience.get(end_field)}")
                return False
                
            return True
        
        # First try LinkedIn data
        if "linkedin_data" in profile_data[profile_id] and profile_data[profile_id]["linkedin_data"]:
            linkedin_data = profile_data[profile_id]["linkedin_data"]
            if "work_experience" in linkedin_data and linkedin_data["work_experience"]:
                # Filter valid experiences
                valid_experiences = []
                for exp in linkedin_data["work_experience"]:
                    # Normalize field names for consistency
                    experience = exp.copy()
                    if 'company' in experience and not experience.get('company_name'):
                        experience['company_name'] = experience['company']
                    
                    if is_valid_experience(experience):
                        valid_experiences.append(experience)
                
                if valid_experiences:
                    # Sort by: 1. Is current (end_date is None), 2. Start date (most recent first)
                    def experience_sort_key(exp):
                        is_current = not exp.get('end')  # True for current positions
                        start_date = parse_date_for_sorting(exp.get('start'))
                        # For current positions, we want is_current=True to come first
                        # When reversed=True in the sort, False comes before True
                        # So we negate is_current to get True values first
                        return (not is_current, start_date if start_date else datetime.datetime.min)
                    
                    # Sort experiences
                    sorted_experiences = sorted(valid_experiences, key=experience_sort_key)
                    
                    # Get the most relevant experience (first after sorting)
                    latest_job = sorted_experiences[0]
                    
                    employment_info["position"] = latest_job.get("position")
                    employment_info["company_name"] = latest_job.get("company_name")
                    
                    # Format period
                    start_date = latest_job.get("start")
                    end_date = latest_job.get("end")
                    
                    if start_date:
                        formatted_start = format_date(start_date)
                        if end_date:
                            formatted_end = format_date(end_date)
                            employment_info["period"] = f"{formatted_start} - {formatted_end}"
                        else:
                            employment_info["period"] = f"{formatted_start} - Present"
        
        # If LinkedIn data doesn't have employment info, try Tamago data
        if (not employment_info["position"] or not employment_info["company_name"]) and "tamago_data" in profile_data[profile_id]:
            tamago_data = profile_data[profile_id]["tamago_data"]
            if "employments" in tamago_data and tamago_data["employments"]:
                # Filter valid experiences
                valid_experiences = []
                for exp in tamago_data["employments"]:
                    if is_valid_experience(exp):
                        valid_experiences.append(exp)
                
                if valid_experiences:
                    # Sort by: 1. Is current (end_date is None), 2. Start date (most recent first)
                    def experience_sort_key(exp):
                        is_current = not exp.get('end_date')  # True for current positions
                        start_date = parse_date_for_sorting(exp.get('start_date'))
                        # For current positions, we want is_current=True to come first
                        # When reversed=True in the sort, False comes before True
                        # So we negate is_current to get True values first
                        return (not is_current, start_date if start_date else datetime.datetime.min)
                    
                    # Sort experiences
                    sorted_experiences = sorted(valid_experiences, key=experience_sort_key)
                    
                    # Get the most relevant experience (first after sorting)
                    latest_job = sorted_experiences[0]
                    
                    if not employment_info["position"]:
                        employment_info["position"] = latest_job.get("position")
                    if not employment_info["company_name"]:
                        employment_info["company_name"] = latest_job.get("company_name")
                    
                    # Only set period if we haven't already
                    if not employment_info["period"]:
                        start_date = latest_job.get("start_date")
                        end_date = latest_job.get("end_date")
                        
                        if start_date:
                            formatted_start = format_date(start_date)
                            if end_date:
                                formatted_end = format_date(end_date)
                                employment_info["period"] = f"{formatted_start} - {formatted_end}"
                            else:
                                employment_info["period"] = f"{formatted_start} - Present"
        
        return employment_info
    
    def process_ranked_profiles(self, llm_ranking_results: Union[str, Dict[str, Any]], profile_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process LLM ranking results and profile data for UI display.
        
        Args:
            llm_ranking_results: Output from LLMProfileRanker (string or dict)
            profile_data: Raw profile data from ProfileRetriever (dict)
            
        Returns:
            List of processed candidate objects for UI display
        """
        self.logger.info("Processing ranked profiles for UI display")
        
        # Log full structure of one profile_data entry to verify contents
        if profile_data and len(profile_data) > 0:
            first_profile_id = next(iter(profile_data))
            self.logger.info(f"Sample profile_data structure for {first_profile_id}:")
            self.logger.info(f"Keys in profile_data[{first_profile_id}]: {profile_data[first_profile_id].keys()}")
            
            # Log metadata
            if "metadata" in profile_data[first_profile_id]:
                self.logger.info(f"Metadata keys: {profile_data[first_profile_id]['metadata'].keys()}")
            
            # Log tamago_data (just the keys to avoid huge logs)
            if "tamago_data" in profile_data[first_profile_id]:
                self.logger.info(f"tamago_data keys: {profile_data[first_profile_id]['tamago_data'].keys()}")
            
            # Log linkedin_data (just the keys to avoid huge logs)
            if "linkedin_data" in profile_data[first_profile_id]:
                self.logger.info(f"linkedin_data keys: {profile_data[first_profile_id]['linkedin_data'].keys()}")
        
        # Log raw llm_ranking_results
        self.logger.info(f"Raw LLM ranking results type: {type(llm_ranking_results)}")
        if isinstance(llm_ranking_results, str):
            self.logger.info(f"Raw LLM ranking results preview: {llm_ranking_results[:500]}...")
        else:
            self.logger.info(f"Raw LLM ranking results keys: {list(llm_ranking_results.keys())}")
        
        # Log profile_data
        self.logger.info(f"Profile data contains {len(profile_data)} profiles")
        for profile_id in profile_data:
            self.logger.info(f"Profile data contains profile_id: {profile_id}")
        
        # Handle string input (if LLM results is a string)
        if isinstance(llm_ranking_results, str):
            try:
                llm_ranking_results = json.loads(llm_ranking_results)
                self.logger.info("Successfully parsed LLM ranking results from string")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM ranking results: {e}")
                return []
        
        # Check if we have valid results
        if not llm_ranking_results:
            self.logger.warning("No LLM ranking results to process")
            return []
        
        # Log LLM ranking results
        cleaned_results = {k: v for k, v in llm_ranking_results.items() if not k.startswith("_")}
        self.logger.info(f"LLM ranking results contain {len(cleaned_results)} profiles")
        for profile_id in cleaned_results:
            self.logger.info(f"LLM ranking results contain profile_id: {profile_id}")
        
        # Convert to list and sort by rank
        candidates = []
        for profile_id, profile_info in cleaned_results.items():
            # Get rank and determine emoji
            rank = profile_info.get("rank", 99)
            rank_display = self._get_rank_display(rank)
            
            # Get display_name from profile_data if available
            display_name = "[No Name]"
            if profile_id in profile_data and "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                if tamago_data and "display_name" in tamago_data:
                    display_name = tamago_data["display_name"]
            
            # Get employment information
            employment_info = self._get_employment_info(profile_data, profile_id)
            
            # Get profile picture URL
            profile_picture_url = None
            if profile_id in profile_data and "linkedin_data" in profile_data[profile_id]:
                linkedin_data = profile_data[profile_id]["linkedin_data"]
                if linkedin_data and "profile_picture_url_large" in linkedin_data:
                    profile_picture_url = linkedin_data["profile_picture_url_large"]
            
            # Use fallback avatar if no LinkedIn profile picture
            if not profile_picture_url:
                profile_picture_url = "https://api.dicebear.com/9.x/avataaars/svg?seed=Oliver"
            
            # Get match score and reasons for recommendation
            match_score = profile_info.get("overallScore", "")
            why_good_fit = profile_info.get("whyGoodFit", [])
            
            # Extract additional metadata fields
            years_of_experience = None
            gender = None
            is_candidate = None
            languages = {}
            
            if profile_id in profile_data and "metadata" in profile_data[profile_id]:
                metadata = profile_data[profile_id]["metadata"]
                
                # Extract years of experience
                if "years_of_experience" in metadata:
                    years_of_experience = metadata["years_of_experience"]
                
                # Extract gender
                if "gender" in metadata:
                    gender = metadata["gender"]
                
                # Extract is_candidate
                if "is_candidate" in metadata:
                    is_candidate = metadata["is_candidate"]
                
                # Extract languages by filtering out non-language fields
                excluded_fields = {"profile_id", "section", "placed", "years_of_experience", 
                                "gender", "last_contacted", "is_candidate", "processed_at", 
                                "position", "keywords"}
                
                languages = {key: value for key, value in metadata.items() if key not in excluded_fields}
            
            # Determine the candidate type based on is_candidate value
            candidate_type = "Candidate" if is_candidate == True else "Lead"
            
            candidates.append({
                "profile_id": profile_id,
                "rank": rank,
                "rank_display": rank_display,
                "short_phrase": profile_info.get("shortPhrase", ""),
                "display_name": display_name,
                "position": employment_info["position"],
                "company_name": employment_info["company_name"],
                "period": employment_info["period"],
                "profile_picture_url": profile_picture_url,
                "match_score": match_score,
                "why_good_fit": why_good_fit,
                "years_of_experience": years_of_experience,
                "gender": gender,
                "type": candidate_type,
                "languages": languages
            })
        
        # Sort by rank
        candidates.sort(key=lambda x: x["rank"])
        
        self.logger.info(f"Processed {len(candidates)} ranked profiles")
        return candidates 