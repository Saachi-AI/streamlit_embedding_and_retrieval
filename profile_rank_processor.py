import json
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileRankProcessor:
    """
    Class to process profile evaluation results for UI display.
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
                        # Fix for invalid month/day values
                        if month == '00':
                            month = '01'
                        if day == '00':
                            day = '01'
                        return datetime(int(year), int(month), int(day))
                
                # Handle Tamago format (YYYY-MM-DD)
                elif '-' in date_str:
                    match = re.match(r'(\d+)-(\d+)-(\d+)', date_str)
                    if match:
                        year, month, day = match.groups()
                        # Fix for invalid month/day values
                        if month == '00':
                            month = '01'
                        if day == '00':
                            day = '01'
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
                        # For current positions (no end date), use priority 1, otherwise 0
                        current_priority = 1 if not exp.get('end') else 0
                        
                        # Get start date for sorting by recency
                        start_date = parse_date_for_sorting(exp.get('start'))
                        if not start_date:
                            start_date = datetime.datetime.min
                            
                        # Return tuple where higher values come first when reverse=True
                        return (current_priority, start_date)
                    
                    # Sort experiences - most recent first with current positions prioritized
                    sorted_experiences = sorted(valid_experiences, key=experience_sort_key, reverse=True)
                    
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
                        # For current positions (no end date), use priority 1, otherwise 0
                        current_priority = 1 if not exp.get('end_date') else 0
                        
                        # Get start date for sorting by recency
                        start_date = parse_date_for_sorting(exp.get('start_date'))
                        if not start_date:
                            start_date = datetime.datetime.min
                            
                        # Return tuple where higher values come first when reverse=True
                        return (current_priority, start_date)
                    
                    # Sort experiences - most recent first with current positions prioritized
                    sorted_experiences = sorted(valid_experiences, key=experience_sort_key, reverse=True)
                    
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
    
    def process_evaluated_profiles(self, evaluation_results: Dict[str, Any], profile_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process profile evaluation results and profile data for UI display.
        
        Args:
            evaluation_results: Output from IndividualProfileEvaluator (dict)
            profile_data: Raw profile data from ProfileRetriever (dict)
            
        Returns:
            List of processed candidate objects for UI display
        """
        self.logger.info("Processing evaluated profiles for UI display")
        
        # Extract profiles and dimensions from the evaluation results
        profiles = evaluation_results.get("profiles", [])
        dimensions = evaluation_results.get("job_dimensions", [])
        
        if not profiles:
            self.logger.warning("No evaluated profiles to process")
            return []
        
        # Log profile_data
        self.logger.info(f"Profile data contains {len(profile_data)} profiles")
        
        # Convert to list and sort by rank (should already be sorted, but ensuring)
        candidates = []
        for profile in profiles:
            profile_id = profile.get("profile_id", "unknown")
            
            # Skip profiles without valid profile_id
            if profile_id == "unknown" or profile_id not in profile_data:
                self.logger.warning(f"Skipping profile with missing or invalid profile_id: {profile_id}")
                continue
                
            # Get rank and determine emoji
            rank = profile.get("rank", 99)
            rank_display = self._get_rank_display(rank)
            
            # Get display_name from profile_data if available
            display_name = "[No Name]"
            if "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                if tamago_data and "display_name" in tamago_data:
                    display_name = tamago_data["display_name"]
            
            # Get employment information
            employment_info = self._get_employment_info(profile_data, profile_id)
            
            # Get profile picture URL and LinkedIn data
            profile_picture_url = None
            linkedin_fetched_at = None
            linkedin_url = None
            if "linkedin_data" in profile_data[profile_id]:
                linkedin_data = profile_data[profile_id]["linkedin_data"]
                if linkedin_data:
                    if "profile_picture_url_large" in linkedin_data:
                        profile_picture_url = linkedin_data["profile_picture_url_large"]
                    if "fetched_at" in linkedin_data:
                        linkedin_fetched_at = linkedin_data["fetched_at"].split("T")[0]  # Convert to YYYY-MM-DD
            
            # Get LinkedIn URL from tamago_data
            if "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                if tamago_data and tamago_data.get("linkedin"):
                    linkedin_url = tamago_data["linkedin"]
                    # Ensure LinkedIn URL has https:// prefix
                    if linkedin_url and not linkedin_url.startswith(('http://', 'https://')):
                        linkedin_url = f"https://{linkedin_url}"
            
            # Construct Tamago URL using profile_id
            tamago_url = f"https://saachi.tamago-db.com/contact/{profile_id}/show"
            
            # Use fallback avatar if no LinkedIn profile picture
            if not profile_picture_url:
                profile_picture_url = "https://api.dicebear.com/9.x/avataaars/svg?seed=Oliver"
            
            # Get match data from evaluation results
            overall_match = profile.get("overall_match", {})
            match_percentage = overall_match.get("percentage", 0)
            match_category = overall_match.get("category", "Unknown")
            match_reasoning = overall_match.get("reasoning", "")
            
            # Get dimension scores
            dimension_scores = profile.get("dimensions", [])
            
            # Get strengths and gaps
            key_strengths = profile.get("key_strengths", [])
            key_gaps = profile.get("key_gaps", [])
            
            # Get overqualification status
            is_overqualified = profile.get("is_overqualified", False)
            overqualification_reasoning = profile.get("overqualification_reasoning", "")
            
            # Format dimension scores for display
            formatted_dimensions = []
            for dim in dimension_scores:
                # Get score and determine color
                score = dim.get("score", 0)
                
                # Determine color based on score
                if score >= 85:
                    color = "green"
                elif score >= 70:
                    color = "lightgreen"
                elif score >= 50:
                    color = "orange"
                else:
                    color = "red"
                
                formatted_dimensions.append({
                    "name": dim.get("name", ""),
                    "score": score,
                    "reasoning": dim.get("reasoning", ""),
                    "color": color
                })
            
            # Extract additional metadata fields
            years_of_experience = None
            gender = None
            is_candidate = None
            languages = {}
            
            if "metadata" in profile_data[profile_id]:
                metadata = profile_data[profile_id]["metadata"]
                
                # Extract years of experience
                if "years_of_experience" in metadata:
                    years_of_experience = metadata["years_of_experience"]
                
                # Extract and format gender
                if "gender" in metadata:
                    raw_gender = metadata["gender"]
                    if raw_gender.lower() in ['male', 'female']:
                        gender = raw_gender.capitalize()
                    else:
                        gender = raw_gender  # Keep original value for other cases
                
                # Extract is_candidate
                if "is_candidate" in metadata:
                    is_candidate = metadata["is_candidate"]
                
                # Extract languages by filtering out non-language fields
                excluded_fields = {"profile_id", "section", "placed", "years_of_experience", 
                                "gender", "last_contacted", "is_candidate", "processed_at", 
                                "position", "keywords"}
                
                languages = {key: value for key, value in metadata.items() if key not in excluded_fields}
            
            # Process profile creation and last contact information
            previous_placements = []
            if "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                
                # Process Previously Placed information
                if tamago_data.get("pipeline"):
                    # Filter and process placement entries
                    for entry in tamago_data["pipeline"]:
                        if entry.get("result") == "placed":
                            # Get the date (prefer updated_at, fallback to created_at)
                            placement_date = entry.get("updated_at") or entry.get("created_at")
                            if (placement_date and 
                                entry.get("company_name") and 
                                entry.get("created_by", {}).get("name")):
                                
                                previous_placements.append({
                                    "company_name": entry["company_name"],
                                    "date": placement_date.split("T")[0],  # Convert to YYYY-MM-DD
                                    "person": entry["created_by"]["name"]
                                })
                    
                    # Sort placements by date (most recent first)
                    if previous_placements:
                        previous_placements.sort(key=lambda x: x["date"], reverse=True)
            
            # Create candidate object
            candidate = {
                "profile_id": profile_id,
                "rank": rank,
                "rank_display": rank_display,
                "display_name": display_name,
                "current_position": employment_info.get("position"),
                "current_company": employment_info.get("company_name"),
                "employment_period": employment_info.get("period"),
                "profile_picture_url": profile_picture_url,
                "match_percentage": match_percentage,
                "match_category": match_category,
                "match_reasoning": match_reasoning,
                "dimension_scores": formatted_dimensions,
                "key_strengths": key_strengths,
                "key_gaps": key_gaps,
                "is_overqualified": is_overqualified,
                "overqualification_reasoning": overqualification_reasoning,
                "years_of_experience": years_of_experience,
                "gender": gender,
                "languages": languages,
                "is_candidate": is_candidate,
                "linkedin_url": linkedin_url,
                "tamago_url": tamago_url,
                "linkedin_fetched_at": linkedin_fetched_at,
                "previous_placements": previous_placements
            }
            
            candidates.append(candidate)
        
        # Sort candidates by rank
        candidates.sort(key=lambda x: x["rank"])
        
        return candidates
    
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
            
            # Get profile picture URL and LinkedIn data
            profile_picture_url = None
            linkedin_fetched_at = None
            linkedin_url = None
            if profile_id in profile_data and "linkedin_data" in profile_data[profile_id]:
                linkedin_data = profile_data[profile_id]["linkedin_data"]
                if linkedin_data:
                    if "profile_picture_url_large" in linkedin_data:
                        profile_picture_url = linkedin_data["profile_picture_url_large"]
                    if "fetched_at" in linkedin_data:
                        linkedin_fetched_at = linkedin_data["fetched_at"].split("T")[0]  # Convert to YYYY-MM-DD
            
            # Get LinkedIn URL from tamago_data
            if profile_id in profile_data and "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                if tamago_data and tamago_data.get("linkedin"):
                    linkedin_url = tamago_data["linkedin"]
                    # Ensure LinkedIn URL has https:// prefix
                    if linkedin_url and not linkedin_url.startswith(('http://', 'https://')):
                        linkedin_url = f"https://{linkedin_url}"
            
            # Construct Tamago URL using profile_id
            tamago_url = f"https://saachi.tamago-db.com/contact/{profile_id}/show"
            
            # Use fallback avatar if no LinkedIn profile picture
            if not profile_picture_url:
                profile_picture_url = "https://api.dicebear.com/9.x/avataaars/svg?seed=Oliver"
            
            # Get match score and reasons for recommendation
            match_score = profile_info.get("overallScore", "")
            why_good_fit = profile_info.get("whyGoodFit", [])
            
            # Get contradictions or warnings
            contradictions = profile_info.get("contradictionsOrWarnings", [])
            
            # Process skills match data
            skills_match = []
            if "skillsMatch" in profile_info:
                # Sort skills by status (has -> partial -> missing)
                status_priority = {"has": 0, "partial": 1, "missing": 2}
                sorted_skills = sorted(
                    profile_info["skillsMatch"].items(),
                    key=lambda x: status_priority[x[1]]
                )
                
                # Format each skill with appropriate emoji
                for skill_name, status in sorted_skills:
                    if status == "has":
                        formatted_status = "✅"
                    elif status == "partial":
                        formatted_status = "🟡 (Partial)"
                    else:  # missing
                        formatted_status = "❌ (Missing)"
                    
                    skills_match.append({
                        "skill": skill_name,
                        "status_display": formatted_status
                    })
            
            # Extract additional metadata fields
            years_of_experience = None
            gender = None
            is_candidate = None
            languages = {}
            profile_created = None
            last_contacted = None
            
            if profile_id in profile_data and "metadata" in profile_data[profile_id]:
                metadata = profile_data[profile_id]["metadata"]
                
                # Extract years of experience
                if "years_of_experience" in metadata:
                    years_of_experience = metadata["years_of_experience"]
                
                # Extract and format gender
                if "gender" in metadata:
                    raw_gender = metadata["gender"]
                    if raw_gender.lower() in ['male', 'female']:
                        gender = raw_gender.capitalize()
                    else:
                        gender = raw_gender  # Keep original value for other cases
                
                # Extract is_candidate
                if "is_candidate" in metadata:
                    is_candidate = metadata["is_candidate"]
                
                # Extract languages by filtering out non-language fields
                excluded_fields = {"profile_id", "section", "placed", "years_of_experience", 
                                "gender", "last_contacted", "is_candidate", "processed_at", 
                                "position", "keywords"}
                
                languages = {key: value for key, value in metadata.items() if key not in excluded_fields}
            
            # Process profile creation and last contact information
            if profile_id in profile_data and "tamago_data" in profile_data[profile_id]:
                tamago_data = profile_data[profile_id]["tamago_data"]
                
                # Process Previously Placed information
                previous_placements = []
                if tamago_data.get("pipeline"):
                    # Filter and process placement entries
                    for entry in tamago_data["pipeline"]:
                        if entry.get("result") == "placed":
                            # Get the date (prefer updated_at, fallback to created_at)
                            placement_date = entry.get("updated_at") or entry.get("created_at")
                            if (placement_date and 
                                entry.get("company_name") and 
                                entry.get("created_by", {}).get("name")):
                                
                                previous_placements.append({
                                    "company_name": entry["company_name"],
                                    "date": placement_date.split("T")[0],  # Convert to YYYY-MM-DD
                                    "person": entry["created_by"]["name"]
                                })
                    
                    # Sort placements by date (most recent first)
                    if previous_placements:
                        previous_placements.sort(key=lambda x: x["date"], reverse=True)
                
                # Process Profile Created By
                if (tamago_data.get("created_by") and 
                    tamago_data["created_by"].get("name") and 
                    tamago_data.get("created_at")):
                    created_date = tamago_data["created_at"].split("T")[0]  # Get YYYY-MM-DD
                    profile_created = {
                        "name": tamago_data["created_by"]["name"],
                        "date": created_date
                    }
                
                # Process Last Contacted By
                if tamago_data.get("notes") and len(tamago_data["notes"]) > 0:
                    # Sort notes by both created_at and updated_at timestamps
                    notes_with_times = []
                    for note in tamago_data["notes"]:
                        if note.get("created_at"):
                            notes_with_times.append({
                                "timestamp": note["created_at"],
                                "is_update": False,
                                "note": note
                            })
                        if note.get("updated_at"):
                            notes_with_times.append({
                                "timestamp": note["updated_at"],
                                "is_update": True,
                                "note": note
                            })
                    
                    # Sort by timestamp in descending order
                    notes_with_times.sort(key=lambda x: x["timestamp"], reverse=True)
                    
                    # Find the most recent note with valid name information
                    for note_info in notes_with_times:
                        note = note_info["note"]
                        if note_info["is_update"] and note.get("updated_by") and note["updated_by"].get("name"):
                            contact_date = note_info["timestamp"].split("T")[0]  # Get YYYY-MM-DD
                            last_contacted = {
                                "name": note["updated_by"]["name"],
                                "date": contact_date
                            }
                            break
                        elif not note_info["is_update"] and note.get("created_by") and note["created_by"].get("name"):
                            contact_date = note_info["timestamp"].split("T")[0]  # Get YYYY-MM-DD
                            last_contacted = {
                                "name": note["created_by"]["name"],
                                "date": contact_date
                            }
                            break
            
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
                "linkedin_fetched_at": linkedin_fetched_at,
                "linkedin_url": linkedin_url,
                "tamago_url": tamago_url,
                "match_score": match_score,
                "why_good_fit": why_good_fit,
                "contradictions": contradictions if contradictions else None,
                "skills_match": skills_match,
                "years_of_experience": years_of_experience,
                "gender": gender,
                "type": candidate_type,
                "languages": languages,
                "previous_placements": previous_placements if previous_placements else None,
                "profile_created": profile_created,
                "last_contacted": last_contacted
            })
        
        # Sort by rank
        candidates.sort(key=lambda x: x["rank"])
        
        self.logger.info(f"Processed {len(candidates)} ranked profiles")
        return candidates 