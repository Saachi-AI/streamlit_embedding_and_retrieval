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
        
        # First try LinkedIn data
        if "linkedin_data" in profile_data[profile_id] and profile_data[profile_id]["linkedin_data"]:
            linkedin_data = profile_data[profile_id]["linkedin_data"]
            if "work_experience" in linkedin_data and linkedin_data["work_experience"]:
                # Sort work experience by recency - current positions first, then by start date
                work_experiences = sorted(
                    linkedin_data["work_experience"],
                    key=lambda x: (x.get("end_date") is not None, x.get("start_date", ""), x.get("end_date", "")),
                    reverse=True
                )
                
                if work_experiences:
                    latest_job = work_experiences[0]
                    employment_info["position"] = latest_job.get("position")
                    employment_info["company_name"] = latest_job.get("company_name")
                    
                    # Format period
                    start_date = latest_job.get("start_date")
                    end_date = latest_job.get("end_date")
                    
                    if start_date:
                        if end_date:
                            employment_info["period"] = f"{start_date} - {end_date}"
                        else:
                            employment_info["period"] = f"{start_date} - Present"
        
        # If LinkedIn data doesn't have employment info, try Tamago data
        if (not employment_info["position"] or not employment_info["company_name"]) and "tamago_data" in profile_data[profile_id]:
            tamago_data = profile_data[profile_id]["tamago_data"]
            if "employments" in tamago_data and tamago_data["employments"]:
                # Sort employments by recency - current positions first, then by start date
                employments = sorted(
                    tamago_data["employments"],
                    key=lambda x: (x.get("end_date") is not None, x.get("start_date", ""), x.get("end_date", "")),
                    reverse=True
                )
                
                if employments:
                    latest_job = employments[0]
                    if not employment_info["position"]:
                        employment_info["position"] = latest_job.get("position")
                    if not employment_info["company_name"]:
                        employment_info["company_name"] = latest_job.get("company_name")
                    
                    # Only set period if we haven't already
                    if not employment_info["period"]:
                        start_date = latest_job.get("start_date")
                        end_date = latest_job.get("end_date")
                        
                        if start_date:
                            if end_date:
                                employment_info["period"] = f"{start_date} - {end_date}"
                            else:
                                employment_info["period"] = f"{start_date} - Present"
        
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
            
            candidates.append({
                "profile_id": profile_id,
                "rank": rank,
                "rank_display": rank_display,
                "short_phrase": profile_info.get("shortPhrase", ""),
                "display_name": display_name,
                "position": employment_info["position"],
                "company_name": employment_info["company_name"],
                "period": employment_info["period"]
            })
        
        # Sort by rank
        candidates.sort(key=lambda x: x["rank"])
        
        self.logger.info(f"Processed {len(candidates)} ranked profiles")
        return candidates 