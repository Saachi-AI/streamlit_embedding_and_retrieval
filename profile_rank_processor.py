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
        elif 4 <= rank <= 10:
            emoji = "🔟"  # Numeric emoji
        else:
            emoji = ""  # No emoji for ranks 11+
        
        return f"{emoji} Rank #{rank}"
    
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
        
        # Remove metadata keys like "_disclaimer"
        cleaned_results = {k: v for k, v in llm_ranking_results.items() if not k.startswith("_")}
        
        # Convert to list and sort by rank
        candidates = []
        for profile_id, profile_info in cleaned_results.items():
            # Get rank and determine emoji
            rank = profile_info.get("rank", 99)
            rank_display = self._get_rank_display(rank)
            
            candidates.append({
                "profile_id": profile_id,
                "rank": rank,
                "rank_display": rank_display,
                "short_phrase": profile_info.get("shortPhrase", "")
            })
        
        # Sort by rank
        candidates.sort(key=lambda x: x["rank"])
        
        self.logger.info(f"Processed {len(candidates)} ranked profiles")
        return candidates 