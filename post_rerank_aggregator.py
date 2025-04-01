import os
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL") if os.getenv("LOG_LEVEL") in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] else "INFO")
logger = logging.getLogger(__name__)

@dataclass
class ProfileScore:
    """Class for storing profile score details."""
    profile_id: str
    final_score: float
    best_chunk_score: float
    above_threshold_count: int
    chunks: List[Tuple[Any, float]]  # List of (document, score) pairs


class ProfileAggregator:
    """
    A class to aggregate reranked chunks into profile-level scores.
    """
    
    def __init__(self):
        """Initialize the ProfileAggregator with configurable parameters from environment variables."""
        # Load parameters from environment variables with defaults
        self.alpha = float(os.getenv("PROFILE_BONUS_ALPHA", "0.05"))
        self.threshold = float(os.getenv("PROFILE_SCORE_THRESHOLD", "0.70"))
        self.top_k_profiles = int(os.getenv("TOP_K_PROFILES", "5"))
        
        logger.debug(
            f"ProfileAggregator initialized with: "
            f"alpha={self.alpha}, threshold={self.threshold}, top_k_profiles={self.top_k_profiles}"
        )
    
    def aggregate_profiles(
        self, 
        reranked_results: List[Tuple[Any, float]], 
        profile_id_field: str = "profile_id", 
        top_k: Optional[int] = None
    ) -> List[ProfileScore]:
        """
        Aggregate reranked chunks by profile using the "Max + Bonus" approach.
        
        The formula is:
            profile_score = best_chunk_score + alpha * count_of_above_threshold
        
        Args:
            reranked_results: List of (document, score) tuples from reranking
            profile_id_field: The metadata field name containing the profile ID (default: "profile_id")
            top_k: Number of top profiles to return (overrides the default from environment)
            
        Returns:
            List of ProfileScore objects for the top profiles, sorted by final_score descending
        """
        if not reranked_results:
            logger.warning("No reranked results provided for profile aggregation")
            return []
        
        # If no custom top_k is provided, use the value from environment
        if top_k is None:
            top_k = self.top_k_profiles
            
        # Group chunks by profile ID
        profile_chunks_map = defaultdict(list)
        
        for doc, score in reranked_results:
            # Extract profile ID from metadata (handle different data types)
            profile_id = doc.metadata.get(profile_id_field)
            
            # Convert numeric IDs to string (to handle potential floating point values)
            if isinstance(profile_id, (int, float)):
                profile_id = str(int(profile_id))
                
            # Skip items without a valid profile ID
            if not profile_id or profile_id == "N/A":
                logger.warning(f"Skipping document with missing profile ID: {doc}")
                continue
                
            # Store document and score in the map
            profile_chunks_map[profile_id].append((doc, score))
            
        # Calculate profile scores
        profile_scores = []
        
        for profile_id, chunks in profile_chunks_map.items():
            # Extract just the scores for calculating max and threshold counts
            scores = [score for _, score in chunks]
            
            # Calculate the best chunk score
            best_chunk_score = max(scores) if scores else 0
            
            # Count chunks above threshold
            above_threshold = sum(1 for s in scores if s >= self.threshold)
            
            # Calculate final score using the formula
            final_score = best_chunk_score + self.alpha * above_threshold
            
            # Create a ProfileScore object with all relevant information
            profile_score = ProfileScore(
                profile_id=profile_id,
                final_score=final_score,
                best_chunk_score=best_chunk_score,
                above_threshold_count=above_threshold,
                chunks=chunks
            )
            
            profile_scores.append(profile_score)
            
        # Sort profiles by final score in descending order
        profile_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Return top K profiles
        return profile_scores[:top_k]
    
    def get_explanation(self, profile_score: ProfileScore) -> str:
        """
        Generate an explanation of how the profile score was calculated.
        
        Args:
            profile_score: A ProfileScore object
            
        Returns:
            A string explaining the score calculation
        """
        return (
            f"Best chunk score ({profile_score.best_chunk_score:.3f}) + "
            f"Bonus ({self.alpha} × {profile_score.above_threshold_count} chunks above {self.threshold}) = "
            f"{profile_score.final_score:.3f}"
        ) 