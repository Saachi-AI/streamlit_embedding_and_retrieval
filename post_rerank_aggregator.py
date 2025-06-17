import os
import logging
import streamlit as st
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
def get_log_level():
    try:
        return st.secrets.get("LOG_LEVEL", "INFO")
    except:
        return os.getenv("LOG_LEVEL", "INFO")

log_level = get_log_level()
if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    logging.basicConfig(level=log_level)
else:
    logging.basicConfig(level="INFO")
    
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
        """Initialize the ProfileAggregator with configurable parameters from secrets or environment variables."""
        # Load parameters from secrets first, then environment variables with defaults
        try:
            self.alpha = float(st.secrets.get("PROFILE_BONUS_ALPHA", "0.05"))
            self.default_threshold = float(st.secrets.get("PROFILE_SCORE_THRESHOLD", "0.70"))
            self.top_k_profiles = int(st.secrets.get("TOP_K_PROFILES", "20"))
            self.min_score_threshold = float(st.secrets.get("MIN_PROFILE_SCORE", "0.80"))
        except:
            # Fallback to environment variables
            self.alpha = float(os.getenv("PROFILE_BONUS_ALPHA", "0.05"))
            self.default_threshold = float(os.getenv("PROFILE_SCORE_THRESHOLD", "0.70"))
            self.top_k_profiles = int(os.getenv("TOP_K_PROFILES", "20"))
            self.min_score_threshold = float(os.getenv("MIN_PROFILE_SCORE", "0.80"))
        
        logger.debug(
            f"ProfileAggregator initialized with: "
            f"alpha={self.alpha}, default_threshold={self.default_threshold}, top_k_profiles={self.top_k_profiles}, "
            f"min_score_threshold={self.min_score_threshold}"
        )
    
    def aggregate_profiles(
        self, 
        reranked_results: List[Tuple[Any, float]], 
        profile_id_field: str = "profile_id", 
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[ProfileScore]:
        """
        Aggregate reranked chunks by profile using the "Max + Bonus" approach.
        
        The formula is:
            profile_score = best_chunk_score + alpha * count_of_above_threshold
        
        Args:
            reranked_results: List of (document, score) tuples from reranking
            profile_id_field: The metadata field name containing the profile ID (default: "profile_id")
            top_k: Number of top profiles to return (overrides the default from environment)
            threshold: Profile score threshold for bonus counting (overrides the default from environment)
            
        Returns:
            List of ProfileScore objects for the top profiles, sorted by final_score descending
            Only profiles with final_score or best_chunk_score above min_score_threshold are included
        """
        if not reranked_results:
            logger.warning("No reranked results provided for profile aggregation")
            return []
        
        # If no custom top_k is provided, use the value from environment
        if top_k is None:
            top_k = self.top_k_profiles
            
        # If no custom threshold is provided, use the default from environment
        if threshold is None:
            threshold = self.default_threshold
            
        logger.info(f"UI threshold parameter: {threshold}, hardcoded min_score_threshold: {self.min_score_threshold}")
        logger.debug(f"Using profile score threshold: {threshold}")
            
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
            
            # Count chunks above threshold (using the passed threshold parameter)
            above_threshold = sum(1 for s in scores if s >= threshold)
            
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
        
        # Use the threshold parameter passed from UI instead of hardcoded min_score_threshold
        # This allows the UI threshold to be respected for profile filtering
        effective_threshold = threshold if threshold is not None else self.min_score_threshold
        
        # Filter profiles by the effective threshold
        filtered_profiles = [
            profile for profile in profile_scores 
            if profile.final_score >= effective_threshold or profile.best_chunk_score >= effective_threshold
        ]
        
        logger.debug(f"Filtered {len(profile_scores) - len(filtered_profiles)} profiles below score threshold of {effective_threshold}")
        
        if not filtered_profiles:
            logger.warning(f"No profiles met the minimum score threshold of {effective_threshold}")
        
        # Return top K profiles from filtered list
        return filtered_profiles[:top_k]
    
    def prepare_for_profile_retrieval(self, profile_scores):
        """
        Transform ProfileScore objects into the format expected by ProfileRetriever.
        
        Args:
            profile_scores: List of ProfileScore objects from aggregate_profiles
            
        Returns:
            List of profile entry dictionaries in the format expected by ProfileRetriever
        """
        profile_entries = []
        
        for profile_score in profile_scores:
            try:
                # Get profile_id
                profile_id = profile_score.profile_id
                if not profile_id:
                    logger.warning(f"Skipping profile with missing profile_id")
                    continue
                    
                # Get metadata from the best chunk (highest score)
                if not profile_score.chunks:
                    logger.warning(f"No chunks found for profile {profile_id}")
                    continue
                    
                best_chunk = max(profile_score.chunks, key=lambda x: x[1])
                document, _ = best_chunk
                
                # Extract metadata from document
                if not hasattr(document, 'metadata'):
                    logger.warning(f"Document missing metadata for profile {profile_id}")
                    continue
                    
                metadata = document.metadata.copy()
                
                # Create profile entry
                profile_entry = {
                    "profile_id": profile_id,
                    "metadata": metadata
                }
                
                profile_entries.append(profile_entry)
                
            except Exception as e:
                logger.error(f"Error processing profile {profile_id if 'profile_id' in locals() else 'unknown'}: {str(e)}")
                continue
        
        if not profile_entries:
            logger.warning("No valid profile entries were prepared for retrieval")
        
        return profile_entries
    
    def get_explanation(self, profile_score: ProfileScore, threshold: Optional[float] = None) -> str:
        """
        Generate an explanation of how the profile score was calculated.
        
        Args:
            profile_score: A ProfileScore object
            threshold: The threshold used for calculation (if None, uses default)
            
        Returns:
            A string explaining the score calculation
        """
        if threshold is None:
            threshold = self.default_threshold
            
        return (
            f"Best chunk score ({profile_score.best_chunk_score:.3f}) + "
            f"Bonus ({self.alpha} × {profile_score.above_threshold_count} chunks above {threshold}) = "
            f"{profile_score.final_score:.3f}"
        ) 