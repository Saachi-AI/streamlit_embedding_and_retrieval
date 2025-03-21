"""
Pinecone filter adapter for the retrieval framework.

Converts generic filters to Pinecone-specific filter format.
"""

import time
from typing import Dict, Any, Optional

from retrieval_framework.utils import get_logger

logger = get_logger(__name__)


class PineconeFilterAdapter:
    """
    Adapter for converting generic filters to Pinecone filter format.
    """
    
    @staticmethod
    def build_filter(extracted_filters: Dict[str, Any], strict_mode: bool = False) -> Dict[str, Any]:
        """
        Convert extracted filters into Pinecone filter format.
        
        Args:
            extracted_filters: Dict of filters extracted by the filter extractor
            strict_mode: If True, only include exact matches (don't include docs with missing fields)
            
        Returns:
            Dict in Pinecone filter format
        """
        logger.info(f"Building Pinecone filter with strict_mode={strict_mode}")
        
        pinecone_filter = {"$and": []}
        
        # Process gender filter
        if "gender" in extracted_filters:
            PineconeFilterAdapter._add_gender_filter(pinecone_filter, extracted_filters, strict_mode)
        
        # Process years of experience filter
        if "years_of_experience" in extracted_filters:
            PineconeFilterAdapter._add_experience_filter(pinecone_filter, extracted_filters, strict_mode)
        
        # Process last_contacted filter
        if "last_contacted" in extracted_filters:
            PineconeFilterAdapter._add_last_contacted_filter(pinecone_filter, extracted_filters, strict_mode)
        
        # Process is_candidate filter
        if "is_candidate" in extracted_filters:
            PineconeFilterAdapter._add_boolean_filter(
                pinecone_filter, "is_candidate", extracted_filters["is_candidate"], strict_mode
            )
        
        # Process placed filter
        if "placed" in extracted_filters:
            PineconeFilterAdapter._add_boolean_filter(
                pinecone_filter, "placed", extracted_filters["placed"], strict_mode
            )
        
        # Process language filters
        if "languages" in extracted_filters and isinstance(extracted_filters["languages"], dict):
            PineconeFilterAdapter._add_language_filters(pinecone_filter, extracted_filters["languages"], strict_mode)
            
        # If no filters were applied, return empty object
        if not pinecone_filter["$and"]:
            logger.info("No filters extracted, returning empty filter")
            return {}
        
        logger.info(f"Built Pinecone filter with {len(pinecone_filter['$and'])} conditions")
        return pinecone_filter
    
    @staticmethod
    def _add_gender_filter(pinecone_filter: Dict[str, Any], extracted_filters: Dict[str, Any], strict_mode: bool):
        """Add gender filter to the Pinecone filter."""
        gender_value = extracted_filters["gender"].lower()
        gender_filter = {"$or": []}
        
        if gender_value == "male":
            gender_filter["$or"].append({"gender": {"$eq": "male"}})
            # Also include "not_mentioned" unless explicitly specified as "only male"
            if not "only" in extracted_filters.get("gender_strict", ""):
                gender_filter["$or"].append({"gender": {"$eq": "not_mentioned"}})
                
        elif gender_value == "female":
            gender_filter["$or"].append({"gender": {"$eq": "female"}})
            # Also include "not_mentioned" unless explicitly specified as "only female"
            if not "only" in extracted_filters.get("gender_strict", ""):
                gender_filter["$or"].append({"gender": {"$eq": "not_mentioned"}})
        
        if gender_filter["$or"]:
            pinecone_filter["$and"].append(gender_filter)
    
    @staticmethod
    def _add_experience_filter(pinecone_filter: Dict[str, Any], extracted_filters: Dict[str, Any], strict_mode: bool):
        """Add years of experience filter to the Pinecone filter."""
        try:
            min_years = int(extracted_filters["years_of_experience"])
        except (ValueError, TypeError):
            min_years = 0
            
        if min_years > 0:
            exp_filter = {"$or": [
                {"years_of_experience": {"$gte": min_years}}
            ]}
            # Include unknown (0) values or missing fields unless in strict mode
            if not strict_mode:
                exp_filter["$or"].append({"years_of_experience": {"$eq": 0}})
                exp_filter["$or"].append({"years_of_experience": {"$exists": False}})
                
            pinecone_filter["$and"].append(exp_filter)
    
    @staticmethod
    def _add_last_contacted_filter(pinecone_filter: Dict[str, Any], extracted_filters: Dict[str, Any], strict_mode: bool):
        """Add last_contacted filter to the Pinecone filter."""
        try:
            years = float(extracted_filters["last_contacted"])
            # Convert years to timestamp (current time - years in seconds)
            current_time = int(time.time())
            cutoff_timestamp = current_time - int(years * 365 * 24 * 3600)
            
            contact_filter = {"$or": [
                {"last_contacted": {"$gte": cutoff_timestamp}}
            ]}
            # In non-strict mode, also include docs where this field is missing
            if not strict_mode:
                contact_filter["$or"].append({"last_contacted": {"$exists": False}})
                
            pinecone_filter["$and"].append(contact_filter)
        except (ValueError, TypeError):
            pass  # Skip if not a valid number
    
    @staticmethod
    def _add_boolean_filter(pinecone_filter: Dict[str, Any], field_name: str, value: Any, strict_mode: bool):
        """Add a boolean filter to the Pinecone filter."""
        try:
            bool_value = str(value).lower() in ['true', 'yes', '1']
            bool_filter = {"$or": [
                {field_name: {"$eq": bool_value}}
            ]}
            # In non-strict mode, also include docs where this field is missing
            if not strict_mode:
                bool_filter["$or"].append({field_name: {"$exists": False}})
                
            pinecone_filter["$and"].append(bool_filter)
        except (ValueError, TypeError):
            pass
    
    @staticmethod
    def _add_language_filters(pinecone_filter: Dict[str, Any], language_filters: Dict[str, Any], strict_mode: bool):
        """Add language filters to the Pinecone filter."""
        for language, level in language_filters.items():
            language_key = language.capitalize()  # Ensure proper capitalization
            
            # Map proficiency level terms to standard values
            if level.lower() in ["native", "native-level", "native or bilingual proficiency", "mother tongue", "native speaker"]:
                proficiency_values = ["Native or Bilingual proficiency"]
            elif level.lower() in ["business", "business-level", "professional", "fluent", "professional working proficiency"]:
                proficiency_values = [
                    "Professional working proficiency",
                    "Full professional proficiency",
                    "Native or Bilingual proficiency"
                ]
            elif level.lower() in ["conversational", "intermediate", "limited working proficiency"]:
                proficiency_values = [
                    "Limited working proficiency",
                    "Professional working proficiency",
                    "Full professional proficiency",
                    "Native or Bilingual proficiency"
                ]
            elif level.lower() in ["basic", "elementary", "beginner", "elementary proficiency"]:
                proficiency_values = [
                    "Elementary Proficiency",
                    "Limited working proficiency",
                    "Professional working proficiency",
                    "Full professional proficiency",
                    "Native or Bilingual proficiency"
                ]
            else:
                # Default to all levels if unrecognized
                proficiency_values = [
                    "Elementary Proficiency",
                    "Limited working proficiency",
                    "Professional working proficiency",
                    "Full professional proficiency",
                    "Native or Bilingual proficiency"
                ]
            
            # Create language filter
            language_filter = {"$or": [
                {language_key: {"$in": proficiency_values}}
            ]}
            
            # In non-strict mode, also include docs where this language field doesn't exist
            if not strict_mode:
                language_filter["$or"].append({language_key: {"$exists": False}})
                
            pinecone_filter["$and"].append(language_filter) 