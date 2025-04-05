import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_profiles(profile_data: Dict[str, Dict[str, Any]], comparison_date: str = "2025-03-20") -> List[Dict[str, Any]]:
    """
    Preprocess raw profile data for LLM processing.
    
    Args:
        profile_data: Dictionary mapping profile IDs to profile data (output from ProfileRetriever)
        comparison_date: Date string to compare with updated_at (default: "2025-03-20")
        
    Returns:
        List of processed profile objects ready for LLM processing
    """
    if not profile_data:
        logger.warning("No profile data provided for preprocessing")
        return []
        
    logger.info(f"Preprocessing {len(profile_data)} profiles")
    
    processed_profiles = []
    
    # Process each profile
    for profile_id, profile in profile_data.items():
        try:
            processed_profile = {}
            
            # 1. Include profile_id
            processed_profile["profile_id"] = profile_id
            
            # Get data components for cleaner access
            tamago_data = profile.get("tamago_data", {})
            linkedin_data = profile.get("linkedin_data", {})
            metadata = profile.get("metadata", {})
            
            # 2. Consultant description (only if exists)
            if tamago_data.get("description"):
                processed_profile["consultant_description"] = tamago_data["description"]
            
            # 3. Created at
            if tamago_data.get("created_at"):
                processed_profile["created_at"] = tamago_data["created_at"]
            
            # 4. Updated at (with comparison logic)
            if tamago_data.get("updated_at"):
                tamago_updated = tamago_data["updated_at"]
                
                if linkedin_data:
                    processed_profile["updated_at"] = tamago_updated if tamago_updated > comparison_date else comparison_date
                else:
                    processed_profile["updated_at"] = tamago_updated
            
            # 5. Headline
            if linkedin_data and linkedin_data.get("headline"):
                processed_profile["headline"] = linkedin_data["headline"]
            elif tamago_data.get("headline"):
                processed_profile["headline"] = tamago_data["headline"]
            
            # 6. Nationality (only if not null)
            if tamago_data.get("nationality"):
                processed_profile["nationality"] = tamago_data["nationality"]
            
            # 7. Employments
            employments = process_employments(profile)
            if employments:
                processed_profile["employments"] = employments
            
            # 8. Notes
            if tamago_data.get("notes"):
                notes = [{"message": note.get("message"), "created_at": note.get("created_at")} 
                        for note in tamago_data["notes"] 
                        if note.get("message") and note.get("created_at")]
                if notes:
                    processed_profile["notes"] = notes
            
            # 9. Skills
            if tamago_data.get("tags_simplified"):
                skills = {}
                tags = tamago_data["tags_simplified"]
                for key in ["industry", "functional", "general"]:
                    if key in tags and tags[key]:
                        skills[key] = tags[key]
                if skills:
                    processed_profile["skills"] = skills
            
            # 10. Education
            if linkedin_data and linkedin_data.get("education"):
                processed_profile["education"] = linkedin_data["education"]
            
            # 11. Languages
            languages = extract_languages(metadata)
            if languages:
                processed_profile["languages"] = languages
            
            # 12. Certifications
            if linkedin_data and linkedin_data.get("certifications"):
                certifications = [{"name": cert.get("name")} 
                                for cert in linkedin_data["certifications"] 
                                if cert.get("name")]
                if certifications:
                    processed_profile["certifications"] = certifications
            
            # 13. LinkedIn description
            if linkedin_data and linkedin_data.get("summary"):
                processed_profile["linkedin_description"] = linkedin_data["summary"]
            
            processed_profiles.append(processed_profile)
            logger.debug(f"Successfully processed profile {profile_id}")
            
        except Exception as e:
            logger.error(f"Error processing profile {profile_id}: {str(e)}")
    
    logger.info(f"Successfully preprocessed {len(processed_profiles)} profiles")
    return processed_profiles

def process_employments(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process employment information based on availability of LinkedIn data.
    
    Args:
        profile: Complete profile data including tamago_data and linkedin_data
        
    Returns:
        List of processed employment entries
    """
    employments = []
    
    # Get data components for cleaner access
    tamago_data = profile.get("tamago_data", {})
    linkedin_data = profile.get("linkedin_data", {})
    
    # Handle case where LinkedIn data exists
    if linkedin_data and linkedin_data.get("work_experience"):
        for work in linkedin_data["work_experience"]:
            # Create a new employment entry without company_id and location
            employment = {k: v for k, v in work.items() if k not in ["company_id", "location"]}
            employments.append(employment)
    
    # Handle case where only Tamago data exists
    elif tamago_data.get("employments"):
        for emp in tamago_data["employments"]:
            # Skip entries without company_name
            if not emp.get("company_name"):
                continue
                
            # Create employment with only specified fields
            employment = {
                "position": emp.get("position"),
                "company_name": emp.get("company_name"),
                "description": emp.get("description"),
                "start_date": emp.get("start_date"),
                "end_date": emp.get("end_date")
            }
            
            # Filter out None values
            employment = {k: v for k, v in employment.items() if v is not None}
            employments.append(employment)
    
    return employments

def extract_languages(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract language fields from metadata.
    
    Args:
        metadata: Metadata dictionary from profile entry
        
    Returns:
        Dictionary of languages where keys are language names and values are proficiency levels
    """
    # List of keys to exclude
    excluded_keys = [
        "profile_id", "section", "placed", "years_of_experience",
        "gender", "last_contacted", "is_candidate", 
        "processed_at", "keywords", "position"
    ]
    
    # Filter out excluded keys
    languages = {k: v for k, v in metadata.items() if k not in excluded_keys}
    return languages

# For direct testing of this module
if __name__ == "__main__":
    try:
        # Read raw profile data from file
        with open('profile_retrieved_output.json', 'r') as f:
            raw_profiles = json.load(f)
            
        # Preprocess profiles
        processed_profiles = preprocess_profiles(raw_profiles)
        
        # Output results for debugging
        output_file = 'processed_profiles.json'
        with open(output_file, 'w') as f:
            json.dump(processed_profiles, f, indent=2)
            
        logger.info(f"Processed profiles saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("profile_retrieved_output.json file not found")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in profile_retrieved_output.json")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}") 