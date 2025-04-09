import os
import json
import logging
import boto3
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Import preprocessor module (for integration)
try:
    from profile_preprocessor import preprocess_profiles
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("Profile preprocessor module not available. Preprocessing will be skipped.")
    PREPROCESSOR_AVAILABLE = False

class ProfileRetriever:
    """
    A class to retrieve additional profile information from DynamoDB tables.
    """
    
    def __init__(self):
        """Initialize with AWS credentials and DynamoDB table names from environment variables."""
        # Load AWS credentials and configuration from environment variables
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-northeast-1")
        
        # Load DynamoDB table names from environment variables
        self.tamago_table_name = os.getenv("TAMAGO_DYNAMODB_TABLE", "tamago_profiles")
        self.linkedin_table_name = os.getenv("LINKEDIN_DYNAMODB_TABLE", "linkedin_profiles")
        
        # Initialize DynamoDB resource
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Get table references
        self.tamago_table = self.dynamodb.Table(self.tamago_table_name)
        self.linkedin_table = self.dynamodb.Table(self.linkedin_table_name)
        
        logger.info(f"ProfileRetriever initialized with tables: {self.tamago_table_name}, {self.linkedin_table_name}")
    
    def get_tamago_profile(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve profile data from tamago_profiles table.
        
        Args:
            person_id: The profile/person ID to retrieve
            
        Returns:
            Dictionary containing profile data or None if not found
        """
        try:
            response = self.tamago_table.get_item(
                Key={
                    'person_id': person_id
                }
            )
            
            # Check if the item was found
            if 'Item' in response:
                profile_data = response['Item'].get('profile_data')
                # Parse the JSON string if it's a string
                if isinstance(profile_data, str):
                    try:
                        return json.loads(profile_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing tamago profile JSON for {person_id}: {str(e)}")
                        return profile_data
                return profile_data
            else:
                logger.warning(f"Profile {person_id} not found in tamago_profiles table")
                return None
                
        except ClientError as e:
            logger.error(f"Error retrieving tamago profile for {person_id}: {str(e)}")
            return None
    
    def get_linkedin_profile(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve profile data from linkedin_profiles table.
        
        Args:
            person_id: The profile/person ID to retrieve
            
        Returns:
            Dictionary containing profile data and fetched_at timestamp or None if not found
        """
        try:
            response = self.linkedin_table.get_item(
                Key={
                    'person_id': person_id
                }
            )
            
            # Check if the item was found
            if 'Item' in response:
                profile_data = response['Item'].get('profile_data')
                timestamp = response['Item'].get('timestamp')
                
                # Parse the JSON string if it's a string
                if isinstance(profile_data, str):
                    try:
                        profile_data = json.loads(profile_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing LinkedIn profile JSON for {person_id}: {str(e)}")
                        return profile_data
                
                # Add fetched_at timestamp if available
                if timestamp:
                    try:
                        # Convert timestamp to YYYY-MM-DD format
                        # Handle different input formats
                        if isinstance(timestamp, str):
                            # If it's already in YYYY-MM-DD format
                            if re.match(r'^\d{4}-\d{2}-\d{2}$', timestamp):
                                profile_data['fetched_at'] = timestamp
                            else:
                                # Try to parse and format the date
                                parsed_date = datetime.strptime(timestamp.split('T')[0], '%Y-%m-%d')
                                profile_data['fetched_at'] = parsed_date.strftime('%Y-%m-%d')
                        else:
                            # Handle other timestamp formats if needed
                            logger.warning(f"Unexpected timestamp format for LinkedIn profile {person_id}: {timestamp}")
                    except Exception as e:
                        logger.warning(f"Error processing timestamp for LinkedIn profile {person_id}: {str(e)}")
                else:
                    logger.warning(f"LinkedIn profile {person_id} missing fetched_at timestamp")
                
                return profile_data
            else:
                logger.info(f"Profile {person_id} not found in linkedin_profiles table")
                return None
                
        except ClientError as e:
            logger.error(f"Error retrieving LinkedIn profile for {person_id}: {str(e)}")
            return None
    
    def retrieve_profile_data(
        self, 
        profile_entries: List[Dict[str, Any]], 
        preprocess: bool = False
    ) -> Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve profile data from both DynamoDB tables for a list of profile entries.
        
        Args:
            profile_entries: List of dictionaries containing:
                - 'profile_id': ID to retrieve from DynamoDB
                - 'metadata': Metadata about the profile from the relevant chunk
            preprocess: Whether to preprocess the retrieved data for LLM (default: False)
                
        Returns:
            If preprocess=False:
                Dictionary mapping each profile ID to its merged data with keys:
                - 'metadata': Metadata from the input
                - 'tamago_data': Data from tamago_profiles table (always expected)
                - 'linkedin_data': Data from linkedin_profiles table (optional)
            If preprocess=True:
                List of processed profile objects ready for LLM processing
        """
        if not profile_entries:
            logger.warning("No profile entries provided for retrieval")
            return {} if not preprocess else []
            
        logger.info(f"Retrieving data for {len(profile_entries)} profiles")
        
        # Dictionary to store results
        results = {}
        
        # Process each profile entry
        for entry in profile_entries:
            profile_id = entry.get('profile_id')
            metadata = entry.get('metadata', {})
            
            if not profile_id:
                logger.warning(f"Skipping entry with missing profile_id: {entry}")
                continue
                
            # Get data from both tables
            tamago_data = self.get_tamago_profile(profile_id)
            linkedin_data = self.get_linkedin_profile(profile_id)
            
            # Only include profiles that exist in tamago_profiles (required table)
            if tamago_data:
                results[profile_id] = {
                    "metadata": metadata,
                    "tamago_data": tamago_data,
                    "linkedin_data": linkedin_data if linkedin_data else {}
                }
                logger.info(f"Retrieved data for profile {profile_id}: " +
                           f"tamago_data={True}, linkedin_data={linkedin_data is not None}")
            else:
                logger.warning(f"Profile {profile_id} not found in tamago_profiles table - skipping")
        
        # Apply preprocessing if requested and available
        if preprocess and PREPROCESSOR_AVAILABLE:
            logger.info("Preprocessing profile data for LLM")
            return preprocess_profiles(results)
        elif preprocess and not PREPROCESSOR_AVAILABLE:
            logger.warning("Preprocessing requested but preprocessor module not available. Returning raw data.")
            
        return results

# For direct testing of this module
if __name__ == "__main__":
    # Read profile entries from a test config file
    try:
        with open('test_config.json', 'r') as f:
            config = json.load(f)
            test_profile_entries = config.get('profile_entries', [])
            
        if not test_profile_entries:
            logger.error("No profile entries found in test_config.json")
            exit(1)
            
        logger.info(f"Test config loaded with {len(test_profile_entries)} profile entries")
        
        # Initialize the retriever
        retriever = ProfileRetriever()
        
        # Check whether to preprocess data
        preprocess_data = os.getenv("PREPROCESS_DATA", "false").lower() == "true"
        
        # Retrieve profile data (with optional preprocessing)
        profile_data = retriever.retrieve_profile_data(
            test_profile_entries, 
            preprocess=preprocess_data
        )
        
        # Determine output filename based on preprocessing
        output_file = 'processed_profiles.json' if preprocess_data else 'profile_retrieved_output.json'
        
        # Output results for debugging
        print(json.dumps(profile_data, indent=2, ensure_ascii=False))
        
        # Save results to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully retrieved data for {len(profile_data)} profiles")
        logger.info(f"Results saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("test_config.json file not found. Create it with format: {\"profile_entries\": [{\"profile_id\": \"id1\", \"metadata\": {...}}, ...]}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in test_config.json file")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}") 