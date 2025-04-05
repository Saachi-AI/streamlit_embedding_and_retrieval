import os
import json
import logging
import boto3
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

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
            Dictionary containing profile data or None if not found
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
                # Parse the JSON string if it's a string
                if isinstance(profile_data, str):
                    try:
                        return json.loads(profile_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing LinkedIn profile JSON for {person_id}: {str(e)}")
                        return profile_data
                return profile_data
            else:
                logger.info(f"Profile {person_id} not found in linkedin_profiles table")
                return None
                
        except ClientError as e:
            logger.error(f"Error retrieving LinkedIn profile for {person_id}: {str(e)}")
            return None
    
    def retrieve_profile_data(self, profile_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve profile data from both DynamoDB tables for a list of profile entries.
        
        Args:
            profile_entries: List of dictionaries containing:
                - 'profile_id': ID to retrieve from DynamoDB
                - 'metadata': Metadata about the profile from the relevant chunk
                
        Returns:
            Dictionary mapping each profile ID to its merged data with keys:
            - 'metadata': Metadata from the input
            - 'tamago_data': Data from tamago_profiles table (always expected)
            - 'linkedin_data': Data from linkedin_profiles table (optional)
        """
        if not profile_entries:
            logger.warning("No profile entries provided for retrieval")
            return {}
            
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
        
        # Retrieve profile data
        profile_data = retriever.retrieve_profile_data(test_profile_entries)
        
        # Output results for debugging
        print(json.dumps(profile_data, indent=2))
        
        # Save results to a JSON file
        output_file = 'profile_retrieved_output.json'
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
            
        logger.info(f"Successfully retrieved data for {len(profile_data)} profiles")
        logger.info(f"Results saved to {output_file}")
        
    except FileNotFoundError:
        logger.error("test_config.json file not found. Create it with format: {\"profile_entries\": [{\"profile_id\": \"id1\", \"metadata\": {...}}, ...]}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in test_config.json file")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}") 