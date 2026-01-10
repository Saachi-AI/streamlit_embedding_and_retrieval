import os
import sys
import json
from typing import Dict, List, Any
from pinecone import Pinecone
import time
import requests  # Add this import for direct API calls

from utils import load_environment

def load_environment_vars():
    """Load environment variables"""
    try:
        return load_environment()
    except Exception as e:
        print(f"Failed to load environment variables: {str(e)}")
        sys.exit(1)

def connect_to_pinecone(env_vars):
    """Initialize Pinecone connection"""
    try:
        pc = Pinecone(api_key=env_vars["pinecone_api_key"])
        index_name = env_vars["pinecone_index_name"]
        index = pc.Index(
            name=index_name,
            host=env_vars["pinecone_host"] if "pinecone_host" in env_vars else None
        )
        stats = index.describe_index_stats()
        print(f"Connected to Pinecone index '{index_name}' with {stats['total_vector_count']} total vectors")
        return index
    except Exception as e:
        print(f"Failed to connect to Pinecone: {str(e)}")
        sys.exit(1)

def get_profile_chunks(index, profile_id: int, debug: bool = False) -> List[Dict[str, Any]]:
    """Get all chunks for a profile ID using multiple approaches to ensure all data is retrieved"""
    try:
        # Convert profile_id to integer format to match database
        int_profile_id = int(profile_id)
        
        if debug:
            print(f"Using multiple approaches to find ALL chunks for profile_id: {int_profile_id}")
        
        # Get environment variables for API access
        env_vars = load_environment()
        api_key = env_vars["pinecone_api_key"]
        index_name = env_vars["pinecone_index_name"]
        
        # Get the index host/endpoint
        pc = Pinecone(api_key=api_key)
        index_info = pc.describe_index(index_name)
        host = index_info.get('host') or env_vars.get("pinecone_host")
        
        if not host:
            raise ValueError("Could not determine Pinecone host URL")
        
        # Prepare the API endpoint
        if not host.startswith('http'):
            host = f"https://{host}"
            
        api_endpoint = f"{host}/query"
        
        # Filter for the profile ID
        filter_dict = {"profile_id": int_profile_id}
        
        if debug:
            print(f"Using endpoint: {api_endpoint}")
            print(f"Using filter: {filter_dict}")
        
        # Get stats to determine vector dimension
        stats = index.describe_index_stats()
        vector_dimension = stats.get('dimension', 1536)  # Default to 1536 if not found
        namespace = ""  # default namespace
        
        all_matches = []
        
        # APPROACH 1: Use the SDK directly with a very high limit
        if debug:
            print("\nAPPROACH 1: Using SDK with high top_k value...")
            approach1_start = time.time()
        
        try:
            sdk_results = index.query(
                vector=[0.0] * vector_dimension,
                filter=filter_dict,
                top_k=10000,  # Very high to try to get all
                include_metadata=True
            )
            
            sdk_matches = sdk_results.get('matches', [])
            all_matches.extend(sdk_matches)
            
            if debug:
                approach1_time = time.time() - approach1_start
                print(f"  Found {len(sdk_matches)} matches in {approach1_time:.2f} seconds")
                print(f"  Section distribution:")
                sections = {}
                for match in sdk_matches:
                    section = match.get('metadata', {}).get('section', 'unknown')
                    sections[section] = sections.get(section, 0) + 1
                for section, count in sections.items():
                    print(f"    - {section}: {count} chunks")
        except Exception as e:
            if debug:
                print(f"  SDK approach failed: {str(e)}")
        
        # APPROACH 2: Direct API calls with different vector directions
        if debug:
            print("\nAPPROACH 2: Using direct API with multiple vector directions...")
            approach2_start = time.time()
        
        # Try multiple different direction vectors to ensure we get all chunks
        # This helps overcome potential vector-similarity-based limitations
        vectors_to_try = [
            [0.0] * vector_dimension,  # Zero vector
            [1.0] * vector_dimension,  # All ones
            [-1.0] * vector_dimension,  # All negative ones
            # Add some random directions
            [1.0 if i % 2 == 0 else -1.0 for i in range(vector_dimension)],
            [-1.0 if i % 2 == 0 else 1.0 for i in range(vector_dimension)]
        ]
        
        api_matches = []
        seen_ids = set()
        
        # Prepare headers
        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
        
        # Try each vector direction
        for vector_idx, query_vector in enumerate(vectors_to_try):
            if debug:
                print(f"  Trying vector direction {vector_idx+1}...")
            
            payload = {
                "vector": query_vector,
                "filter": filter_dict,
                "topK": 10000,  # High limit
                "includeMetadata": True,
                "namespace": namespace
            }
            
            try:
                response = requests.post(api_endpoint, headers=headers, json=payload)
                
                if response.status_code == 200:
                    results = response.json()
                    matches = results.get('matches', [])
                    
                    # Add only new matches (not seen before) based on ID
                    new_matches = []
                    for match in matches:
                        match_id = match.get('id')
                        if match_id and match_id not in seen_ids:
                            seen_ids.add(match_id)
                            new_matches.append(match)
                    
                    api_matches.extend(new_matches)
                    
                    if debug:
                        print(f"    Found {len(matches)} total matches, {len(new_matches)} new matches")
                else:
                    if debug:
                        print(f"    API request failed: {response.status_code} - {response.text}")
            except Exception as e:
                if debug:
                    print(f"    Error with vector {vector_idx+1}: {str(e)}")
        
        # Add unique API matches to our collection
        all_api_ids = set(m.get('id') for m in all_matches)
        for match in api_matches:
            if match.get('id') not in all_api_ids:
                all_matches.append(match)
                all_api_ids.add(match.get('id'))
        
        if debug:
            approach2_time = time.time() - approach2_start
            print(f"  Found {len(api_matches)} unique matches in {approach2_time:.2f} seconds")
            print(f"  Combined unique matches so far: {len(all_matches)}")
        
        # APPROACH 3: Try the stats to see all namespaces and attempt to fetch from each
        if debug:
            print("\nAPPROACH 3: Checking index stats for namespaces...")
            approach3_start = time.time()
        
        try:
            # Get all namespaces from stats
            available_namespaces = list(stats.get('namespaces', {}).keys())
            
            if debug:
                print(f"  Found {len(available_namespaces)} namespaces: {available_namespaces}")
            
            # If there are namespaces besides default, try each one
            namespace_matches = []
            seen_namespace_ids = set(all_api_ids)  # Start with IDs we've already seen
            
            for ns in available_namespaces:
                if debug:
                    print(f"  Querying namespace: '{ns}'")
                
                payload = {
                    "vector": [0.0] * vector_dimension,
                    "filter": filter_dict,
                    "topK": 10000,
                    "includeMetadata": True,
                    "namespace": ns
                }
                
                try:
                    response = requests.post(api_endpoint, headers=headers, json=payload)
                    
                    if response.status_code == 200:
                        results = response.json()
                        matches = results.get('matches', [])
                        
                        # Add only new matches
                        new_matches = []
                        for match in matches:
                            match_id = match.get('id')
                            if match_id and match_id not in seen_namespace_ids:
                                seen_namespace_ids.add(match_id)
                                new_matches.append(match)
                        
                        namespace_matches.extend(new_matches)
                        
                        if debug:
                            print(f"    Found {len(matches)} total matches, {len(new_matches)} new matches")
                    else:
                        if debug:
                            print(f"    API request failed: {response.status_code}")
                except Exception as e:
                    if debug:
                        print(f"    Error with namespace '{ns}': {str(e)}")
            
            # Add unique namespace matches to our collection
            for match in namespace_matches:
                if match.get('id') not in all_api_ids:
                    all_matches.append(match)
                    all_api_ids.add(match.get('id'))
            
            if debug:
                approach3_time = time.time() - approach3_start
                print(f"  Found {len(namespace_matches)} additional matches in {approach3_time:.2f} seconds")
        except Exception as e:
            if debug:
                print(f"  Namespace approach failed: {str(e)}")
        
        if debug:
            print(f"\nFinal total: {len(all_matches)} unique matches found across all approaches")
            # Count chunks by section
            sections = {}
            for match in all_matches:
                section = match.get('metadata', {}).get('section', 'unknown')
                sections[section] = sections.get(section, 0) + 1
            print("Section distribution in final results:")
            for section, count in sections.items():
                print(f"  - {section}: {count} chunks")
        
        # Extract the chunks from all_matches
        chunks = []
        for match in all_matches:
            vector_id = match.get('id', '')
            metadata = match.get('metadata', {})
            
            # Get the chunk text - try different possible locations
            chunk_text = None
            
            # 1. Try to get chunk_text directly from metadata
            if 'chunk_text' in metadata:
                chunk_text = metadata.pop('chunk_text')  # Remove from metadata to avoid duplication
            
            # 2. Check if text is directly in match
            elif 'text' in match:
                chunk_text = match['text']
            
            # 3. Look for text field in metadata
            elif 'text' in metadata:
                chunk_text = metadata.pop('text')  # Remove from metadata to avoid duplication
            
            # 4. Try to find any field that might contain the text
            if not chunk_text:  # Only if we haven't found text yet
                for key, value in list(metadata.items()):  # Use list() to avoid modification during iteration
                    if isinstance(value, str) and len(value) > 50:
                        chunk_text = value
                        metadata.pop(key)  # Remove from metadata to avoid duplication
                        break
            
            # If we still don't have text, use a default message
            if not chunk_text:
                chunk_text = "No text available"
            
            chunks.append({
                "id": vector_id,
                "metadata": metadata,
                "chunk_text": chunk_text
            })
        
        # Sort chunks by section order
        section_order = [
            "consultant_description",
            "linkedin_profile_description",
            "experience",
            "skills",
            "education",
            "notes",
            "projects"
        ]
        
        def get_section_order(chunk):
            section = chunk.get('metadata', {}).get('section', '')
            try:
                return section_order.index(section)
            except ValueError:
                return len(section_order)  # Put unknown sections at the end
        
        # Sort chunks first by section order, then by position within experience sections
        chunks.sort(key=lambda x: (
            get_section_order(x),
            x.get('metadata', {}).get('position', '') if x.get('metadata', {}).get('section') == 'experience' else ''
        ))
        
        return chunks
    except Exception as e:
        print(f"Error retrieving profile chunks: {str(e)}")
        if debug:
            import traceback
            print(f"Error details:\n{traceback.format_exc()}")
        return []

def display_chunks(chunks, index):
    """Display chunks in a readable format"""
    if not chunks:
        print("No chunks found for this profile ID.")
        return
    
    # Get current index stats
    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    
    print(f"Found {len(chunks)} chunks for this profile (out of {total_vectors} total vectors in the index)\n")
    
    for i, chunk in enumerate(chunks):
        section = chunk.get('metadata', {}).get('section', 'Unknown')
        position = chunk.get('metadata', {}).get('position', '')
        company = chunk.get('metadata', {}).get('company', '')
        vector_id = chunk.get('id', 'No ID')
        
        # Create a descriptive header
        if section == 'experience' and position and company:
            header = f"Chunk {i+1} - {section}: {position} at {company}"
        else:
            header = f"Chunk {i+1} - {section}"
        
        print("=" * 80)
        print(header)
        print(f"Vector ID: {vector_id}")
        print("-" * 80)
        print("Text:")
        print(chunk.get("chunk_text", "No text available"))
        print("-" * 80)
        print("Metadata:")
        print(json.dumps(chunk.get("metadata", {}), indent=2))
        print()

def main():
    """
    Main function to run the script.
    
    Usage:
      python get_all_chunks.py [profile_id] [debug_flag]
    
    Examples:
      python get_all_chunks.py                 # Uses default profile ID 59957 with debug mode
      python get_all_chunks.py 32138           # Gets chunks for profile ID 32138 with debug mode
      python get_all_chunks.py 32138 nodebug   # Gets chunks for profile ID 32138 without debug output
    """
    # Default profile ID from the original code
    profile_id = 59957
    debug_mode = True  # Enable debug mode by default
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        profile_id = int(sys.argv[1])
    
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['nodebug', '--nodebug', '-nd']:
        debug_mode = False
    
    print(f"Fetching chunks for Profile ID: {profile_id} (Debug mode: {debug_mode})")
    print("This script will try multiple approaches to ensure ALL chunks are retrieved.")
    print("To run: python get_all_chunks.py [profile_id] [nodebug]")
    
    # Load environment and connect to Pinecone
    env_vars = load_environment_vars()
    index = connect_to_pinecone(env_vars)
    
    # Get and display chunks
    chunks = get_profile_chunks(index, profile_id, debug=debug_mode)
    display_chunks(chunks, index)

if __name__ == "__main__":
    main() 