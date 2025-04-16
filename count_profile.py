import os
import sys
import json
from typing import Dict, List, Any
from pinecone import Pinecone
import time

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
        index = pc.Index(
            name=env_vars["pinecone_index_name"],
            host=env_vars["pinecone_host"] if "pinecone_host" in env_vars else None
        )
        stats = index.describe_index_stats()
        print(f"Connected to Pinecone index with {stats['total_vector_count']} total vectors")
        return index
    except Exception as e:
        print(f"Failed to connect to Pinecone: {str(e)}")
        sys.exit(1)

def get_profile_chunks(index, profile_id: int, debug: bool = False) -> List[Dict[str, Any]]:
    """Get all chunks for a profile ID"""
    try:
        # Convert profile_id to integer format to match database
        int_profile_id = int(profile_id)
        
        if debug:
            print(f"Querying for profile_id: {int_profile_id} (type: {type(int_profile_id)})")
            query_start_time = time.time()
        
        # Query with numeric profile ID
        filter_dict = {"profile_id": int_profile_id}
        
        if debug:
            print(f"Using filter: {filter_dict}")
        
        results = index.query(
            vector=[0.0] * 1024,  # 1024 dimensions to match the index
            filter=filter_dict,
            top_k=10000,  # Get up to 10000 chunks per profile
            include_metadata=True
        )
        
        if debug:
            query_time = time.time() - query_start_time
            print(f"Query completed in {query_time:.2f} seconds")
            print(f"Found {len(results['matches'])} matches")
        
        # Extract the chunks
        chunks = []
        for match in results['matches']:
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

def display_chunks(chunks):
    """Display chunks in a readable format"""
    if not chunks:
        print("No chunks found for this profile ID.")
        return
    
    print(f"Found {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks):
        section = chunk.get('metadata', {}).get('section', 'Unknown')
        position = chunk.get('metadata', {}).get('position', '')
        company = chunk.get('metadata', {}).get('company', '')
        
        # Create a descriptive header
        if section == 'experience' and position and company:
            header = f"Chunk {i+1} - {section}: {position} at {company}"
        else:
            header = f"Chunk {i+1} - {section}"
        
        print("=" * 80)
        print(header)
        print("-" * 80)
        print("Text:")
        print(chunk.get("chunk_text", "No text available"))
        print("-" * 80)
        print("Metadata:")
        print(json.dumps(chunk.get("metadata", {}), indent=2))
        print()

def main():
    # Default profile ID from the original code
    profile_id = 59957
    debug_mode = False
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        profile_id = int(sys.argv[1])
    
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['debug', '--debug', '-d']:
        debug_mode = True
    
    print(f"Fetching chunks for Profile ID: {profile_id} (Debug mode: {debug_mode})")
    
    # Load environment and connect to Pinecone
    env_vars = load_environment_vars()
    index = connect_to_pinecone(env_vars)
    
    # Get and display chunks
    chunks = get_profile_chunks(index, profile_id, debug=debug_mode)
    display_chunks(chunks)

if __name__ == "__main__":
    main() 