import os
from dotenv import load_dotenv
import sys
sys.path.append('.')  # Add current directory to path

# Import the load_environment function from utils.py
from utils import load_environment

# Print raw environment variables from os.environ
print("\n=== Environment variables before loading .env ===")
print(f"PINECONE_API_KEY: {os.environ.get('PINECONE_API_KEY', 'Not set')[:10]}... (if set)")
print(f"PINECONE_INDEX_NAME: {os.environ.get('PINECONE_INDEX_NAME', 'Not set')}")
print(f"PINECONE_HOST: {os.environ.get('PINECONE_HOST', 'Not set')}")

# Load via dotenv directly
print("\n=== After loading with dotenv.load_dotenv() ===")
load_dotenv()
print(f"PINECONE_API_KEY: {os.environ.get('PINECONE_API_KEY', 'Not set')[:10]}... (if set)")
print(f"PINECONE_INDEX_NAME: {os.environ.get('PINECONE_INDEX_NAME', 'Not set')}")
print(f"PINECONE_HOST: {os.environ.get('PINECONE_HOST', 'Not set')}")

# Now load using the application's load_environment function
print("\n=== Using utils.load_environment() ===")
env_vars = load_environment()
print(f"pinecone_api_key: {env_vars.get('pinecone_api_key', 'Not in dict')[:10]}... (if in dict)")
print(f"pinecone_index_name: {env_vars.get('pinecone_index_name', 'Not in dict')}")
print(f"pinecone_host: {env_vars.get('pinecone_host', 'Not in dict')}")
print(f"pinecone_environment: {env_vars.get('pinecone_environment', 'Not in dict')}")

# Print the full .env file content (without sensitive values)
print("\n=== .env file contents (first 10 chars of sensitive values) ===")
try:
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key or 'TOKEN' in key:
                    # Show only first 10 chars of sensitive values
                    print(f"{key}={value[:10]}...")
                else:
                    print(f"{key}={value}")
except Exception as e:
    print(f"Error reading .env file: {e}")

# Check for multiple .env files in the project directory
print("\n=== Searching for multiple .env files in project directory ===")
try:
    import subprocess
    result = subprocess.run(['find', '.', '-name', '.env', '-type', 'f'], capture_output=True, text=True)
    env_files = result.stdout.strip().split('\n')
    print(f"Found {len(env_files)} .env files:")
    for file in env_files:
        print(f"  - {file}")
except Exception as e:
    print(f"Error searching for .env files: {e}") 