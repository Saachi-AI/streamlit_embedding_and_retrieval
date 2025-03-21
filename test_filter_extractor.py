import os
import json
from dotenv import load_dotenv
from filter_extractor import FilterExtractor

# Load environment variables
load_dotenv()

# Sample test queries
test_queries = [
    "Find male candidates with 5+ years of experience",
    "I need female candidates with business-level English proficiency",
    "Show me candidates who are fluent in Japanese and have been contacted in the last 2 years",
    "Find unplaced candidates with at least 8 years of experience who speak professional-level Hindi",
    "I'm looking for only male candidates with native-level French and 10 years of experience",
    "Show active candidates with native-level Chinese who have not been placed"
]

# Initialize filter extractor
filter_extractor = FilterExtractor(api_key=os.getenv("GROQ_API_KEY"))

# Test each query
for i, query in enumerate(test_queries):
    print(f"\n--- Test Query {i+1} ---")
    print(f"Query: {query}")
    
    try:
        # Process the query
        result = filter_extractor.process_query(query)
        extracted_filters = result["extracted_filters"]
        pinecone_filter = result["pinecone_filter"]
        
        # Print extracted filters
        print("\nExtracted Filters:")
        print(json.dumps(extracted_filters, indent=2))
        
        # Print Pinecone filter
        print("\nPinecone Filter:")
        print(json.dumps(pinecone_filter, indent=2))
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
    
    print("-" * 80)

print("\nFilter extraction testing complete!") 