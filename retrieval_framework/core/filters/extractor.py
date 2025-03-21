"""
Filter extractor implementation for the retrieval framework.

Uses language models to extract structured filters from natural language queries.
"""

import json
import time
from typing import Dict, Any, Optional

from langchain_groq import ChatGroq

from retrieval_framework.core.filters.base import FilterExtractorBase
from retrieval_framework.core.filters.adapters.pinecone import PineconeFilterAdapter
from retrieval_framework.config.constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from retrieval_framework.utils import get_logger, FilterExtractionError, retry_with_backoff

logger = get_logger(__name__)


class FilterExtractor(FilterExtractorBase):
    """
    Extracts metadata filters from natural language queries using language models.
    
    Currently uses Groq's API with DeepSeek R1 Distill Llama 70B by default.
    """
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "deepseek-r1-distill-llama-70b",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        """
        Initialize with Groq API key and model settings.
        
        Args:
            api_key: Groq API key
            model_name: Model to use for extraction
            temperature: Temperature for LLM (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initializing FilterExtractor with model: {model_name}")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    @retry_with_backoff(logger=logger)
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filters from natural language query using LLM.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary of extracted filters
            
        Raises:
            FilterExtractionError: If extraction fails
        """
        # Define the prompt template
        prompt = f"""You are a metadata extraction assistant. Extract the following fields from the user's candidate search query:

- gender: Extract if user specifies "male" or "female"
- years_of_experience: Extract numeric value if the user mentions years of experience
- last_contacted: Extract time period in years if user mentions when candidates were last contacted
- is_candidate: Extract boolean if user explicitly mentions active candidate status
- placed: Extract boolean if user explicitly mentions placement status
- languages: Extract any natural/human language requirements (like English, Japanese, Spanish) with their proficiency levels

Rules:
1. Leave a field empty if not mentioned in the query
2. For gender, note if the query indicates "only male/female" or if it could include unspecified
3. For experience, extract the minimum years as a number
4. For last_contacted, extract the time period in years
5. For language proficiency levels, map to standard values:
   - "native-level", "mother tongue", "native speaker" -> "Native or Bilingual proficiency"
   - "business-level", "professional", "fluent" -> "Professional working proficiency"
   - "conversational", "intermediate" -> "Limited working proficiency"
   - "basic", "elementary", "beginner" -> "Elementary Proficiency"
6. Return languages as an object with language names as keys and proficiency levels as values
7. If proficiency level is not specified for a language, assume "Professional working proficiency"
8. Only include human/natural languages like English, French, Japanese, etc. Do not include programming languages.

USER QUERY: "{query}"

Return a JSON object with only the fields that were mentioned in the query.
"""
        
        logger.info(f"Extracting filters from query: {query[:50]}...")
        start_time = time.time()
        
        # Get response from LLM
        try:
            response = self.llm.invoke(prompt).content
            duration = time.time() - start_time
            logger.info(f"Filter extraction completed in {duration:.2f}s")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    extracted_filters = json.loads(json_str)
                    logger.info(f"Successfully extracted filters: {list(extracted_filters.keys())}")
                    return extracted_filters
                else:
                    logger.warning("LLM response did not contain valid JSON. Using empty filter.")
                    return {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise FilterExtractionError(f"Failed to extract filters with Groq API: {str(e)}")
    
    def process_query(self, query: str, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query and return Pinecone filter.
        
        Args:
            query: Natural language query string
            strict_mode: If True, only include exact matches in the filter
            
        Returns:
            Dict containing extracted filters and Pinecone filter
            
        Raises:
            FilterExtractionError: If query processing fails
        """
        try:
            # Extract filters
            extracted_filters = self.extract_filters(query)
            
            # Build Pinecone filter
            pinecone_filter = PineconeFilterAdapter.build_filter(extracted_filters, strict_mode)
            
            # Return both
            return {
                "extracted_filters": extracted_filters,
                "pinecone_filter": pinecone_filter
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise FilterExtractionError(f"Failed to process query: {str(e)}") 