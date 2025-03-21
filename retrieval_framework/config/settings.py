"""
Configuration settings for the retrieval framework.
Uses Pydantic for validation and environment variable loading.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PineconeSettings(BaseModel):
    """Pinecone vector database settings."""
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field("us-east-1", description="Pinecone environment/region")
    index_name: str = Field(..., description="Pinecone index name")
    host: Optional[str] = Field(None, description="Optional Pinecone host URL")

class OpenAISettings(BaseModel):
    """OpenAI API settings."""
    api_key: str = Field(..., description="OpenAI API key")

class CohereSettings(BaseModel):
    """Cohere API settings."""
    api_key: str = Field(..., description="Cohere API key")

class GroqSettings(BaseModel):
    """Groq API settings."""
    api_key: str = Field(..., description="Groq API key")

class LangChainSettings(BaseModel):
    """LangChain API settings."""
    api_key: str = Field(..., description="LangChain API key")
    project: Optional[str] = Field(None, description="LangChain project name")

class Settings(BaseModel):
    """Main application settings."""
    pinecone: PineconeSettings
    openai: OpenAISettings
    cohere: CohereSettings
    groq: GroqSettings
    langchain: LangChainSettings

def get_settings() -> Settings:
    """
    Load and validate settings from environment variables.
    
    Returns:
        Settings: Validated settings object
    
    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        settings = Settings(
            pinecone=PineconeSettings(
                api_key=os.getenv("PINECONE_API_KEY", ""),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                index_name=os.getenv("PINECONE_INDEX_NAME", ""),
                host=os.getenv("PINECONE_HOST")
            ),
            openai=OpenAISettings(
                api_key=os.getenv("OPENAI_API_KEY", "")
            ),
            cohere=CohereSettings(
                api_key=os.getenv("COHERE_API_KEY", "")
            ),
            groq=GroqSettings(
                api_key=os.getenv("GROQ_API_KEY", "")
            ),
            langchain=LangChainSettings(
                api_key=os.getenv("LANGCHAIN_API_KEY", ""),
                project=os.getenv("LANGCHAIN_PROJECT")
            )
        )
        return settings
    except Exception as e:
        raise ValueError(f"Error loading settings: {str(e)}") 