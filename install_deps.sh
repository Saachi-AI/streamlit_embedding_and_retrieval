#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install basic packages first
pip install python-dotenv streamlit typing-extensions argparse

# Install embedding providers
pip install openai==1.6.0
pip install cohere>=4.32.0

# Install Pinecone client
pip install pinecone-client==3.0.0

# Install LangSmith
pip install langsmith==0.0.63

# Install LangChain dependencies
pip install langchain-core==0.1.32
pip install langchain==0.0.339
pip install langchain-openai==0.0.5
pip install langchain-pinecone==0.1.0

# Install pydantic
pip install "pydantic<3.0.0,>=2.0.0"

echo "All dependencies installed successfully!" 