# Modular RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline built with LangChain for embedding and retrieval, and LangSmith for monitoring.

## Project Structure

- `embedders/`: Contains modular embedding implementations
  - `openai_embedder.py`: OpenAI embedding implementation
  - `cohere_embedder.py`: Cohere embedding implementation
  - `embed.py`: Main script to run the embedding process
- `app.py`: Streamlit application for retrieval and visualization
- `utils.py`: Shared utility functions
- `.env`: Configuration file for API keys

## Setup Instructions

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Configure your API keys in the `.env` file.

3. Embed documents into Pinecone:
   ```
   python embedders/embed.py --model openai
   ```
   or
   ```
   python embedders/embed.py --model cohere
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Features

- Modular embedding system supporting multiple models
- Streamlit UI for querying and retrieving relevant chunks
- LangSmith integration for monitoring embedding and retrieval performance 