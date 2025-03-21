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

# Streamlit Embedding and Retrieval

This project demonstrates a powerful RAG (Retrieval-Augmented Generation) system with advanced metadata filtering capabilities. The application allows users to search for candidates based on semantic similarity and specific metadata criteria such as gender, experience, language proficiency, and more.

## Features

- **Semantic search** using OpenAI or Cohere embeddings
- **LLM-powered metadata extraction** from natural language queries
- **Advanced filtering** based on candidate metadata:
  - Gender (male/female/not_mentioned)
  - Years of experience
  - Last contacted date
  - Candidate status
  - Placement status
  - Language proficiency levels

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

## Using Metadata Filtering

The system uses DeepSeek R1 Distill Llama 70B via Groq Cloud to extract metadata filters from natural language queries. Simply enable metadata filtering in the sidebar and enter queries like:

- "Find male candidates with 5+ years of experience"
- "Show me candidates with business-level English proficiency"
- "I need female candidates who were contacted in the last 2 years"
- "Show me unplaced candidates with native Japanese skills"

The system will automatically extract relevant filters and display them in the UI.

## Filter Implementation Details

The metadata filter extraction system follows these rules:

1. **Gender**:
   - If user wants "male," also include `gender="not_mentioned"`
   - If user wants "female," also include `gender="not_mentioned"`

2. **Years of Experience**:
   - If user specifies X+ years, filter for `years_of_experience >= X`
   - Also include profiles with `years_of_experience = 0` (unknown)

3. **Last Contacted**:
   - If user says "contacted in last N years," convert to timestamp cutoff
   - Filter for `last_contacted >= cutoff_timestamp`

4. **Candidate Status**:
   - Only filter if explicitly mentioned in query

5. **Placement Status**:
   - Only filter if explicitly mentioned in query

6. **Language Proficiency**:
   - Maps natural language terms to appropriate proficiency levels
   - For each proficiency level, includes all higher levels in the filter:
     - "Native or Bilingual proficiency" (only exact match)
     - "Professional working proficiency" (includes Professional, Full, and Native levels)
     - "Limited working proficiency" (includes Limited, Professional, Full, and Native levels)
     - "Elementary Proficiency" (includes all proficiency levels)
   - Include both profiles with the specified proficiency AND profiles that don't mention that language

## Language Proficiency Mapping

The system maps language proficiency descriptions to standardized levels:

| User Input | Mapped to |
|------------|-----------|
| "native", "mother tongue", "native speaker" | "Native or Bilingual proficiency" |
| "business", "professional", "fluent" | "Professional working proficiency" and above |
| "conversational", "intermediate" | "Limited working proficiency" and above |
| "basic", "elementary", "beginner" | "Elementary Proficiency" and above |

When a user specifies a language without mentioning proficiency level, the system defaults to "Professional working proficiency" and above.

## Using Natural Language Queries

Examples of language-specific queries the system can handle:

- "Find candidates who speak fluent Japanese and conversational Spanish"
- "I need someone with native English and professional German"
- "Show me candidates who speak basic French"
- "I'm looking for candidates who can speak Mandarin"
- "Find candidates with English proficiency and 5+ years of experience"

The system will automatically extract language requirements and apply the appropriate filters based on specified proficiency levels.

## Testing the Filter Extractor

Run the test script to see how the filter extractor works with various queries:
```
python test_filter_extractor.py
```

## Architecture

The system combines:
- **Pinecone** for vector search
- **LangChain** for orchestration
- **Groq** for LLM-powered metadata extraction
- **Streamlit** for the user interface
- **LangSmith** for tracing and monitoring 