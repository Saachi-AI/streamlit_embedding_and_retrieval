import os
import streamlit as st
import cohere
from langsmith.run_helpers import traceable

class CohereReranker:
    """
    A class to rerank documents using Cohere's rerank API.
    """
    
    def __init__(self, api_key):
        """
        Initialize the CohereReranker with the API key.
        
        Args:
            api_key (str): Cohere API key
        """
        self.client = cohere.Client(api_key=api_key)
        # Get default model from environment variable or use fallback
        self.default_model = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
    
    @traceable(name="rerank_documents")
    def rerank(self, query, documents, top_k, model=None):
        """
        Rerank documents using Cohere's rerank API.
        
        Args:
            query (str): Search query text
            documents (list): List of (document, score) tuples from retrieve_documents
            top_k (int): Number of results to return after reranking
            model (str): Cohere rerank model to use. If None, uses the model from environment variables
            
        Returns:
            reranked_results: List of (document, rerank_score) tuples sorted by rerank score
        """
        if not documents:
            return []
        
        # If no model is provided, use the default from environment
        if model is None:
            model = self.default_model
        
        # Extract document texts for reranking
        document_texts = [doc.page_content for doc, _ in documents]
        
        # Call Cohere's rerank API
        try:
            rerank_results = self.client.rerank(
                model=model,
                query=query,
                documents=document_texts,
                top_n=min(top_k, len(documents))
            )
            
            # Map reranked results back to original documents
            reranked_documents = []
            for result in rerank_results.results:
                index = result.index
                doc, _ = documents[index]  # Get original document
                reranked_documents.append((doc, result.relevance_score))
            
            return reranked_documents
        except Exception as e:
            error_msg = str(e)
            if "model" in error_msg and "not found" in error_msg:
                # Provide more helpful message for model not found errors
                suggestions = [
                    "rerank-english-v3.0",  # Current model as of 2023-2024
                    "rerank-multilingual-v3.0",  # Multilingual version
                ]
                
                st.error(
                    f"Error: The rerank model '{model}' was not found or you don't have access to it.\n\n"
                    f"Original error: {error_msg}\n\n"
                    f"Suggested model names to try:\n" +
                    "\n".join([f"- `{m}`" for m in suggestions])
                )
                
                # Add a message about checking Cohere documentation
                st.info(
                    "You may need to check the latest Cohere documentation for the current model names, "
                    "or verify that your API key has access to the reranking models."
                )
            else:
                # Generic error message for other errors
                st.error(f"Error during reranking: {error_msg}")
            
            # Fall back to original results if reranking fails
            return documents[:top_k] 