import os
import json
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import Optional, Dict, Any, Tuple

class DocumentParser:
    """
    Class for parsing PDF and other document formats using the Upstage AI API.
    """
    
    def __init__(self, api_key: str):
        """Initialize with Upstage API key"""
        self.api_key = api_key
        self.url = "https://api.upstage.ai/v1/document-digitization"
        self.output_folder = "./output"
        os.makedirs(self.output_folder, exist_ok=True)
    
    def parse_document(self, uploaded_file) -> Tuple[bool, str, Optional[str]]:
        """
        Parse a document using Upstage AI API
        
        Args:
            uploaded_file: A Streamlit UploadedFile object
        
        Returns:
            Tuple of (success, message, parsed_text)
        """
        # Save uploaded file temporarily
        temp_filename = os.path.join(self.output_folder, uploaded_file.name)
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Set up API request
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": open(temp_filename, "rb")}
        data = {
            "ocr": "force",
            "base64_encoding": "['table']",
            "model": "document-parse"
        }
        
        # Call the API
        try:
            response = requests.post(self.url, headers=headers, files=files, data=data)
            
            # Close and potentially remove the temp file
            files["document"].close()
            
            if response.status_code == 200:
                # Parse the JSON response
                result = response.json()
                html_content = result.get('content', {}).get('html', '')
                
                if not html_content:
                    return False, "No HTML content found in the document 😥", None
                
                # Convert HTML to plain text
                soup = BeautifulSoup(html_content, "html.parser")
                plain_text = soup.get_text(separator='\n')
                
                # Save the text to a .txt file
                output_file_path = os.path.join(self.output_folder, "job_description.txt")
                with open(output_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(plain_text)
                
                return True, f"Document parsed successfully 💌", plain_text
            else:
                return False, f"API call failed with status code {response.status_code} 😢", None
        except Exception as e:
            return False, f"Error parsing document: {str(e)}", None
        finally:
            # Clean up - remove temp file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass
