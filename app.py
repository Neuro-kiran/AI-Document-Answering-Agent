import os
import openai
import requests
import json
import logging
import torch
from PyPDF2 import PdfReader
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configure logger to display info and error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Slack app with the bot token
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Set the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the pre-trained model and tokenizer for text encoding
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index for storing and retrieving text embeddings
dimension = 384  # The dimension of the sentence-transformer model output
index = faiss.IndexFlatL2(dimension)

# Global variable to hold the extracted PDF text
pdf_text = ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            # Extract text from each page in the PDF
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text.strip()

def encode_text(text: str):
    """Encode text using the transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        # Get embeddings for the text
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

def add_text_to_faiss(text: str):
    """Add encoded text to the FAISS index."""
    embeddings = encode_text(text)
    index.add(embeddings)

def search_faiss(query: str):
    """Search the FAISS index for the most relevant text."""
    query_embedding = encode_text(query)
    _, indices = index.search(query_embedding, k=1)  # Retrieve the top 1 result
    return indices

def answer_question(query: str) -> str:
    """Use OpenAI to answer questions based on the PDF content."""
    # Search FAISS index for relevant text
    indices = search_faiss(query)
    if indices.size > 0:
        # Use the closest text snippet for the query
        start_index = indices[0][0]
        snippet = pdf_text[start_index:start_index + 500]  # Adjust snippet length as needed
    else:
        snippet = pdf_text[:500]  # Fallback snippet if no relevant text found
    
    prompt = f"Here is the document snippet:\n\n{snippet}\n\nNow answer the following question word-to-word from the document or say 'Data Not Available' if it's not found: {query}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use the specified model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        # Extract the answer from the response
        answer = response.choices[0].message['content'].strip()
        if "Data Not Available" in answer:
            return "Data Not Available"
        return answer
    except openai.error.OpenAIAPIError as e:
        logger.error(f"OpenAI API error: {e}")
        return "Error retrieving answer."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "Error retrieving answer."

@app.event("file_shared")
def handle_file_shared(event, say):
    """Handle file shared events in Slack."""
    global pdf_text
    try:
        file_id = event['file']['id']
        # Fetch file info using the file ID
        file_info = app.client.files_info(file=file_id)
        pdf_url = file_info['file']['url_private_download']
        
        # Download the PDF file from Slack
        headers = {
            'Authorization': f'Bearer {os.getenv("SLACK_BOT_TOKEN")}'
        }
        response = requests.get(pdf_url, headers=headers)
        pdf_path = "uploaded_document.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Add the text to the FAISS index
        add_text_to_faiss(pdf_text)
        
        say("PDF uploaded and processed successfully! You can now ask questions about the content.")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        say(f"Error processing PDF: {str(e)}")

@app.message(".*")
def handle_questions(message, say):
    """Handle user questions and provide answers based on the PDF content."""
    global pdf_text
    if not pdf_text:
        say("No PDF has been uploaded yet. Please upload a PDF file first.")
    else:
        user_question = message['text']
        answer = answer_question(user_question)
        say(f"Answer: {answer}")

# Start the Slack app with SocketModeHandler
if __name__ == "__main__":
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()
