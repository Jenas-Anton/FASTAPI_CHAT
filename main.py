from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import pathlib

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
chat_session = model.start_chat(history=[])

# Read personal info from file - adjust path for Vercel
try:
    data_path = pathlib.Path(__file__).parent.parent / "data" / "sample_text.txt"
    with open(data_path, 'r', encoding='utf-8') as file:
        personal_info = file.read()
except FileNotFoundError:
    logger.error("Could not find sample_text.txt file")
    personal_info = "Information not available"

# Define request model
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    message = request.message
    
    prompt = f"""
You are a knowledgeable and helpful AI chatbot designed to answer questions about B. A. Akith Chandinu, an undergraduate from the University of Moratuwa, Faculty of IT. You specialize in providing clear, accurate, and informative answers based on the following details:

**Background**:
{personal_info}

With this information, answer the following question in a friendly, detailed, and accurate manner. Give short answers if possible: "{message}"
"""
    
    try:
        response = chat_session.send_message(prompt)
        logger.info("Response generated successfully.")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request.")

@app.get("/api")
async def root():
    return {"message": "The Backend Server is running!"}