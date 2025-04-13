from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Simple Personal RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
chat_sessions = {}

# Load documents
try:
    data_path = pathlib.Path(__file__).parent / "sample_text.txt"
    with open(data_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Split into paragraphs
    documents = [p for p in text.split('\n\n') if p.strip()]
    logger.info(f"Loaded {len(documents)} paragraphs")
except Exception as e:
    logger.error(f"Error loading documents: {str(e)}")
    documents = ["Information not available"]

# Create TF-IDF vectorizer and index documents
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Define request model
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Search function
def search_documents(query, top_k=3):
    # Convert query to TF-IDF vector
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity with all documents
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [documents[i] for i in top_indices]
    
    return results

# Get or create chat session
def get_session(session_id=None):
    if session_id and session_id in chat_sessions:
        return session_id, chat_sessions[session_id]
    
    # Create new session
    new_id = session_id or f"session_{len(chat_sessions) + 1}"
    chat_sessions[new_id] = model.start_chat(history=[])
    return new_id, chat_sessions[new_id]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Search for relevant context
        relevant_docs = search_documents(request.message, top_k=3)
        context = "\n\n".join(relevant_docs)
        
        # Get or create chat session
        session_id, chat = get_session(request.session_id)
        
        # Create prompt
        prompt = f"""
You are a helpful AI assistant for Jenas Anton Vimal. Use this information to answer:
{context}

Question: {request.message}

Answer naturally and concisely without mentioning that you're using a database.
"""
        
        # Get response from Gemini
        response = chat.send_message(prompt)
        
        # Return plain text response without JSON formatting
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=response.text.strip())
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")

@app.get("/api")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)