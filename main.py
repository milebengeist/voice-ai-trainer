from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
import shutil
import os

from utils.transcribe import AudioTranscriber
from utils.speak import TextToSpeech
from utils.knowledge import KnowledgeBase

# Create FastAPI app
app = FastAPI(
    title="Voice AI Trainer",
    description="API for voice-based AI assistant that answers employee questions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path("static").mkdir(exist_ok=True)
Path("knowledgebase").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)

# Initialize components
transcriber = AudioTranscriber()
tts = TextToSpeech()
knowledge_base = KnowledgeBase()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint returning a simple HTML page."""
    return """
    <html>
        <head>
            <title>Voice AI Trainer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                h1 { color: #333; }
                .status { color: green; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to Voice AI Trainer</h1>
                <p>Status: <span class="status">Active</span></p>
                <p>Version: 1.0.0</p>
                <p>API Documentation: <a href="/docs">/docs</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and transcribe an audio file."""
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Only audio files are allowed")
    
    # Save uploaded file temporarily
    temp_file = Path("temp") / file.filename
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribe audio to text
        text = transcriber.transcribe(str(temp_file))
        return {"text": text}
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

@app.post("/upload/knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
    """Upload a knowledge base file."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only text files are allowed")
    
    # Save file to knowledgebase directory
    file_path = Path("knowledgebase") / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Add to knowledge base
    knowledge_base.load_from_file(str(file_path))
    
    return {"message": f"Successfully uploaded and processed: {file.filename}"}

@app.post("/ask")
async def ask_question(audio_file: UploadFile = File(...)):
    """Process an audio question and return a voice response."""
    if not audio_file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Only audio files are allowed")
    
    # Save uploaded file temporarily
    temp_file = Path("temp") / audio_file.filename
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe audio to text
        question = transcriber.transcribe(str(temp_file))
        
        # Search knowledge base
        relevant_info = knowledge_base.search(question)
        
        # Generate response
        response = f"Based on the available information: {' '.join(relevant_info)}"
        
        # Convert response to speech
        audio_path = Path("temp") / "response.mp3"
        tts.speak(response, str(audio_path))
        
        return {
            "question": question,
            "response": response,
            "audio_url": "/audio/response.mp3"
        }
    finally:
        # Clean up question audio
        if temp_file.exists():
            temp_file.unlink()

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Get an audio file."""
    file_path = Path("temp") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # This is the correct way to run it on Replit
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Required for Replit
        port=8080,       # Standard port for Replit
        reload=True
    ) 