# Voice AI Trainer (POC)

A proof-of-concept voice-based AI assistant that answers employee questions based on a custom knowledge base. This project demonstrates the integration of speech-to-text, natural language processing, and text-to-speech technologies to create an interactive voice assistant.

## Goal

Build a proof-of-concept of a voice-based AI assistant that answers employee questions based on a custom knowledge base. The system processes voice input, understands the query, retrieves relevant information, and responds with natural-sounding speech.

## Features

- **Speech-to-Text Processing**: Transcribe user audio input using OpenAI's Whisper model
- **Natural Language Understanding**: Process queries using OpenAI's GPT-4
- **Knowledge Base Retrieval**: Use LangChain and FAISS for efficient semantic search
- **Voice Response**: Convert text responses to natural speech using ElevenLabs/pyttsx3

## Tech Stack

- **Backend Framework**: FastAPI (Python)
- **Speech Processing**:
  - OpenAI Whisper for Speech-to-Text
  - ElevenLabs/pyttsx3 for Text-to-Speech
- **Natural Language Processing**:
  - OpenAI GPT-4 for query understanding and response generation
  - LangChain for building the AI pipeline
  - FAISS for efficient vector similarity search
- **Knowledge Base**: Custom text-based knowledge base with semantic search capabilities

## Project Structure

```
voice-ai-trainer/
├── main.py              # Application entry point
├── requirements.txt     # Project dependencies
├── knowledgebase/       # Knowledge base files
└── utils/              # Helper functions
    ├── stt.py          # Speech-to-Text utilities
    ├── tts.py          # Text-to-Speech utilities
    ├── embeddings.py   # Embedding generation utilities
    └── llm.py          # LLM integration utilities
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key  # If using ElevenLabs
```

4. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the application is running, you can access:
- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## Usage

1. Upload your knowledge base files (text format) through the `/upload/knowledge` endpoint
2. Record and upload audio questions through the `/upload/audio` endpoint
3. The system will:
   - Transcribe the audio to text
   - Process the query using GPT-4
   - Retrieve relevant information from the knowledge base
   - Generate a response
   - Convert the response to speech

## Development Status

This is a proof-of-concept project. Future improvements may include:
- Enhanced error handling and retry mechanisms
- Support for multiple languages
- Improved voice quality and customization
- Real-time audio processing
- Web interface for easier interaction 