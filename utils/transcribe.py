import whisper
from pathlib import Path
from typing import Optional

class AudioTranscriber:
    def __init__(self, model_name: str = "base"):
        """Initialize the audio transcriber.
        
        Args:
            model_name (str): Name of the Whisper model to use
        """
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code (e.g., 'en', 'de')
            
        Returns:
            str: Transcribed text
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe"
        )
        
        return result["text"].strip()

    def transcribe_with_timestamps(self, audio_path: str, language: Optional[str] = None) -> dict:
        """Transcribe audio file to text with timestamps.
        
        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code
            
        Returns:
            dict: Dictionary containing text and timestamps
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            return_timestamps=True
        )
        
        return result 