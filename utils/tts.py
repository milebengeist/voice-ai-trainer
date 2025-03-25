import edge_tts
import asyncio
from pathlib import Path
from typing import Optional

class TextToSpeech:
    def __init__(self, voice: str = "en-US-JennyNeural"):
        """Initialize the Text-to-Speech converter.
        
        Args:
            voice (str): Voice to use for speech synthesis
        """
        self.voice = voice
        self.available_voices = asyncio.run(self._get_available_voices())

    async def _get_available_voices(self) -> list:
        """Get list of available voices."""
        voices = await edge_tts.list_voices()
        return [voice["ShortName"] for voice in voices]

    async def synthesize(self, text: str, output_path: str, voice: Optional[str] = None) -> bool:
        """Synthesize text to speech and save to file.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the audio file
            voice (str, optional): Voice to use (overrides default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            voice = voice or self.voice
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"Error in text-to-speech synthesis: {e}")
            return False

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        return self.available_voices 