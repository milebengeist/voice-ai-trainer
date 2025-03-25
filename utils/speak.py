import pyttsx3
from pathlib import Path
from typing import Optional

class TextToSpeech:
    def __init__(self, voice: Optional[str] = None, rate: int = 150):
        """Initialize the text-to-speech engine.
        
        Args:
            voice (str, optional): Voice to use for speech synthesis
            rate (int): Speech rate (words per minute)
        """
        self.engine = pyttsx3.init()
        
        # Set voice if provided
        if voice:
            self.set_voice(voice)
        
        # Set default properties
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', 0.9)

    def set_voice(self, voice: str) -> bool:
        """Set the voice for speech synthesis.
        
        Args:
            voice (str): Voice identifier or language code
            
        Returns:
            bool: True if voice was set successfully
        """
        voices = self.engine.getProperty('voices')
        for v in voices:
            if voice.lower() in v.name.lower():
                self.engine.setProperty('voice', v.id)
                return True
        return False

    def speak(self, text: str, save_path: Optional[str] = None) -> Optional[str]:
        """Convert text to speech and optionally save to file.
        
        Args:
            text (str): Text to convert to speech
            save_path (str, optional): Path to save the audio file
            
        Returns:
            str: Path to saved audio file if save_path is provided
        """
        if save_path:
            # Ensure the directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            self.engine.save_to_file(text, save_path)
            self.engine.runAndWait()
            return save_path
        else:
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            return None

    def get_available_voices(self) -> list:
        """Get list of available voices.
        
        Returns:
            list: List of available voice names
        """
        voices = self.engine.getProperty('voices')
        return [voice.name for voice in voices] 