from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional

class KnowledgeBase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the knowledge base.
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.index = None
        
    def add_text(self, text: str):
        """Add text to the knowledge base.
        
        Args:
            text (str): Text to add
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        self.texts.extend(paragraphs)
        
        # Create embeddings
        embeddings = self.model.encode(paragraphs)
        
        if self.index is None:
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search the knowledge base.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[str]: List of relevant text passages
        """
        if not self.texts:
            return []
            
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        # Return relevant texts
        return [self.texts[idx] for idx in indices[0]]
    
    def load_from_file(self, file_path: str):
        """Load text from a file into the knowledge base.
        
        Args:
            file_path (str): Path to the text file
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            self.add_text(text)
    
    def load_from_directory(self, directory: str):
        """Load all text files from a directory.
        
        Args:
            directory (str): Path to the directory
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        for file_path in dir_path.glob('*.txt'):
            self.load_from_file(str(file_path)) 