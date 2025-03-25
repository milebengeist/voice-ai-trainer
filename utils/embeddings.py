from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            
        Returns:
            np.ndarray: Generated embeddings
        """
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(text)
        return embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def find_similar_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[tuple]:
        """Find most similar texts to a query.
        
        Args:
            query (str): Query text
            texts (List[str]): List of texts to search through
            top_k (int): Number of similar texts to return
            
        Returns:
            List[tuple]: List of (text, similarity_score) tuples
        """
        query_embedding = self.generate_embedding(query)
        text_embeddings = self.generate_embedding(texts)
        
        similarities = [
            self.compute_similarity(query_embedding, text_embedding)
            for text_embedding in text_embeddings
        ]
        
        # Sort by similarity score in descending order
        results = list(zip(texts, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k] 