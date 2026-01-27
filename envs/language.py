"""
Language encoding utilities using sentence transformers.
"""

from typing import Union, List
import numpy as np

# Lazy load sentence transformer to avoid import overhead
_model = None


def get_sentence_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode_instruction(text: Union[str, List[str]]) -> np.ndarray:
    """
    Encode instruction text using sentence transformer.
    
    Args:
        text: Single instruction string or list of instructions
        
    Returns:
        Encoded vector(s) of shape (384,) or (N, 384)
    """
    model = get_sentence_model()
    return model.encode(text)


def encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a batch of instructions efficiently.
    
    Args:
        texts: List of instruction strings
        batch_size: Batch size for encoding
        
    Returns:
        Encoded vectors of shape (N, 384)
    """
    model = get_sentence_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 100)


def instruction_similarity(instr1: str, instr2: str) -> float:
    """
    Compute cosine similarity between two instructions.
    """
    enc1 = encode_instruction(instr1)
    enc2 = encode_instruction(instr2)
    
    return np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))
