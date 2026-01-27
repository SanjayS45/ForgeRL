from typing import Union, List
import numpy as np

_model = None

def get_sentence_model():
    
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def encode_instruction(text: Union[str, List[str]]) -> np.ndarray:
    
    model = get_sentence_model()
    return model.encode(text)

def encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    
    model = get_sentence_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 100)

def instruction_similarity(instr1: str, instr2: str) -> float:
    
    enc1 = encode_instruction(instr1)
    enc2 = encode_instruction(instr2)
    
    return np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))
