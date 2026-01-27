from typing import Dict, List, Union, Optional
import numpy as np

class InstructionEncoder:
    
    
    def __init__(
        self,
        mode: str = "sentence_transformer",
        vocab: Optional[List[str]] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        
        self.mode = mode
        self.model_name = model_name
        
        if mode == "simple":
            self._init_simple_encoder(vocab)
        elif mode == "sentence_transformer":
            self._init_sentence_transformer()
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def _init_simple_encoder(self, vocab: Optional[List[str]]):
        
        if vocab is None:

            vocab = [
                "reach", "push", "pick", "place", "grab", "move",
                "red", "blue", "green", "yellow", "target", "goal",
                "object", "block", "cube", "marker", "position",
                "the", "to", "at", "toward", "forward",
            ]
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.embedding_dim = len(vocab)
        
    def _init_sentence_transformer(self):
        
        self._model = None
        self.embedding_dim = 384
        
    @property
    def sentence_model(self):
        
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
        
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        
        if self.mode == "simple":
            return self._encode_simple(text)
        else:
            return self._encode_sentence_transformer(text)
            
    def _encode_simple(self, text: Union[str, List[str]]) -> np.ndarray:
        
        if isinstance(text, str):
            words = text.lower().split()
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            for w in words:
                if w in self.word2idx:
                    vec[self.word2idx[w]] = 1.0
            return vec
        else:
            return np.array([self._encode_simple(t) for t in text])
            
    def _encode_sentence_transformer(self, text: Union[str, List[str]]) -> np.ndarray:
        
        return self.sentence_model.encode(text)
        
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        
        return self.encode(text)

_global_encoder: Optional[InstructionEncoder] = None

def get_encoder(mode: str = "sentence_transformer") -> InstructionEncoder:
    
    global _global_encoder
    if _global_encoder is None or _global_encoder.mode != mode:
        _global_encoder = InstructionEncoder(mode=mode)
    return _global_encoder

def encode_instruction(
    text: Union[str, List[str]],
    mode: str = "sentence_transformer"
) -> np.ndarray:
    
    encoder = get_encoder(mode)
    return encoder.encode(text)

_instruction_to_id: Dict[str, int] = {}
_next_id = 0

def instruction_to_id(instruction: str) -> int:
    
    global _next_id
    
    instruction = instruction.lower().strip()
    
    if instruction not in _instruction_to_id:
        _instruction_to_id[instruction] = _next_id
        _next_id += 1
        
    return _instruction_to_id[instruction]

def get_vocab_size() -> int:
    
    return max(_next_id, 1)

