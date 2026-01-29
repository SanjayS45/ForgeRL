from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path

@dataclass
class RewardModelConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    loss_type: str = "cross_entropy"
    optimizer: str = "adamw"
    
    def validate(self):
        assert self.epochs > 0, "epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be between 0 and 1"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert 0 < self.validation_split < 1, "validation_split must be between 0 and 1"
        assert self.loss_type in ["cross_entropy", "hinge", "bce"], "invalid loss_type"
        return True

@dataclass
class PolicyConfig:
    task: str = "reach-v3"
    instruction: str = "reach the target"
    total_steps: int = 100000
    steps_per_update: int = 2048
    epochs_per_update: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    num_layers: int = 2
    custom_goals: Optional[List[List[float]]] = None
    
    def validate(self):
        assert self.total_steps > 0, "total_steps must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.learning_rate < 1, "learning_rate must be between 0 and 1"
        assert 0 < self.gamma <= 1, "gamma must be between 0 and 1"
        assert 0 < self.clip_ratio < 1, "clip_ratio must be between 0 and 1"
        return True
    
    def to_dict(self):
        return {
            "task": self.task,
            "instruction": self.instruction,
            "total_steps": self.total_steps,
            "steps_per_update": self.steps_per_update,
            "epochs_per_update": self.epochs_per_update,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "custom_goals": self.custom_goals,
        }

def save_config(config, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, 'to_dict'):
        data = config.to_dict()
    else:
        data = config.__dict__
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_config(path: str, config_class):
    with open(path, 'r') as f:
        data = json.load(f)
    return config_class(**data)

