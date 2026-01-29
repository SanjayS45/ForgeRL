from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DatasetInfo(BaseModel):
    name: str
    num_pairs: int
    num_instructions: int
    obs_dim: int
    act_dim: int
    file_size_mb: float
    created_at: Optional[datetime] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    path: str
    created_at: Optional[datetime] = None
    metrics: Optional[dict] = None

class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0001
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    validation_split: float = 0.1
    early_stopping_patience: int = 10

class PolicyTrainingConfig(BaseModel):
    task: str = "reach-v3"
    instruction: str = "reach the target"
    total_steps: int = 100000
    steps_per_update: int = 2048
    epochs_per_update: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    hidden_dim: int = 256
    num_layers: int = 2
    custom_goals: Optional[List[List[float]]] = None

class TrainingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    cumulative_reward: Optional[float] = None
    policy_loss: Optional[float] = None
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    job_type: str
    task_name: Optional[str] = None

class TrajectoryPoint(BaseModel):
    observation: List[float]
    action: List[float]

class SampleTrajectory(BaseModel):
    instruction: str
    traj_a: List[List[List[float]]]
    traj_b: List[List[List[float]]]
    predicted_preference: str

