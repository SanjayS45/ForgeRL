"""
FastAPI backend for RLHF pipeline control and monitoring.

Provides REST API endpoints for:
- Dataset upload and management
- Reward model training
- Policy training
- Training progress monitoring
- Model export

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import os
import sys
import json
import asyncio
import pickle
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="RLHF Pipeline API",
    description="API for training and monitoring RLHF models for MetaWorld robotics",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking training jobs
training_jobs: Dict[str, Dict] = {}
job_lock = threading.Lock()


# Pydantic models for request/response
class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    hidden_dim: int = 256
    loss_type: str = "cross_entropy"
    early_stopping_patience: int = 10
    

class PolicyTrainingConfig(BaseModel):
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
    task: str = "reach-v3"
    instruction: Optional[str] = "reach the target"
    reward_model_path: Optional[str] = None
    use_env_reward: bool = True
    custom_goals: Optional[List[List[float]]] = None  # User-defined goal positions for training


class DatasetInfo(BaseModel):
    name: str
    path: str
    num_pairs: int
    num_instructions: int
    obs_dim: int
    act_dim: int
    created_at: str
    

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    current_epoch: int
    total_epochs: int
    train_loss: Optional[float]
    train_accuracy: Optional[float]
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


# Utility functions
def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_datasets_dir() -> Path:
    return get_project_root() / "experiments"


def get_checkpoints_dir() -> Path:
    return get_project_root() / "checkpoints"


def generate_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "RLHF Pipeline API"}


@app.get("/api/info")
async def get_info():
    """Get system information."""
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        pytorch_version = "Not available"
        cuda_available = False
        cuda_device = None
    
    return {
        "pytorch_version": pytorch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "project_root": str(get_project_root()),
    }


# Dataset endpoints

@app.get("/api/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List available datasets."""
    datasets = []
    datasets_dir = get_datasets_dir()
    
    for path in datasets_dir.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                
            if data and len(data) > 0:
                sample = data[0]
                obs_dim = len(sample[1][0][0]) if sample[1] else 0
                act_dim = len(sample[1][0][1]) if sample[1] else 0
                instructions = set(d[0] for d in data)
                
                datasets.append(DatasetInfo(
                    name=path.stem,
                    path=str(path),
                    num_pairs=len(data),
                    num_instructions=len(instructions),
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    created_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                ))
        except Exception as e:
            continue
            
    return datasets


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset file."""
    if not file.filename.endswith((".pkl", ".pickle", ".json")):
        raise HTTPException(400, "Unsupported file format. Use .pkl, .pickle, or .json")
        
    datasets_dir = get_datasets_dir()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = datasets_dir / file.filename
    
    # Read and validate
    content = await file.read()
    
    try:
        if file.filename.endswith(".json"):
            data = json.loads(content)
        else:
            data = pickle.loads(content)
            
        # Basic validation
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Dataset must be a non-empty list")
            
        if len(data[0]) != 3:
            raise ValueError("Each item must be (instruction, traj_a, traj_b)")
            
    except Exception as e:
        raise HTTPException(400, f"Invalid dataset format: {str(e)}")
        
    # Save
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
    return {
        "message": "Dataset uploaded successfully",
        "path": str(save_path),
        "num_pairs": len(data),
    }


@app.delete("/api/datasets/{name}")
async def delete_dataset(name: str):
    """Delete a dataset."""
    path = get_datasets_dir() / f"{name}.pkl"
    
    if not path.exists():
        raise HTTPException(404, "Dataset not found")
        
    path.unlink()
    return {"message": f"Dataset {name} deleted"}


@app.get("/api/datasets/{name}/samples")
async def get_dataset_samples(name: str, n: int = 5):
    """Get sample trajectory pairs from a dataset."""
    path = get_datasets_dir() / f"{name}.pkl"
    
    if not path.exists():
        raise HTTPException(404, "Dataset not found")
        
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        # Sample n random pairs
        import random
        samples = random.sample(data, min(n, len(data)))
        
        # Format for frontend
        formatted_samples = []
        for instr, traj_a, traj_b in samples:
            # Convert numpy arrays to lists for JSON serialization
            traj_a_list = [
                [obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                 act.tolist() if hasattr(act, 'tolist') else list(act)]
                for obs, act in traj_a
            ]
            traj_b_list = [
                [obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                 act.tolist() if hasattr(act, 'tolist') else list(act)]
                for obs, act in traj_b
            ]
            
            # Simple heuristic for predicted preference
            score_a = -np.linalg.norm(
                np.array(traj_a[-1][0][:3]) - np.array(traj_a[-1][0][-3:])
            ) if traj_a else 0
            score_b = -np.linalg.norm(
                np.array(traj_b[-1][0][:3]) - np.array(traj_b[-1][0][-3:])
            ) if traj_b else 0
            
            formatted_samples.append({
                "instruction": instr,
                "traj_a": traj_a_list,
                "traj_b": traj_b_list,
                "predicted_preference": "A" if score_a > score_b else "B"
            })
            
        return {"samples": formatted_samples}
        
    except Exception as e:
        raise HTTPException(500, f"Error loading samples: {str(e)}")


# Reward model training endpoints

@app.post("/api/reward-model/train")
async def start_reward_model_training(
    dataset_name: str,
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
):
    """Start reward model training."""
    dataset_path = get_datasets_dir() / f"{dataset_name}.pkl"
    
    if not dataset_path.exists():
        raise HTTPException(404, f"Dataset not found: {dataset_name}")
        
    job_id = generate_job_id()
    
    with job_lock:
        training_jobs[job_id] = {
            "type": "reward_model",
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.epochs,
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": None,
            "val_accuracy": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "config": config.dict(),
            "dataset": dataset_name,
        }
        
    # Start training in background
    background_tasks.add_task(
        run_reward_model_training,
        job_id,
        str(dataset_path),
        config,
    )
    
    return {"job_id": job_id, "message": "Training started"}


async def run_reward_model_training(job_id: str, dataset_path: str, config: TrainingConfig):
    """Background task for reward model training."""
    import torch
    
    try:
        with job_lock:
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["started_at"] = datetime.now().isoformat()
            
        # Import training modules
        from utils.data_utils import load_dataset, split_dataset
        from utils.instruction_encoder import encode_instruction
        from reward_model.model import RewardModelWithSentenceEncoder
        from reward_model.trainer import RewardModelTrainer
        
        # Load data
        data = load_dataset(dataset_path)
        train_data, val_data, _ = split_dataset(data, 0.8, 0.1, 0.1)
        
        # Get dimensions
        sample = data[0]
        obs_dim = len(sample[1][0][0])
        act_dim = len(sample[1][0][1])
        
        # Create model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = RewardModelWithSentenceEncoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        
        # Create trainer with callback for progress updates
        trainer = RewardModelTrainer(
            model=model,
            device=device,
            learning_rate=config.learning_rate,
            loss_type=config.loss_type,
        )
        
        checkpoint_dir = get_checkpoints_dir() / "reward_model" / job_id
        
        # Training with progress updates
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            instruction_encoder=encode_instruction,
            epochs=config.epochs,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            checkpoint_dir=str(checkpoint_dir),
        )
        
        # Update final status
        with job_lock:
            training_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "current_epoch": len(history["train_loss"]),
                "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "train_accuracy": history["train_accuracy"][-1] if history["train_accuracy"] else None,
                "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
                "completed_at": datetime.now().isoformat(),
                "checkpoint_path": str(checkpoint_dir / "best_model.pt"),
            })
            
    except Exception as e:
        with job_lock:
            training_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })


@app.get("/api/training/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status."""
    with job_lock:
        if job_id not in training_jobs:
            raise HTTPException(404, "Job not found")
        
        job = training_jobs[job_id]
        
    # Handle both reward model and policy training jobs
    # Policy jobs use episode_reward/policy_loss, reward model uses train_loss/train_accuracy
    train_loss = job.get("train_loss") or job.get("policy_loss")
    train_accuracy = job.get("train_accuracy")
    episode_reward = job.get("episode_reward")
    
    return TrainingStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        current_epoch=job.get("current_epoch", 0),
        total_epochs=job.get("total_epochs", 0),
        train_loss=train_loss,
        train_accuracy=train_accuracy if train_accuracy else (episode_reward / 100 if episode_reward else None),
        val_loss=job.get("val_loss"),
        val_accuracy=job.get("val_accuracy"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )


@app.get("/api/training")
async def list_training_jobs():
    """List all training jobs."""
    with job_lock:
        return [
            {"job_id": job_id, **job}
            for job_id, job in training_jobs.items()
        ]


# Policy training endpoints

@app.post("/api/policy/train")
async def start_policy_training(
    config: PolicyTrainingConfig,
    background_tasks: BackgroundTasks,
):
    """Start policy training."""
    job_id = generate_job_id()
    
    with job_lock:
        training_jobs[job_id] = {
            "type": "policy",
            "status": "pending",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.total_steps // config.steps_per_update,
            "episode_reward": None,
            "policy_loss": None,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "config": config.dict(),
        }
        
    background_tasks.add_task(run_policy_training, job_id, config)
    
    return {"job_id": job_id, "message": "Policy training started"}


async def run_policy_training(job_id: str, config: PolicyTrainingConfig):
    """Background task for policy training."""
    import torch
    import traceback
    
    try:
        with job_lock:
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["started_at"] = datetime.now().isoformat()
            
        from envs.metaworld_wrapper import make_metaworld_env
        from policy.ppo import PPOAgent, PPOTrainer
        from reward_model.model import RewardModelWithSentenceEncoder
        from utils.instruction_encoder import encode_instruction
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Policy Training] Starting job {job_id}")
        print(f"[Policy Training] Task: {config.task}")
        print(f"[Policy Training] Instruction: {config.instruction}")
        print(f"[Policy Training] Device: {device}")
        if config.custom_goals:
            print(f"[Policy Training] Custom goals provided: {len(config.custom_goals)} targets")
        
        # Create environment with the specified task
        env = make_metaworld_env(task_name=config.task)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        print(f"[Policy Training] Environment created - obs_dim: {obs_dim}, act_dim: {act_dim}")
        
        # Load reward model if specified
        reward_model = None
        if config.reward_model_path and Path(config.reward_model_path).exists():
            print(f"[Policy Training] Loading reward model from {config.reward_model_path}")
            reward_model = RewardModelWithSentenceEncoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
            ).to(device)
            checkpoint = torch.load(config.reward_model_path, map_location=device, weights_only=False)
            reward_model.load_state_dict(checkpoint["model_state_dict"])
            reward_model.eval()
            print("[Policy Training] Reward model loaded successfully")
        else:
            print("[Policy Training] Using environment reward (no reward model)")
            
        # Create agent with all parameters
        agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            device=device,
            lr=config.learning_rate,
            clip_ratio=config.clip_ratio,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
        )
        
        log_dir = get_project_root() / "logs" / "policy" / job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training config
        config_path = log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        trainer = PPOTrainer(
            env=env,
            agent=agent,
            reward_model=reward_model,
            instruction=config.instruction,
            instruction_encoder=encode_instruction,
            use_env_reward=config.use_env_reward,
            log_dir=str(log_dir),
            custom_goals=config.custom_goals,  # Pass custom goals for potential goal-conditioned training
        )
        
        # Training with progress callback
        def progress_callback(step, total, episode_reward, policy_loss):
            progress = (step / total) * 100
            with job_lock:
                training_jobs[job_id].update({
                    "progress": progress,
                    "cumulative_reward": episode_reward,
                    "policy_loss": policy_loss,
                })
        
        history = trainer.train(
            total_timesteps=config.total_steps,
            steps_per_update=config.steps_per_update,
            epochs_per_update=config.epochs_per_update,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            progress_callback=progress_callback,
        )
        
        with job_lock:
            training_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "cumulative_reward": history["episode_reward"][-1] if history.get("episode_reward") else None,
                "policy_loss": history["policy_loss"][-1] if history.get("policy_loss") else None,
                "completed_at": datetime.now().isoformat(),
                "checkpoint_path": str(log_dir / "final_model.pt"),
            })
        
        print(f"[Policy Training] Job {job_id} completed successfully")
        env.close()
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Policy Training] Job {job_id} failed: {error_msg}")
        with job_lock:
            training_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            })


# Model export endpoints

@app.get("/api/models")
async def list_models():
    """List available trained models."""
    models = []
    checkpoints_dir = get_checkpoints_dir()
    
    # Reward models
    reward_dir = checkpoints_dir / "reward_model"
    if reward_dir.exists():
        for path in reward_dir.glob("*/best_model.pt"):
            models.append({
                "type": "reward_model",
                "name": path.parent.name,
                "path": str(path),
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })
            
    # Policy models
    logs_dir = get_project_root() / "logs" / "policy"
    if logs_dir.exists():
        for path in logs_dir.glob("*/final_model.pt"):
            models.append({
                "type": "policy",
                "name": path.parent.name,
                "path": str(path),
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })
            
    return models


@app.get("/api/models/{model_type}/{name}/download")
async def download_model(model_type: str, name: str):
    """Download a trained model."""
    if model_type == "reward_model":
        path = get_checkpoints_dir() / "reward_model" / name / "best_model.pt"
    elif model_type == "policy":
        path = get_project_root() / "logs" / "policy" / name / "final_model.pt"
    else:
        raise HTTPException(400, "Invalid model type")
        
    if not path.exists():
        raise HTTPException(404, "Model not found")
        
    return FileResponse(path, filename=f"{model_type}_{name}.pt")


# History and visualization endpoints

@app.get("/api/training/{job_id}/history")
async def get_training_history(job_id: str):
    """Get training history for visualization."""
    # Check reward model history
    history_path = get_checkpoints_dir() / "reward_model" / job_id / "history.json"
    if not history_path.exists():
        # Check policy history
        history_path = get_project_root() / "logs" / "policy" / job_id / "history.json"
        
    if not history_path.exists():
        raise HTTPException(404, "History not found")
        
    with open(history_path) as f:
        history = json.load(f)
        
    return history


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

