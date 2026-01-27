"""
Logging utilities for training pipelines.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
    file_handler.setLevel(level)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    if console:
        console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    if console:
        logger.addHandler(console_handler)
    
    return logger


class TrainingLogger:
    """
    Logger for tracking training metrics.
    
    Supports:
    - Metric logging with automatic averaging
    - JSON export of training history
    - Integration with TensorBoard
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        use_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Create directories
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.history: Dict[str, List[float]] = {}
        self.current_epoch_metrics: Dict[str, List[float]] = {}
        self.current_epoch = 0
        self.global_step = 0
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.experiment_dir / "tensorboard"))
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                
        # Text logger
        self.logger = setup_logger(
            experiment_name,
            str(self.experiment_dir),
        )
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (uses global_step if not provided)
        """
        if step is None:
            step = self.global_step
            
        # Store in current epoch
        if name not in self.current_epoch_metrics:
            self.current_epoch_metrics[name] = []
        self.current_epoch_metrics[name].append(value)
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar(name, value, step)
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def end_epoch(self):
        """
        End current epoch and compute averages.
        """
        epoch_summary = {}
        
        for name, values in self.current_epoch_metrics.items():
            avg_value = np.mean(values)
            
            # Store in history
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(avg_value)
            
            epoch_summary[name] = avg_value
            
        # Log summary
        summary_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_summary.items()])
        self.logger.info(f"Epoch {self.current_epoch}: {summary_str}")
        
        # Reset for next epoch
        self.current_epoch_metrics = {}
        self.current_epoch += 1
        
        return epoch_summary
        
    def step(self):
        """Increment global step."""
        self.global_step += 1
        
    def save_history(self, filename: str = "history.json"):
        """Save training history to JSON."""
        path = self.experiment_dir / filename
        
        # Convert numpy types to Python types
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]
            
        with open(path, "w") as f:
            json.dump(serializable_history, f, indent=2)
            
        self.logger.info(f"Saved history to {path}")
        
    def load_history(self, filename: str = "history.json"):
        """Load training history from JSON."""
        path = self.experiment_dir / filename
        
        if path.exists():
            with open(path, "r") as f:
                self.history = json.load(f)
            self.current_epoch = max(len(v) for v in self.history.values()) if self.history else 0
            
    def get_best_metric(self, name: str, mode: str = "min") -> tuple:
        """
        Get best value of a metric.
        
        Args:
            name: Metric name
            mode: 'min' or 'max'
            
        Returns:
            (best_value, best_epoch)
        """
        if name not in self.history or not self.history[name]:
            return None, None
            
        values = self.history[name]
        if mode == "min":
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
            
        return values[best_idx], best_idx
        
    def close(self):
        """Close the logger and save final state."""
        self.save_history()
        if self.writer:
            self.writer.close()


class CheckpointManager:
    """
    Manages model checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Path] = []
        
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Dict[str, float],
        name: str = "checkpoint",
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Current metrics
            name: Checkpoint name prefix
            
        Returns:
            Path to saved checkpoint
        """
        import torch
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_epoch{epoch}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        torch.save(checkpoint, path)
        self.checkpoints.append(path)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                
        return path
        
    def load_latest(self, model, optimizer=None) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        import torch
        
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
            
        latest = checkpoints[-1]
        return self.load(latest, model, optimizer)
        
    def load(
        self,
        path: Path,
        model,
        optimizer=None,
    ) -> Dict[str, Any]:
        """Load a specific checkpoint."""
        import torch
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
        }

