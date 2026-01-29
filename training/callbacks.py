from typing import Callable, Dict, Any, Optional
import time
from pathlib import Path

class TrainingCallback:
    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        pass
    
    def on_batch_start(self, batch: int, logs: Dict[str, Any] = None):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        pass
    
    def on_train_start(self, logs: Dict[str, Any] = None):
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        pass

class EarlyStopping(TrainingCallback):
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == "min":
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class ModelCheckpoint(TrainingCallback):
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        save_fn: Optional[Callable] = None,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_fn = save_fn
        self.best = float("inf") if mode == "min" else float("-inf")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if logs is None or self.save_fn is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == "min":
            improved = current < self.best
        else:
            improved = current > self.best
        
        if improved or not self.save_best_only:
            self.best = current
            self.save_fn(str(self.filepath))

class ProgressLogger(TrainingCallback):
    def __init__(self, log_fn: Optional[Callable] = None):
        self.log_fn = log_fn or print
        self.epoch_start_time = None
    
    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        msg = f"Epoch {epoch} - {elapsed:.1f}s"
        if logs:
            for key, value in logs.items():
                if isinstance(value, float):
                    msg += f" - {key}: {value:.4f}"
        
        self.log_fn(msg)

