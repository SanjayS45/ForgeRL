import math
from typing import Optional

class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr: float):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, epoch: int):
        raise NotImplementedError
    
    def set_lr(self, lr: float):
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class StepLR(LearningRateScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        step_size: int = 30,
        gamma: float = 0.1,
    ):
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch: int):
        new_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        self.set_lr(new_lr)

class ExponentialLR(LearningRateScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        gamma: float = 0.99,
    ):
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma
    
    def step(self, epoch: int):
        new_lr = self.initial_lr * (self.gamma ** epoch)
        self.set_lr(new_lr)

class CosineAnnealingLR(LearningRateScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0,
    ):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def step(self, epoch: int):
        new_lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max)
        ) / 2
        self.set_lr(new_lr)

class WarmupLR(LearningRateScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        warmup_epochs: int = 5,
        base_scheduler: Optional[LearningRateScheduler] = None,
    ):
        super().__init__(optimizer, initial_lr)
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            new_lr = self.initial_lr * warmup_factor
            self.set_lr(new_lr)
        elif self.base_scheduler is not None:
            self.base_scheduler.step(epoch - self.warmup_epochs)

class LinearDecayLR(LearningRateScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        total_epochs: int,
        end_lr: float = 0,
    ):
        super().__init__(optimizer, initial_lr)
        self.total_epochs = total_epochs
        self.end_lr = end_lr
    
    def step(self, epoch: int):
        progress = min(epoch / self.total_epochs, 1.0)
        new_lr = self.initial_lr + (self.end_lr - self.initial_lr) * progress
        self.set_lr(new_lr)

