import numpy as np
from typing import Dict, Optional
import torch

class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = 100000,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_observations": torch.tensor(self.next_observations[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }
    
    def __len__(self):
        return self.size

class TrajectoryBuffer:
    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.rewards = []
    
    def add_trajectory(self, trajectory: list, total_reward: float):
        if len(self.trajectories) >= self.max_trajectories:
            self.trajectories.pop(0)
            self.rewards.pop(0)
        
        self.trajectories.append(trajectory)
        self.rewards.append(total_reward)
    
    def sample_trajectories(self, n: int) -> list:
        indices = np.random.choice(len(self.trajectories), size=min(n, len(self.trajectories)), replace=False)
        return [self.trajectories[i] for i in indices]
    
    def get_best_trajectories(self, n: int) -> list:
        sorted_indices = np.argsort(self.rewards)[::-1][:n]
        return [self.trajectories[i] for i in sorted_indices]
    
    def __len__(self):
        return len(self.trajectories)

