import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Callable

class PreferenceDataset(Dataset):
    def __init__(
        self,
        data: List[Tuple],
        instruction_encoder: Callable,
        max_trajectory_length: int = 150,
    ):
        self.data = data
        self.instruction_encoder = instruction_encoder
        self.max_trajectory_length = max_trajectory_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instruction, traj_a, traj_b = self.data[idx]
        
        instr_enc = self.instruction_encoder(instruction)
        
        obs_a, act_a, mask_a = self._process_trajectory(traj_a)
        obs_b, act_b, mask_b = self._process_trajectory(traj_b)
        
        return {
            "instruction": torch.tensor(instr_enc, dtype=torch.float32),
            "obs_a": obs_a,
            "act_a": act_a,
            "mask_a": mask_a,
            "obs_b": obs_b,
            "act_b": act_b,
            "mask_b": mask_b,
            "preference": torch.tensor(1.0),
        }
    
    def _process_trajectory(self, trajectory):
        observations = []
        actions = []
        
        for obs, act in trajectory[:self.max_trajectory_length]:
            observations.append(obs)
            actions.append(act)
        
        length = len(observations)
        
        if length < self.max_trajectory_length:
            pad_length = self.max_trajectory_length - length
            obs_dim = observations[0].shape[0] if observations else 39
            act_dim = actions[0].shape[0] if actions else 4
            
            observations.extend([np.zeros(obs_dim)] * pad_length)
            actions.extend([np.zeros(act_dim)] * pad_length)
        
        mask = torch.zeros(self.max_trajectory_length)
        mask[:length] = 1.0
        
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
        act_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
        
        return obs_tensor, act_tensor, mask

def create_preference_dataloader(
    data: List[Tuple],
    instruction_encoder: Callable,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_trajectory_length: int = 150,
) -> DataLoader:
    dataset = PreferenceDataset(
        data=data,
        instruction_encoder=instruction_encoder,
        max_trajectory_length=max_trajectory_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

