import pickle
import json
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

Trajectory = List[Tuple[np.ndarray, np.ndarray]]
PreferencePair = Tuple[str, Trajectory, Trajectory]
Dataset = List[PreferencePair]

@dataclass
class DatasetStats:
    
    num_pairs: int
    num_unique_instructions: int
    avg_trajectory_length: float
    min_trajectory_length: int
    max_trajectory_length: int
    obs_dim: int
    act_dim: int
    instructions: List[str]
    
    def __str__(self):
        return (
            f"Dataset Statistics:\n"
            f"  Pairs: {self.num_pairs}\n"
            f"  Unique Instructions: {self.num_unique_instructions}\n"
            f"  Avg Trajectory Length: {self.avg_trajectory_length:.1f}\n"
            f"  Length Range: [{self.min_trajectory_length}, {self.max_trajectory_length}]\n"
            f"  Observation Dim: {self.obs_dim}\n"
            f"  Action Dim: {self.act_dim}"
        )

def load_dataset(path: Union[str, Path]) -> Dataset:
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
        
    suffix = path.suffix.lower()
    
    if suffix in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)

        data = _json_to_numpy(data)
    elif suffix == ".csv":
        data = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
        
    return data

def save_dataset(data: Dataset, path: Union[str, Path], format: str = "pickle"):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif format == "json":
        json_data = _numpy_to_json(data)
        with open(path, "w") as f:
            json.dump(json_data, f)
    elif format == "csv":
        _save_csv(data, path)
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    print(f"Saved {len(data)} pairs to {path}")

def validate_dataset(data: Dataset) -> Tuple[bool, List[str]]:
    
    errors = []
    
    if not isinstance(data, list):
        errors.append("Dataset must be a list")
        return False, errors
        
    if len(data) == 0:
        errors.append("Dataset is empty")
        return False, errors
        
    for i, item in enumerate(data):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            errors.append(f"Item {i}: Expected (instruction, traj_a, traj_b) tuple")
            continue
            
        instr, traj_a, traj_b = item
        
        if not isinstance(instr, str):
            errors.append(f"Item {i}: Instruction must be a string")
            
        for traj_name, traj in [("traj_a", traj_a), ("traj_b", traj_b)]:
            if not isinstance(traj, list):
                errors.append(f"Item {i}: {traj_name} must be a list")
                continue
                
            if len(traj) == 0:
                errors.append(f"Item {i}: {traj_name} is empty")
                continue
                
            for j, step in enumerate(traj):
                if not isinstance(step, (tuple, list)) or len(step) != 2:
                    errors.append(f"Item {i}, {traj_name}[{j}]: Expected (obs, action) pair")
                    break
                    
    is_valid = len(errors) == 0
    return is_valid, errors

def compute_stats(data: Dataset) -> DatasetStats:
    
    if not data:
        raise ValueError("Empty dataset")
        
    instructions = set()
    traj_lengths = []
    obs_dim = None
    act_dim = None
    
    for instr, traj_a, traj_b in data:
        instructions.add(instr)
        traj_lengths.append(len(traj_a))
        traj_lengths.append(len(traj_b))
        
        if obs_dim is None and traj_a:
            obs_dim = len(traj_a[0][0])
            act_dim = len(traj_a[0][1])
            
    return DatasetStats(
        num_pairs=len(data),
        num_unique_instructions=len(instructions),
        avg_trajectory_length=np.mean(traj_lengths),
        min_trajectory_length=min(traj_lengths),
        max_trajectory_length=max(traj_lengths),
        obs_dim=obs_dim or 0,
        act_dim=act_dim or 0,
        instructions=sorted(instructions),
    )

def preprocess_dataset(
    data: Dataset,
    max_length: Optional[int] = None,
    normalize_obs: bool = False,
    normalize_actions: bool = False,
) -> Dataset:
    
    processed = []
    

    if normalize_obs or normalize_actions:
        all_obs = []
        all_acts = []
        for _, traj_a, traj_b in data:
            for traj in [traj_a, traj_b]:
                for obs, act in traj:
                    all_obs.append(obs)
                    all_acts.append(act)
        
        if normalize_obs:
            all_obs = np.array(all_obs)
            obs_mean = all_obs.mean(axis=0)
            obs_std = all_obs.std(axis=0) + 1e-8
        if normalize_actions:
            all_acts = np.array(all_acts)
            act_mean = all_acts.mean(axis=0)
            act_std = all_acts.std(axis=0) + 1e-8
    
    for instr, traj_a, traj_b in tqdm(data, desc="Preprocessing"):

        new_traj_a = _process_trajectory(
            traj_a, max_length,
            obs_mean if normalize_obs else None,
            obs_std if normalize_obs else None,
            act_mean if normalize_actions else None,
            act_std if normalize_actions else None,
        )
        new_traj_b = _process_trajectory(
            traj_b, max_length,
            obs_mean if normalize_obs else None,
            obs_std if normalize_obs else None,
            act_mean if normalize_actions else None,
            act_std if normalize_actions else None,
        )
        
        processed.append((instr, new_traj_a, new_traj_b))
        
    return processed

def _process_trajectory(
    traj: Trajectory,
    max_length: Optional[int],
    obs_mean: Optional[np.ndarray],
    obs_std: Optional[np.ndarray],
    act_mean: Optional[np.ndarray],
    act_std: Optional[np.ndarray],
) -> Trajectory:
    

    if max_length is not None:
        traj = traj[:max_length]
        

    new_traj = []
    for obs, act in traj:
        obs = np.array(obs, dtype=np.float32)
        act = np.array(act, dtype=np.float32)
        
        if obs_mean is not None:
            obs = (obs - obs_mean) / obs_std
        if act_mean is not None:
            act = (act - act_mean) / act_std
            
        new_traj.append((obs, act))
        
    return new_traj

def split_dataset(
    data: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    
    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]
    
    return train_data, val_data, test_data

def shuffle_dataset(data: Dataset, seed: Optional[int] = None) -> Dataset:
    
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(data))
    return [data[i] for i in indices]

def merge_datasets(*datasets: Dataset) -> Dataset:
    
    merged = []
    for d in datasets:
        merged.extend(d)
    return merged

def _numpy_to_json(data: Dataset) -> List:
    
    json_data = []
    for instr, traj_a, traj_b in data:
        json_traj_a = [(obs.tolist(), act.tolist()) for obs, act in traj_a]
        json_traj_b = [(obs.tolist(), act.tolist()) for obs, act in traj_b]
        json_data.append([instr, json_traj_a, json_traj_b])
    return json_data

def _json_to_numpy(data: List) -> Dataset:
    
    numpy_data = []
    for item in data:
        instr = item[0]
        traj_a = [(np.array(obs), np.array(act)) for obs, act in item[1]]
        traj_b = [(np.array(obs), np.array(act)) for obs, act in item[2]]
        numpy_data.append((instr, traj_a, traj_b))
    return numpy_data

def _load_csv(path: Path) -> Dataset:
    
    import base64
    
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instr = row["instruction"]
            traj_a = pickle.loads(base64.b64decode(row["traj_a"]))
            traj_b = pickle.loads(base64.b64decode(row["traj_b"]))
            data.append((instr, traj_a, traj_b))
    return data

def _save_csv(data: Dataset, path: Path):
    
    import base64
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "traj_a", "traj_b"])
        writer.writeheader()
        for instr, traj_a, traj_b in data:
            writer.writerow({
                "instruction": instr,
                "traj_a": base64.b64encode(pickle.dumps(traj_a)).decode(),
                "traj_b": base64.b64encode(pickle.dumps(traj_b)).decode(),
            })

