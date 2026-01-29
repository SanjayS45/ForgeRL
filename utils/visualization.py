import numpy as np
from typing import List, Tuple, Optional
import json

def trajectory_to_json(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    data = []
    for obs, act in trajectory:
        data.append({
            "observation": obs.tolist() if hasattr(obs, 'tolist') else list(obs),
            "action": act.tolist() if hasattr(act, 'tolist') else list(act)
        })
    return json.dumps(data)

def extract_end_effector_path(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> List[List[float]]:
    path = []
    for obs, _ in trajectory:
        path.append([float(obs[0]), float(obs[1]), float(obs[2])])
    return path

def extract_goal_positions(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> List[List[float]]:
    goals = []
    for obs, _ in trajectory:
        if len(obs) >= 39:
            goals.append([float(obs[36]), float(obs[37]), float(obs[38])])
    return goals

def compute_path_length(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if len(trajectory) < 2:
        return 0.0
    
    total = 0.0
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i-1][0][:3]
        curr_pos = trajectory[i][0][:3]
        total += np.linalg.norm(curr_pos - prev_pos)
    return float(total)

def interpolate_trajectory(trajectory: List[Tuple[np.ndarray, np.ndarray]], num_points: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(trajectory) <= 1:
        return trajectory
    
    if num_points <= len(trajectory):
        indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
        return [trajectory[i] for i in indices]
    
    return trajectory

def format_observation_for_display(obs: np.ndarray) -> dict:
    return {
        "end_effector_pos": obs[:3].tolist(),
        "gripper_state": float(obs[3]) if len(obs) > 3 else 0.0,
        "goal_pos": obs[36:39].tolist() if len(obs) >= 39 else obs[-3:].tolist(),
    }

