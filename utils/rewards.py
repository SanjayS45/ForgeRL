import numpy as np
from typing import Optional

def compute_distance_reward(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    scale: float = 1.0,
) -> float:
    distance = np.linalg.norm(ee_pos - goal_pos)
    return -scale * distance

def compute_sparse_reward(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    threshold: float = 0.05,
    success_reward: float = 1.0,
) -> float:
    distance = np.linalg.norm(ee_pos - goal_pos)
    return success_reward if distance < threshold else 0.0

def compute_shaped_reward(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    prev_distance: Optional[float] = None,
    scale: float = 1.0,
) -> float:
    distance = np.linalg.norm(ee_pos - goal_pos)
    
    if prev_distance is None:
        return -scale * distance
    
    improvement = prev_distance - distance
    return scale * improvement

def compute_velocity_penalty(
    action: np.ndarray,
    max_velocity: float = 1.0,
    penalty_scale: float = 0.1,
) -> float:
    velocity_magnitude = np.linalg.norm(action[:3])
    if velocity_magnitude > max_velocity:
        return -penalty_scale * (velocity_magnitude - max_velocity)
    return 0.0

def compute_smoothness_reward(
    current_action: np.ndarray,
    prev_action: Optional[np.ndarray] = None,
    penalty_scale: float = 0.05,
) -> float:
    if prev_action is None:
        return 0.0
    
    action_diff = np.linalg.norm(current_action - prev_action)
    return -penalty_scale * action_diff

def combine_rewards(
    rewards: dict,
    weights: Optional[dict] = None,
) -> float:
    if weights is None:
        weights = {k: 1.0 for k in rewards}
    
    total = 0.0
    for key, reward in rewards.items():
        weight = weights.get(key, 1.0)
        total += weight * reward
    return total

