import numpy as np
from typing import Tuple

def normalize_observation(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / (std + 1e-8)

def denormalize_observation(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return obs * (std + 1e-8) + mean

def normalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return 2.0 * (action - low) / (high - low + 1e-8) - 1.0

def denormalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (action + 1.0) / 2.0 * (high - low) + low

def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(action, low, high)

def compute_running_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def augment_observation(obs: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    noise = np.random.randn(*obs.shape) * noise_scale
    return obs + noise

def augment_action(action: np.ndarray, noise_scale: float = 0.05) -> np.ndarray:
    noise = np.random.randn(*action.shape) * noise_scale
    return action + noise

def extract_state_features(obs: np.ndarray) -> dict:
    return {
        "ee_pos": obs[:3],
        "ee_vel": obs[3:6] if len(obs) > 6 else np.zeros(3),
        "gripper": obs[6] if len(obs) > 6 else 0.0,
        "goal_pos": obs[36:39] if len(obs) >= 39 else obs[-3:],
    }

