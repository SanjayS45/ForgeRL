import numpy as np
from typing import List, Tuple, Dict

def compute_trajectory_success(trajectory: List[Tuple[np.ndarray, np.ndarray]], threshold: float = 0.05) -> bool:
    if len(trajectory) == 0:
        return False
    
    final_obs = trajectory[-1][0]
    ee_pos = final_obs[:3]
    goal_pos = final_obs[-3:] if len(final_obs) >= 39 else final_obs[36:39]
    
    distance = np.linalg.norm(ee_pos - goal_pos)
    return distance < threshold

def compute_dataset_success_rate(data: List, threshold: float = 0.05) -> Dict:
    good_successes = 0
    bad_successes = 0
    total = len(data)
    
    for item in data:
        if len(item) >= 3:
            traj_good = item[1]
            traj_bad = item[2]
            
            if compute_trajectory_success(traj_good, threshold):
                good_successes += 1
            if compute_trajectory_success(traj_bad, threshold):
                bad_successes += 1
    
    return {
        "total_pairs": total,
        "good_traj_success_rate": good_successes / total if total > 0 else 0,
        "bad_traj_success_rate": bad_successes / total if total > 0 else 0,
        "threshold": threshold,
    }

def compute_trajectory_distance_curve(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    distances = []
    
    for obs, _ in trajectory:
        ee_pos = obs[:3]
        goal_pos = obs[-3:] if len(obs) >= 39 else obs[36:39]
        distance = np.linalg.norm(ee_pos - goal_pos)
        distances.append(float(distance))
    
    return distances

def compute_trajectory_smoothness(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if len(trajectory) < 2:
        return 0.0
    
    action_diffs = []
    for i in range(1, len(trajectory)):
        prev_action = trajectory[i-1][1]
        curr_action = trajectory[i][1]
        diff = np.linalg.norm(curr_action - prev_action)
        action_diffs.append(diff)
    
    return float(np.mean(action_diffs))

def compute_trajectory_efficiency(trajectory: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if len(trajectory) == 0:
        return 0.0
    
    initial_obs = trajectory[0][0]
    final_obs = trajectory[-1][0]
    
    initial_dist = np.linalg.norm(initial_obs[:3] - initial_obs[-3:])
    final_dist = np.linalg.norm(final_obs[:3] - final_obs[-3:])
    
    total_path_length = sum(
        np.linalg.norm(trajectory[i][0][:3] - trajectory[i-1][0][:3])
        for i in range(1, len(trajectory))
    )
    
    direct_distance = np.linalg.norm(final_obs[:3] - initial_obs[:3])
    
    if total_path_length > 0:
        return float(direct_distance / total_path_length)
    return 0.0

