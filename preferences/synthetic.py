"""
Synthetic preference oracle for trajectory comparison.

Provides heuristic-based preference labeling when human labels are unavailable.
Can be replaced with actual human preferences for real RLHF experiments.
"""

import numpy as np
from typing import List, Tuple, Optional


def trajectory_score(
    traj: List[Tuple[np.ndarray, np.ndarray]],
    task: str = "reach",
) -> float:
    """
    Compute heuristic score for a trajectory.
    
    Args:
        traj: List of (observation, action) pairs
        task: Task type for task-specific scoring
        
    Returns:
        Scalar score (higher is better)
    """
    if len(traj) == 0:
        return -np.inf
        
    final_obs = traj[-1][0]
    
    if task == "reach":
        return _reach_score(traj, final_obs)
    elif task == "push":
        return _push_score(traj, final_obs)
    elif task == "pick-place":
        return _pick_place_score(traj, final_obs)
    else:
        # Default: distance-based scoring
        return _reach_score(traj, final_obs)


def _reach_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    """
    Score for reaching tasks.
    
    MetaWorld reach observation structure:
    - gripper pos: obs[0:3]
    - gripper velocity: obs[3:6] (sometimes)
    - goal pos: obs[-3:] or obs[36:39]
    """
    gripper_pos = final_obs[0:3]
    
    # Goal is typically in the last 3 dimensions
    goal_pos = final_obs[-3:]
    
    # Distance to goal (negative, so closer = higher score)
    distance = np.linalg.norm(gripper_pos - goal_pos)
    
    # Penalize for excessive movement/energy
    energy_penalty = 0.0
    if len(traj) > 1:
        actions = np.array([t[1] for t in traj])
        energy_penalty = 0.01 * np.sum(actions ** 2)
        
    return -distance - energy_penalty


def _push_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    """Score for pushing tasks."""
    # Object position typically around obs[4:7]
    # Goal position typically around obs[-3:]
    obj_pos = final_obs[4:7] if len(final_obs) > 7 else final_obs[0:3]
    goal_pos = final_obs[-3:]
    
    distance = np.linalg.norm(obj_pos - goal_pos)
    return -distance


def _pick_place_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    """Score for pick-and-place tasks."""
    # Object position and goal position
    obj_pos = final_obs[4:7] if len(final_obs) > 7 else final_obs[0:3]
    goal_pos = final_obs[-3:]
    
    distance = np.linalg.norm(obj_pos - goal_pos)
    
    # Bonus for successful grasp (if gripper is closed and near object)
    gripper_pos = final_obs[0:3]
    gripper_to_obj = np.linalg.norm(gripper_pos - obj_pos)
    
    return -distance - 0.1 * gripper_to_obj


def preference(
    traj_A: List[Tuple[np.ndarray, np.ndarray]],
    traj_B: List[Tuple[np.ndarray, np.ndarray]],
    task: str = "reach",
    noise: float = 0.0,
) -> int:
    """
    Determine preference between two trajectories.
    
    Args:
        traj_A: First trajectory
        traj_B: Second trajectory
        task: Task type for scoring
        noise: Probability of random preference (for noise injection)
        
    Returns:
        0 if A is preferred, 1 if B is preferred
    """
    # Optionally inject noise (simulate human uncertainty)
    if noise > 0 and np.random.random() < noise:
        return np.random.randint(2)
        
    score_A = trajectory_score(traj_A, task)
    score_B = trajectory_score(traj_B, task)
    
    return 0 if score_A > score_B else 1


def batch_preference(
    trajectories_A: List[List[Tuple]],
    trajectories_B: List[List[Tuple]],
    task: str = "reach",
    noise: float = 0.0,
) -> List[int]:
    """
    Compute preferences for a batch of trajectory pairs.
    
    Args:
        trajectories_A: List of first trajectories
        trajectories_B: List of second trajectories
        task: Task type for scoring
        noise: Noise probability
        
    Returns:
        List of preferences (0 or 1)
    """
    return [
        preference(traj_A, traj_B, task, noise)
        for traj_A, traj_B in zip(trajectories_A, trajectories_B)
    ]


class PreferenceOracle:
    """
    Configurable preference oracle for experiments.
    
    Can be extended to use learned models or actual human feedback.
    """
    
    def __init__(
        self,
        task: str = "reach",
        noise: float = 0.0,
        use_learned_model: bool = False,
        model_path: Optional[str] = None,
    ):
        self.task = task
        self.noise = noise
        self.use_learned_model = use_learned_model
        self.model = None
        
        if use_learned_model and model_path:
            self._load_model(model_path)
            
    def _load_model(self, path: str):
        """Load a learned preference model."""
        import torch
        # Load reward model for preference prediction
        # This would use the trained reward model to predict preferences
        pass
        
    def __call__(
        self,
        traj_A: List[Tuple],
        traj_B: List[Tuple],
    ) -> int:
        """Get preference for a trajectory pair."""
        if self.use_learned_model and self.model is not None:
            return self._model_preference(traj_A, traj_B)
        else:
            return preference(traj_A, traj_B, self.task, self.noise)
            
    def _model_preference(self, traj_A, traj_B) -> int:
        """Use learned model for preference prediction."""
        # Implement model-based preference
        # This would encode trajectories and use the reward model
        raise NotImplementedError("Model-based preference not yet implemented")
