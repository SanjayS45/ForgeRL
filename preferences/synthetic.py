import numpy as np
from typing import List, Tuple, Optional

def trajectory_score(
    traj: List[Tuple[np.ndarray, np.ndarray]],
    task: str = "reach",
) -> float:
    
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

        return _reach_score(traj, final_obs)

def _reach_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    
    gripper_pos = final_obs[0:3]
    

    goal_pos = final_obs[-3:]
    

    distance = np.linalg.norm(gripper_pos - goal_pos)
    

    energy_penalty = 0.0
    if len(traj) > 1:
        actions = np.array([t[1] for t in traj])
        energy_penalty = 0.01 * np.sum(actions ** 2)
        
    return -distance - energy_penalty

def _push_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    

    obj_pos = final_obs[4:7] if len(final_obs) > 7 else final_obs[0:3]
    goal_pos = final_obs[-3:]
    
    distance = np.linalg.norm(obj_pos - goal_pos)
    return -distance

def _pick_place_score(traj: List[Tuple], final_obs: np.ndarray) -> float:
    

    obj_pos = final_obs[4:7] if len(final_obs) > 7 else final_obs[0:3]
    goal_pos = final_obs[-3:]
    
    distance = np.linalg.norm(obj_pos - goal_pos)
    

    gripper_pos = final_obs[0:3]
    gripper_to_obj = np.linalg.norm(gripper_pos - obj_pos)
    
    return -distance - 0.1 * gripper_to_obj

def preference(
    traj_A: List[Tuple[np.ndarray, np.ndarray]],
    traj_B: List[Tuple[np.ndarray, np.ndarray]],
    task: str = "reach",
    noise: float = 0.0,
) -> int:
    

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
    
    return [
        preference(traj_A, traj_B, task, noise)
        for traj_A, traj_B in zip(trajectories_A, trajectories_B)
    ]

class PreferenceOracle:
    
    
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
        
        import torch

        pass
        
    def __call__(
        self,
        traj_A: List[Tuple],
        traj_B: List[Tuple],
    ) -> int:
        
        if self.use_learned_model and self.model is not None:
            return self._model_preference(traj_A, traj_B)
        else:
            return preference(traj_A, traj_B, self.task, self.noise)
            
    def _model_preference(self, traj_A, traj_B) -> int:
        

        raise NotImplementedError("Model-based preference not yet implemented")
