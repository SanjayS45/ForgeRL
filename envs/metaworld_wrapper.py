import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import metaworld

class MetaWorldEnv(gym.Env):
    
    def __init__(
        self,
        task_name: str = "reach-v3",
        seed: int = 42,
        render_mode: Optional[str] = None,
        custom_goal: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.task_name = task_name
        self.seed_value = seed
        self.render_mode = render_mode
        self.custom_goal = custom_goal

        self.ml1 = metaworld.ML1(task_name, seed=seed)

        env_cls = self.ml1.train_classes[task_name]

        self._env = env_cls(render_mode=render_mode)

        task = self.ml1.train_tasks[0]
        self._env.set_task(task)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._current_task_idx = 0
        self._tasks = self.ml1.train_tasks
        self._goal_position = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        if seed is not None:
            np.random.seed(seed)

        result = self._env.reset(seed=seed)
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        if self.custom_goal is not None:
            self._goal_position = np.array(self.custom_goal)
            obs = self._inject_goal_into_obs(obs)
            info['custom_goal'] = self._goal_position.tolist()
        else:
            self._goal_position = self._extract_goal_from_obs(obs)
            
        info['goal_position'] = self._goal_position.tolist() if self._goal_position is not None else None
            
        return obs, info
    
    def _extract_goal_from_obs(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) >= 39:
            return obs[36:39].copy()
        return None
    
    def _inject_goal_into_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._goal_position is not None and len(obs) >= 39:
            obs = obs.copy()
            obs[36:39] = self._goal_position
        return obs
    
    def set_custom_goal(self, goal: List[float]):
        self.custom_goal = goal
        self._goal_position = np.array(goal)
    
    def get_goal_position(self) -> Optional[np.ndarray]:
        return self._goal_position
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        
        result = self._env.step(action)
        

        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        
        return self._env.render()
    
    def close(self):
        
        self._env.close()
        
    def set_task_by_index(self, idx: int):
        
        self._current_task_idx = idx % len(self._tasks)
        self._env.set_task(self._tasks[self._current_task_idx])
        
    def sample_task(self):
        
        idx = np.random.randint(len(self._tasks))
        self.set_task_by_index(idx)
        return idx
    
    @property
    def num_tasks(self) -> int:
        
        return len(self._tasks)

class RewardModelEnv(gym.Wrapper):
    
    
    def __init__(
        self,
        env: gym.Env,
        reward_model: "torch.nn.Module",
        instruction: str,
        instruction_encoder: callable,
        device: str = "cpu",
        use_env_reward: bool = False,
        reward_scale: float = 1.0,
    ):
        super().__init__(env)
        
        import torch
        
        self.reward_model = reward_model
        self.instruction = instruction
        self.instruction_encoder = instruction_encoder
        self.device = device
        self.use_env_reward = use_env_reward
        self.reward_scale = reward_scale
        

        self._instr_encoding = instruction_encoder(instruction)
        if isinstance(self._instr_encoding, np.ndarray):
            self._instr_tensor = torch.tensor(
                self._instr_encoding, dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            self._instr_tensor = self._instr_encoding
            

        self._current_obs = None
        self._trajectory = []
        
    def reset(self, **kwargs):
        
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._trajectory = []
        return obs, info
    
    def step(self, action: np.ndarray):
        
        import torch
        
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        

        self._trajectory.append((self._current_obs.copy(), action.copy()))
        

        with torch.no_grad():
            obs_tensor = torch.tensor(
                self._current_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            
            act_tensor = torch.tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            

            predicted_reward = self.reward_model.forward_step(
                obs_tensor, act_tensor, self._instr_tensor
            ).item()
            

        if self.use_env_reward:
            reward = env_reward + self.reward_scale * predicted_reward
        else:
            reward = self.reward_scale * predicted_reward
            

        self._current_obs = obs
        

        info["env_reward"] = env_reward
        info["predicted_reward"] = predicted_reward
        
        return obs, reward, terminated, truncated, info
    
    def get_trajectory(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        
        return self._trajectory.copy()

def make_metaworld_env(
    task_name: str = "reach-v3",
    seed: int = 42,
    reward_model: Optional["torch.nn.Module"] = None,
    instruction: Optional[str] = None,
    instruction_encoder: Optional[callable] = None,
    device: str = "cpu",
    render_mode: Optional[str] = None,
    custom_goal: Optional[List[float]] = None,
    custom_goals: Optional[List[List[float]]] = None,
) -> gym.Env:
    
    goal_to_use = custom_goal
    if custom_goals is not None and len(custom_goals) > 0:
        goal_to_use = custom_goals[np.random.randint(len(custom_goals))]
    
    env = MetaWorldEnv(
        task_name=task_name, 
        seed=seed, 
        render_mode=render_mode,
        custom_goal=goal_to_use,
    )
    
    if reward_model is not None and instruction is not None:
        if instruction_encoder is None:
            from envs.language import encode_instruction
            instruction_encoder = encode_instruction
            
        env = RewardModelEnv(
            env=env,
            reward_model=reward_model,
            instruction=instruction,
            instruction_encoder=instruction_encoder,
            device=device,
        )
        
    return env


def generate_goal_conditioned_trajectory(
    env: gym.Env,
    goal: List[float],
    horizon: int = 50,
    noise_scale: float = 0.1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    
    if hasattr(env, 'set_custom_goal'):
        env.set_custom_goal(goal)
    
    obs, _ = env.reset()
    trajectory = []
    
    for _ in range(horizon):
        goal_pos = np.array(goal)
        ee_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        
        direction = goal_pos - ee_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        action = np.zeros(env.action_space.shape[0])
        action[:3] = direction * 0.5
        action = action + np.random.randn(*action.shape) * noise_scale
        action = np.clip(action, -1, 1)
        
        trajectory.append((obs.copy(), action.copy()))
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
    
    return trajectory

