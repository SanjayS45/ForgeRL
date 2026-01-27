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
    ):
        super().__init__()
        
        self.task_name = task_name
        self.seed_value = seed
        self.render_mode = render_mode
        

        self.ml1 = metaworld.ML1(task_name, seed=seed)
        

        env_cls = self.ml1.train_classes[task_name]
        

        self._env = env_cls(render_mode=render_mode)
        

        task = self.ml1.train_tasks[0]
        self._env.set_task(task)
        

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        

        self._current_task_idx = 0
        self._tasks = self.ml1.train_tasks
        
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
            
        return obs, info
    
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
) -> gym.Env:
    
    env = MetaWorldEnv(task_name=task_name, seed=seed, render_mode=render_mode)
    
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

