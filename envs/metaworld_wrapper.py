"""
MetaWorld V3 environment wrappers.

Provides proper instantiation of MetaWorld environments and reward model integration.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import metaworld


class MetaWorldEnv(gym.Env):
    """
    Wrapper for MetaWorld ML1 V3 environments.
    
    Properly handles the Task vs Env distinction in MetaWorld V3:
    - ML1.train_classes contains the environment classes
    - ML1.train_tasks contains task configurations (goal positions, etc.)
    """
    
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
        
        # Initialize MetaWorld ML1
        self.ml1 = metaworld.ML1(task_name, seed=seed)
        
        # Get the environment class from train_classes (NOT train_tasks!)
        env_cls = self.ml1.train_classes[task_name]
        
        # Instantiate the environment
        self._env = env_cls(render_mode=render_mode)
        
        # Set a task (goal configuration)
        task = self.ml1.train_tasks[0]
        self._env.set_task(task)
        
        # Copy spaces from wrapped environment
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        # Track current task index
        self._current_task_idx = 0
        self._tasks = self.ml1.train_tasks
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # MetaWorld reset returns (obs, info) in newer versions
        result = self._env.reset(seed=seed)
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        result = self._env.step(action)
        
        # Handle both 4-tuple (old) and 5-tuple (new gymnasium) returns
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self._env.render()
    
    def close(self):
        """Close the environment."""
        self._env.close()
        
    def set_task_by_index(self, idx: int):
        """Set a specific task by index."""
        self._current_task_idx = idx % len(self._tasks)
        self._env.set_task(self._tasks[self._current_task_idx])
        
    def sample_task(self):
        """Sample and set a random task."""
        idx = np.random.randint(len(self._tasks))
        self.set_task_by_index(idx)
        return idx
    
    @property
    def num_tasks(self) -> int:
        """Number of available tasks."""
        return len(self._tasks)


class RewardModelEnv(gym.Wrapper):
    """
    Wrapper that replaces environment reward with reward model predictions.
    
    Used for training RL policies with learned reward functions.
    """
    
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
        
        # Encode instruction once
        self._instr_encoding = instruction_encoder(instruction)
        if isinstance(self._instr_encoding, np.ndarray):
            self._instr_tensor = torch.tensor(
                self._instr_encoding, dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            self._instr_tensor = self._instr_encoding
            
        # Track trajectory for reward computation
        self._current_obs = None
        self._trajectory = []
        
    def reset(self, **kwargs):
        """Reset environment and trajectory buffer."""
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._trajectory = []
        return obs, info
    
    def step(self, action: np.ndarray):
        """Step with reward model prediction."""
        import torch
        
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Store transition
        self._trajectory.append((self._current_obs.copy(), action.copy()))
        
        # Compute reward from model
        with torch.no_grad():
            obs_tensor = torch.tensor(
                self._current_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
            
            act_tensor = torch.tensor(
                action, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, act_dim)
            
            # Get predicted reward
            predicted_reward = self.reward_model.forward_step(
                obs_tensor, act_tensor, self._instr_tensor
            ).item()
            
        # Combine rewards if needed
        if self.use_env_reward:
            reward = env_reward + self.reward_scale * predicted_reward
        else:
            reward = self.reward_scale * predicted_reward
            
        # Update current obs
        self._current_obs = obs
        
        # Add debug info
        info["env_reward"] = env_reward
        info["predicted_reward"] = predicted_reward
        
        return obs, reward, terminated, truncated, info
    
    def get_trajectory(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the current trajectory."""
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
    """
    Factory function to create MetaWorld environments.
    
    Args:
        task_name: MetaWorld task name (e.g., 'reach-v3', 'push-v3')
        seed: Random seed for reproducibility
        reward_model: Optional trained reward model
        instruction: Language instruction for the task
        instruction_encoder: Function to encode instructions
        device: Device for reward model inference
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        
    Returns:
        A gym.Env instance, optionally wrapped with reward model
    """
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

