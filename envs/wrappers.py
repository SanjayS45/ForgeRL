import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean = np.zeros(env.observation_space.shape)
        self.running_var = np.ones(env.observation_space.shape)
        self.count = 0
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        self.running_var += delta * (observation - self.running_mean)
        
        std = np.sqrt(self.running_var / max(self.count, 1)) + self.epsilon
        return (observation - self.running_mean) / std

class ClipAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)

class TimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_steps: int = 150):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class RewardScaling(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float = 1.0, shift: float = 0.0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift
    
    def reward(self, reward: float) -> float:
        return reward * self.scale + self.shift

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0
        self._current_length = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_reward += reward
        self._current_length += 1
        
        if terminated or truncated:
            info["episode"] = {
                "r": self._current_reward,
                "l": self._current_length,
            }
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._current_reward = 0
        self._current_length = 0
        return self.env.reset(**kwargs)

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        
        self.observation_space = gym.spaces.Box(
            low=low.flatten(),
            high=high.flatten(),
            dtype=env.observation_space.dtype,
        )
        
        self.frames = np.zeros(
            (num_stack,) + env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = observation
        return self.frames.flatten()
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = 0
        self.frames[-1] = obs
        return self.frames.flatten(), info

