import numpy as np
from typing import List, Tuple, Optional, Callable
import gymnasium as gym

def collect_rollout(
    env: gym.Env,
    policy: Callable,
    max_steps: int = 150,
    deterministic: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float, bool]:
    trajectory = []
    total_reward = 0.0
    success = False
    
    obs, info = env.reset()
    
    for step in range(max_steps):
        action = policy(obs, deterministic=deterministic)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        trajectory.append((obs.copy(), action.copy()))
        total_reward += reward
        obs = next_obs
        
        if info.get("success", False):
            success = True
            
        if terminated or truncated:
            break
    
    return trajectory, total_reward, success

def collect_multiple_rollouts(
    env: gym.Env,
    policy: Callable,
    num_rollouts: int,
    max_steps: int = 150,
    deterministic: bool = False,
) -> List[Tuple[List, float, bool]]:
    results = []
    
    for _ in range(num_rollouts):
        traj, reward, success = collect_rollout(
            env, policy, max_steps, deterministic
        )
        results.append((traj, reward, success))
    
    return results

def evaluate_policy(
    env: gym.Env,
    policy: Callable,
    num_episodes: int = 10,
    max_steps: int = 150,
) -> dict:
    rewards = []
    successes = []
    lengths = []
    
    for _ in range(num_episodes):
        traj, reward, success = collect_rollout(
            env, policy, max_steps, deterministic=True
        )
        rewards.append(reward)
        successes.append(success)
        lengths.append(len(traj))
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
    }

def random_policy(obs: np.ndarray, action_space: gym.spaces.Box, deterministic: bool = False) -> np.ndarray:
    return action_space.sample()

def create_random_policy(env: gym.Env) -> Callable:
    def policy(obs, deterministic=False):
        return env.action_space.sample()
    return policy

