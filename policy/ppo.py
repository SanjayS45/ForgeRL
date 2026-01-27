"""
Proximal Policy Optimization (PPO) implementation.

Clean, modular PPO for continuous action spaces with support for
learned reward models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from collections import deque
import gymnasium as gym
from tqdm import tqdm

from policy.networks import ActorCritic


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """
    
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start = 0
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add a transition to the buffer."""
        assert self.ptr < self.buffer_size
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        
    def finish_path(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute advantages using GAE when a trajectory ends.
        """
        path_slice = slice(self.path_start, self.ptr)
        rewards = self.rewards[path_slice]
        values = np.append(self.values[path_slice], last_value)
        dones = self.dones[path_slice]
        
        # GAE computation
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + self.values[path_slice]
        
        self.path_start = self.ptr
        
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        # Normalize advantages
        adv_mean = self.advantages[:self.ptr].mean()
        adv_std = self.advantages[:self.ptr].std() + 1e-8
        advantages = (self.advantages[:self.ptr] - adv_mean) / adv_std
        
        return {
            "observations": torch.tensor(self.observations[:self.ptr], device=self.device),
            "actions": torch.tensor(self.actions[:self.ptr], device=self.device),
            "log_probs": torch.tensor(self.log_probs[:self.ptr], device=self.device),
            "advantages": torch.tensor(advantages, device=self.device),
            "returns": torch.tensor(self.returns[:self.ptr], device=self.device),
            "values": torch.tensor(self.values[:self.ptr], device=self.device),
        }
        
    def reset(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start = 0


class PPOAgent:
    """
    PPO Agent for continuous control.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        device: str = "auto",
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        target_kl: Optional[float] = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        
        # Optimizer
        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy.
        
        Returns:
            action: Action array
            value: State value
            log_prob: Log probability
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.actor_critic.get_action_and_value(
                obs_t, deterministic=deterministic
            )
            
        return (
            action.cpu().numpy()[0],
            value.cpu().item(),
            log_prob.cpu().item(),
        )
        
    def update(self, buffer: RolloutBuffer, epochs: int = 10, batch_size: int = 64) -> Dict:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        data = buffer.get()
        
        n_samples = len(data["observations"])
        indices = np.arange(n_samples)
        
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_fraction": [],
        }
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                obs = data["observations"][batch_idx]
                actions = data["actions"][batch_idx]
                old_log_probs = data["log_probs"][batch_idx]
                advantages = data["advantages"][batch_idx]
                returns = data["returns"][batch_idx]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.actor_critic.get_action_and_value(
                    obs, action=actions
                )
                
                # Policy loss
                ratio = (new_log_probs - old_log_probs).exp()
                clip_ratio_tensor = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * advantages, clip_ratio_tensor * advantages).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    kl = (old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                    
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["kl"].append(kl)
                metrics["clip_fraction"].append(clip_frac)
                
            # Early stopping on KL divergence
            if self.target_kl is not None:
                avg_kl = np.mean(metrics["kl"][-n_samples // batch_size:])
                if avg_kl > self.target_kl:
                    break
                    
        return {k: np.mean(v) for k, v in metrics.items()}
        
    def save(self, path: str):
        """Save agent."""
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class PPOTrainer:
    """
    Trainer for PPO with reward model integration.
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent: PPOAgent,
        reward_model: Optional[nn.Module] = None,
        instruction: Optional[str] = None,
        instruction_encoder: Optional[callable] = None,
        use_env_reward: bool = True,
        reward_scale: float = 1.0,
        log_dir: str = "logs/ppo",
        custom_goals: Optional[List[List[float]]] = None,
    ):
        self.env = env
        self.agent = agent
        self.reward_model = reward_model
        self.instruction = instruction
        self.instruction_encoder = instruction_encoder
        self.use_env_reward = use_env_reward
        self.reward_scale = reward_scale
        self.custom_goals = custom_goals  # User-defined goal positions
        
        # Encode instruction if using reward model
        self._instr_tensor = None
        if reward_model is not None and instruction is not None:
            if instruction_encoder is not None:
                enc = instruction_encoder(instruction)
                self._instr_tensor = torch.tensor(
                    enc, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Log configuration
        print(f"[PPOTrainer] Initialized with instruction: {instruction}")
        print(f"[PPOTrainer] Using env reward: {use_env_reward}")
        print(f"[PPOTrainer] Reward model: {'Loaded' if reward_model else 'None'}")
        if custom_goals:
            print(f"[PPOTrainer] Custom goals: {len(custom_goals)} targets provided")
        
    def _compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        env_reward: float,
    ) -> float:
        """Compute reward from model or environment."""
        if self.reward_model is None:
            return env_reward
            
        with torch.no_grad():
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self.agent.device
            ).unsqueeze(0).unsqueeze(0)
            act_t = torch.tensor(
                action, dtype=torch.float32, device=self.agent.device
            ).unsqueeze(0).unsqueeze(0)
            
            predicted = self.reward_model.forward_step(
                obs_t, act_t, self._instr_tensor
            ).item()
            
        if self.use_env_reward:
            return env_reward + self.reward_scale * predicted
        else:
            return self.reward_scale * predicted
            
    def train(
        self,
        total_timesteps: int,
        steps_per_update: int = 2048,
        epochs_per_update: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        checkpoint_freq: int = 10,
        log_freq: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total environment steps
            steps_per_update: Steps between policy updates
            epochs_per_update: PPO epochs per update
            batch_size: Mini-batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
            checkpoint_freq: Checkpoint frequency (updates)
            log_freq: Logging frequency (updates)
            progress_callback: Optional callback(step, total, episode_reward, policy_loss)
            
        Returns:
            Training history
        """
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        
        buffer = RolloutBuffer(
            steps_per_update, obs_dim, act_dim, device=self.agent.device
        )
        
        history = {
            "episode_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        
        # Initialize
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        num_updates = total_timesteps // steps_per_update
        global_step = 0
        
        pbar = tqdm(range(num_updates), desc="Training")
        
        for update in pbar:
            buffer.reset()
            
            # Collect rollout
            for step in range(steps_per_update):
                # Get action
                action, value, log_prob = self.agent.get_action(obs)
                
                # Step environment
                next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Compute reward
                reward = self._compute_reward(obs, action, env_reward)
                
                # Store transition
                buffer.add(obs, action, reward, done, value, log_prob)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                global_step += 1
                
                if done:
                    # Finish path with zero value (terminal)
                    buffer.finish_path(0.0, gamma, gae_lambda)
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    
            # Bootstrap value for non-terminal state
            if not done:
                _, last_value, _ = self.agent.get_action(obs)
                buffer.finish_path(last_value, gamma, gae_lambda)
                
            # Update policy
            update_metrics = self.agent.update(buffer, epochs_per_update, batch_size)
            
            # Logging
            mean_reward = 0.0
            policy_loss = update_metrics["policy_loss"]
            
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                
                history["episode_reward"].append(mean_reward)
                history["episode_length"].append(mean_length)
                history["policy_loss"].append(policy_loss)
                history["value_loss"].append(update_metrics["value_loss"])
                history["entropy"].append(update_metrics["entropy"])
                
                pbar.set_postfix({
                    "reward": f"{mean_reward:.2f}",
                    "length": f"{mean_length:.0f}",
                    "loss": f"{policy_loss:.4f}",
                })
            
            # Progress callback for UI updates
            if progress_callback is not None:
                try:
                    progress_callback(global_step, total_timesteps, mean_reward, policy_loss)
                except Exception as e:
                    print(f"[PPOTrainer] Progress callback error: {e}")
                
            # Checkpoint
            if (update + 1) % checkpoint_freq == 0:
                ckpt_path = self.log_dir / f"checkpoint_{update + 1}.pt"
                self.agent.save(str(ckpt_path))
                
        # Save final model
        self.agent.save(str(self.log_dir / "final_model.pt"))
        
        # Save history
        import json
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
            
        return history

