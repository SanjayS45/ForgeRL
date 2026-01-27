"""
Neural network architectures for RL policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Initialize layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Actor network for continuous action spaces.
    
    Outputs mean and log_std for Gaussian policy.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build MLP
        layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
            
        self.trunk = nn.Sequential(*layers)
        
        # Output heads
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.log_std_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observations (B, obs_dim)
            
        Returns:
            mean: Action means (B, act_dim)
            log_std: Log standard deviations (B, act_dim)
        """
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
        
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observations
            deterministic: Use mean action if True
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            mean: Mean action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        # Squash to [-1, 1]
        action = torch.tanh(action)
        
        # Correct log_prob for tanh squashing
        if log_prob is not None:
            log_prob = log_prob - (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1)
            
        return action, log_prob, mean


class Critic(nn.Module):
    """
    Critic network (value function).
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
            
        layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict state value.
        
        Args:
            obs: Observations (B, obs_dim)
            
        Returns:
            value: State values (B,)
        """
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        shared_backbone: bool = False,
    ):
        super().__init__()
        
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # Shared feature extractor
            layers = []
            input_dim = obs_dim
            for i in range(num_layers - 1):
                layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
                layers.append(nn.Tanh())
                input_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            
            # Separate heads
            self.actor_head = nn.Sequential(
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim * 2), std=0.01),
            )
            self.critic_head = nn.Sequential(
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
        else:
            self.actor = Actor(obs_dim, act_dim, hidden_dim, num_layers)
            self.critic = Critic(obs_dim, hidden_dim, num_layers)
            
        self.act_dim = act_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Action mean
            log_std: Action log std
            value: State value
        """
        if self.shared_backbone:
            features = self.backbone(obs)
            
            actor_out = self.actor_head(features)
            mean = actor_out[..., :self.act_dim]
            log_std = actor_out[..., self.act_dim:]
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            value = self.critic_head(features).squeeze(-1)
        else:
            mean, log_std = self.actor.forward(obs)
            value = self.critic.forward(obs)
            
        return mean, log_std, value
        
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log prob, entropy, and value.
        
        Args:
            obs: Observations
            action: Optional action (for computing log prob of given action)
            deterministic: Use mean action if True
            
        Returns:
            action: Action tensor
            log_prob: Log probability
            entropy: Distribution entropy
            value: State value
        """
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        if action is None:
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
                
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Tanh squashing
        action_squashed = torch.tanh(action)
        log_prob = log_prob - (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1)
        
        return action_squashed, log_prob, entropy, value
        
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value only."""
        if self.shared_backbone:
            features = self.backbone(obs)
            return self.critic_head(features).squeeze(-1)
        else:
            return self.critic.forward(obs)

