"""
Neural reward models for trajectory preference learning.

Implements Bradley-Terry preference model for learning from pairwise comparisons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class RewardModel(nn.Module):
    """
    Reward model using embedding-based instruction encoding.
    
    Takes (observation, action, instruction) and outputs scalar reward.
    For trajectory comparison, rewards are summed over timesteps.
    """
    
    def __init__(
        self,
        obs_dim: int = 39,
        act_dim: int = 4,
        instr_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.instr_dim = instr_dim
        
        # Instruction embedding
        self.instr_emb = nn.Embedding(vocab_size, instr_dim)
        
        # Build MLP layers
        layers = []
        input_dim = obs_dim + act_dim + instr_dim
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
            input_dim = hidden_dim
            
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                
    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute trajectory reward.
        
        Args:
            obs: Observations (B, T, obs_dim)
            act: Actions (B, T, act_dim)
            instr: Instruction IDs (B,) - integer tensor
            
        Returns:
            Total trajectory rewards (B,)
        """
        B, T, _ = obs.shape
        
        # Get instruction embedding and expand to all timesteps
        instr_e = self.instr_emb(instr)  # (B, instr_dim)
        instr_e = instr_e.unsqueeze(1).expand(-1, T, -1)  # (B, T, instr_dim)
        
        # Concatenate inputs
        x = torch.cat([obs, act, instr_e], dim=-1)  # (B, T, input_dim)
        
        # Compute per-step rewards
        r = self.net(x).squeeze(-1)  # (B, T)
        
        # Sum over trajectory
        return r.sum(dim=1)  # (B,)
        
    def forward_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute single-step reward.
        
        Args:
            obs: Single observation (B, 1, obs_dim) or (B, obs_dim)
            act: Single action (B, 1, act_dim) or (B, act_dim)
            instr: Instruction IDs (B,)
            
        Returns:
            Step rewards (B,)
        """
        # Add time dimension if needed
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            act = act.unsqueeze(1)
            
        return self.forward(obs, act, instr)


class RewardModelWithSentenceEncoder(nn.Module):
    """
    Reward model using sentence transformer embeddings for instructions.
    
    This is the recommended model for production use, as it can handle
    arbitrary natural language instructions.
    """
    
    def __init__(
        self,
        obs_dim: int = 39,
        act_dim: int = 4,
        instr_dim: int = 384,  # Sentence transformer embedding dim
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.instr_dim = instr_dim
        self.use_attention = use_attention
        
        # Instruction projection (sentence embedding -> hidden)
        self.instr_proj = nn.Sequential(
            nn.Linear(instr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # State-action encoder
        self.state_action_enc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if use_attention:
            # Cross-attention between instruction and state-action
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            
        # Reward prediction head
        combined_dim = hidden_dim * 2 if not use_attention else hidden_dim
        layers = []
        input_dim = combined_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.reward_head = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute trajectory reward.
        
        Args:
            obs: Observations (B, T, obs_dim)
            act: Actions (B, T, act_dim)
            instr_emb: Pre-computed instruction embeddings (B, instr_dim)
            mask: Optional mask for variable-length trajectories (B, T)
            
        Returns:
            Total trajectory rewards (B,)
        """
        B, T, _ = obs.shape
        
        # Encode instruction
        instr_h = self.instr_proj(instr_emb)  # (B, hidden_dim)
        
        # Encode state-action pairs
        sa = torch.cat([obs, act], dim=-1)  # (B, T, obs_dim + act_dim)
        sa_h = self.state_action_enc(sa)  # (B, T, hidden_dim)
        
        if self.use_attention:
            # Cross-attention: instruction attends to state-action sequence
            instr_h_expanded = instr_h.unsqueeze(1)  # (B, 1, hidden_dim)
            
            attn_out, _ = self.attention(
                query=instr_h_expanded,
                key=sa_h,
                value=sa_h,
                key_padding_mask=~mask if mask is not None else None,
            )  # (B, 1, hidden_dim)
            
            combined = attn_out.squeeze(1)  # (B, hidden_dim)
            
            # Predict single reward for whole trajectory
            reward = self.reward_head(combined).squeeze(-1)  # (B,)
            
        else:
            # Simple concatenation approach
            instr_h_expanded = instr_h.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            combined = torch.cat([sa_h, instr_h_expanded], dim=-1)  # (B, T, 2*hidden_dim)
            
            # Per-step rewards
            step_rewards = self.reward_head(combined).squeeze(-1)  # (B, T)
            
            # Apply mask and sum
            if mask is not None:
                step_rewards = step_rewards * mask.float()
                
            reward = step_rewards.sum(dim=1)  # (B,)
            
        return reward
        
    def forward_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute single-step reward.
        
        Args:
            obs: Single observation (B, 1, obs_dim) or (B, obs_dim)
            act: Single action (B, 1, act_dim) or (B, act_dim)
            instr_emb: Instruction embeddings (B, instr_dim)
            
        Returns:
            Step rewards (B,)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            act = act.unsqueeze(1)
            
        return self.forward(obs, act, instr_emb)


def compute_preference_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
    loss_type: str = "cross_entropy",
) -> torch.Tensor:
    """
    Compute preference learning loss.
    
    Args:
        reward_a: Predicted rewards for trajectory A (B,)
        reward_b: Predicted rewards for trajectory B (B,)
        preference: Labels (B,) - 0 if A preferred, 1 if B preferred
        loss_type: 'cross_entropy', 'hinge', or 'bce'
        
    Returns:
        Scalar loss
    """
    if loss_type == "cross_entropy":
        # Bradley-Terry model with cross-entropy
        logits = torch.stack([reward_a, reward_b], dim=1)  # (B, 2)
        return F.cross_entropy(logits, preference)
        
    elif loss_type == "hinge":
        # Hinge loss: r_preferred > r_other + margin
        r_preferred = torch.where(preference == 0, reward_a, reward_b)
        r_other = torch.where(preference == 0, reward_b, reward_a)
        margin = 1.0
        return F.relu(margin - (r_preferred - r_other)).mean()
        
    elif loss_type == "bce":
        # Binary cross-entropy on probability
        logits = reward_b - reward_a  # Positive if B preferred
        return F.binary_cross_entropy_with_logits(
            logits,
            preference.float(),
        )
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_accuracy(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
) -> float:
    """
    Compute preference prediction accuracy.
    
    Args:
        reward_a: Predicted rewards for trajectory A
        reward_b: Predicted rewards for trajectory B
        preference: Ground truth preference labels
        
    Returns:
        Accuracy as float
    """
    predicted = (reward_b > reward_a).long()
    correct = (predicted == preference).float().sum()
    return (correct / len(preference)).item()
