import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class RewardModel(nn.Module):
    
    
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
        

        self.instr_emb = nn.Embedding(vocab_size, instr_dim)
        

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
        

        self._init_weights()
        
    def _init_weights(self):
        
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
        
        B, T, _ = obs.shape
        

        instr_e = self.instr_emb(instr)
        instr_e = instr_e.unsqueeze(1).expand(-1, T, -1)
        

        x = torch.cat([obs, act, instr_e], dim=-1)
        

        r = self.net(x).squeeze(-1)
        

        return r.sum(dim=1)
        
    def forward_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr: torch.Tensor,
    ) -> torch.Tensor:
        

        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            act = act.unsqueeze(1)
            
        return self.forward(obs, act, instr)

class RewardModelWithSentenceEncoder(nn.Module):
    
    
    def __init__(
        self,
        obs_dim: int = 39,
        act_dim: int = 4,
        instr_dim: int = 384,
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
        

        self.instr_proj = nn.Sequential(
            nn.Linear(instr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        

        self.state_action_enc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if use_attention:

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            

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
        
        B, T, _ = obs.shape
        

        instr_h = self.instr_proj(instr_emb)
        

        sa = torch.cat([obs, act], dim=-1)
        sa_h = self.state_action_enc(sa)
        
        if self.use_attention:

            instr_h_expanded = instr_h.unsqueeze(1)
            
            attn_out, _ = self.attention(
                query=instr_h_expanded,
                key=sa_h,
                value=sa_h,
                key_padding_mask=~mask if mask is not None else None,
            )
            
            combined = attn_out.squeeze(1)
            

            reward = self.reward_head(combined).squeeze(-1)
            
        else:

            instr_h_expanded = instr_h.unsqueeze(1).expand(-1, T, -1)
            combined = torch.cat([sa_h, instr_h_expanded], dim=-1)
            

            step_rewards = self.reward_head(combined).squeeze(-1)
            

            if mask is not None:
                step_rewards = step_rewards * mask.float()
                
            reward = step_rewards.sum(dim=1)
            
        return reward
        
    def forward_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        instr_emb: torch.Tensor,
    ) -> torch.Tensor:
        
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
    
    if loss_type == "cross_entropy":

        logits = torch.stack([reward_a, reward_b], dim=1)
        return F.cross_entropy(logits, preference)
        
    elif loss_type == "hinge":

        r_preferred = torch.where(preference == 0, reward_a, reward_b)
        r_other = torch.where(preference == 0, reward_b, reward_a)
        margin = 1.0
        return F.relu(margin - (r_preferred - r_other)).mean()
        
    elif loss_type == "bce":

        logits = reward_b - reward_a
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
    
    predicted = (reward_b > reward_a).long()
    correct = (predicted == preference).float().sum()
    return (correct / len(preference)).item()
