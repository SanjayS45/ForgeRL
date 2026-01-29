import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def preference_cross_entropy_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
) -> torch.Tensor:
    logits = reward_a - reward_b
    loss = F.binary_cross_entropy_with_logits(logits, preference)
    return loss

def preference_hinge_loss(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    sign = 2 * preference - 1
    loss = F.relu(margin - sign * (reward_a - reward_b))
    return loss.mean()

def compute_accuracy(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
) -> float:
    predicted = (reward_a > reward_b).float()
    correct = (predicted == preference).float()
    return correct.mean().item()

def compute_preference_metrics(
    reward_a: torch.Tensor,
    reward_b: torch.Tensor,
    preference: torch.Tensor,
) -> dict:
    with torch.no_grad():
        logits = reward_a - reward_b
        probs = torch.sigmoid(logits)
        
        predicted = (logits > 0).float()
        accuracy = (predicted == preference).float().mean().item()
        
        confidence = torch.abs(probs - 0.5).mean().item() * 2
        
        ce_loss = F.binary_cross_entropy_with_logits(logits, preference).item()
    
    return {
        "accuracy": accuracy,
        "confidence": confidence,
        "ce_loss": ce_loss,
        "mean_reward_a": reward_a.mean().item(),
        "mean_reward_b": reward_b.mean().item(),
    }

class PreferenceLoss(nn.Module):
    def __init__(self, loss_type: str = "cross_entropy", margin: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
    
    def forward(
        self,
        reward_a: torch.Tensor,
        reward_b: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return preference_cross_entropy_loss(reward_a, reward_b, preference)
        elif self.loss_type == "hinge":
            return preference_hinge_loss(reward_a, reward_b, preference, self.margin)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

