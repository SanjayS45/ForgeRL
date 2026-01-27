import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from pathlib import Path
from tqdm import tqdm

from reward_model.model import (
    RewardModel,
    RewardModelWithSentenceEncoder,
    compute_preference_loss,
    compute_accuracy,
)

class PreferenceDataset(Dataset):
    
    
    def __init__(
        self,
        data: List[Tuple],
        instruction_encoder: Callable,
        max_length: Optional[int] = None,
        use_sentence_encoder: bool = True,
    ):
        
        self.data = data
        self.instruction_encoder = instruction_encoder
        self.max_length = max_length
        self.use_sentence_encoder = use_sentence_encoder
        

        self.instructions = [d[0] for d in data]
        self.instr_encodings = self._encode_instructions()
        
    def _encode_instructions(self):
        
        if self.use_sentence_encoder:

            return self.instruction_encoder(self.instructions)
        else:

            return [self.instruction_encoder(instr) for instr in self.instructions]
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        instr, traj_a, traj_b = self.data[idx]
        

        obs_a, act_a, len_a = self._process_trajectory(traj_a)
        obs_b, act_b, len_b = self._process_trajectory(traj_b)
        

        if self.use_sentence_encoder:
            instr_enc = torch.tensor(self.instr_encodings[idx], dtype=torch.float32)
        else:
            instr_enc = torch.tensor(self.instr_encodings[idx], dtype=torch.long)
            

        preference = self._compute_preference(traj_a, traj_b)
        
        return {
            "obs_a": obs_a,
            "act_a": act_a,
            "len_a": len_a,
            "obs_b": obs_b,
            "act_b": act_b,
            "len_b": len_b,
            "instruction": instr_enc,
            "preference": preference,
        }
        
    def _process_trajectory(self, traj):
        
        obs = np.array([t[0] for t in traj], dtype=np.float32)
        act = np.array([t[1] for t in traj], dtype=np.float32)
        length = len(traj)
        

        if self.max_length is not None:
            if length > self.max_length:
                obs = obs[:self.max_length]
                act = act[:self.max_length]
                length = self.max_length
            elif length < self.max_length:
                pad_obs = np.zeros((self.max_length - length, obs.shape[1]), dtype=np.float32)
                pad_act = np.zeros((self.max_length - length, act.shape[1]), dtype=np.float32)
                obs = np.concatenate([obs, pad_obs], axis=0)
                act = np.concatenate([act, pad_act], axis=0)
                
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(act, dtype=torch.float32),
            length,
        )
        
    def _compute_preference(self, traj_a, traj_b) -> int:
        
        from preferences.synthetic import preference
        return preference(traj_a, traj_b)

def collate_fn(batch):
    

    max_len = max(
        max(b["len_a"] for b in batch),
        max(b["len_b"] for b in batch),
    )
    

    obs_a = []
    act_a = []
    mask_a = []
    obs_b = []
    act_b = []
    mask_b = []
    instructions = []
    preferences = []
    
    for b in batch:

        len_a = b["obs_a"].shape[0]
        if len_a < max_len:
            pad_a = torch.zeros(max_len - len_a, b["obs_a"].shape[1])
            obs_a.append(torch.cat([b["obs_a"], pad_a], dim=0))
            pad_act = torch.zeros(max_len - len_a, b["act_a"].shape[1])
            act_a.append(torch.cat([b["act_a"], pad_act], dim=0))
        else:
            obs_a.append(b["obs_a"][:max_len])
            act_a.append(b["act_a"][:max_len])
        mask_a.append(torch.arange(max_len) < b["len_a"])
        

        len_b = b["obs_b"].shape[0]
        if len_b < max_len:
            pad_b = torch.zeros(max_len - len_b, b["obs_b"].shape[1])
            obs_b.append(torch.cat([b["obs_b"], pad_b], dim=0))
            pad_act = torch.zeros(max_len - len_b, b["act_b"].shape[1])
            act_b.append(torch.cat([b["act_b"], pad_act], dim=0))
        else:
            obs_b.append(b["obs_b"][:max_len])
            act_b.append(b["act_b"][:max_len])
        mask_b.append(torch.arange(max_len) < b["len_b"])
        
        instructions.append(b["instruction"])
        preferences.append(b["preference"])
        
    return {
        "obs_a": torch.stack(obs_a),
        "act_a": torch.stack(act_a),
        "mask_a": torch.stack(mask_a),
        "obs_b": torch.stack(obs_b),
        "act_b": torch.stack(act_b),
        "mask_b": torch.stack(mask_b),
        "instruction": torch.stack(instructions),
        "preference": torch.tensor(preferences, dtype=torch.long),
    }

class RewardModelTrainer:
    
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_type: str = "cross_entropy",
        use_sentence_encoder: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.model = model.to(device)
        self.loss_type = loss_type
        self.use_sentence_encoder = use_sentence_encoder
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.scheduler = None
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
    def train(
        self,
        train_data: List[Tuple],
        val_data: Optional[List[Tuple]] = None,
        instruction_encoder: Callable = None,
        epochs: int = 50,
        batch_size: int = 32,
        max_trajectory_length: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
    ) -> Dict:
        

        if instruction_encoder is None:
            from utils.instruction_encoder import encode_instruction
            instruction_encoder = encode_instruction
            

        train_dataset = PreferenceDataset(
            train_data,
            instruction_encoder,
            max_length=max_trajectory_length,
            use_sentence_encoder=self.use_sentence_encoder,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        val_loader = None
        if val_data:
            val_dataset = PreferenceDataset(
                val_data,
                instruction_encoder,
                max_length=max_trajectory_length,
                use_sentence_encoder=self.use_sentence_encoder,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
        

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        

        for epoch in range(epochs):

            train_metrics = self._train_epoch(train_loader, log_interval)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            

            if val_loader:
                val_metrics = self._validate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                

                if val_metrics["accuracy"] > self.best_accuracy:
                    self.best_accuracy = val_metrics["accuracy"]
                    self.patience_counter = 0
                    

                    self.save(checkpoint_path / "best_model.pt")
                else:
                    self.patience_counter += 1
                    
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
                
            self.scheduler.step()
            

        self.save(checkpoint_path / "final_model.pt")
        
        return history
        
    def _train_epoch(self, loader, log_interval) -> Dict:
        
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):

            obs_a = batch["obs_a"].to(self.device)
            act_a = batch["act_a"].to(self.device)
            mask_a = batch["mask_a"].to(self.device)
            obs_b = batch["obs_b"].to(self.device)
            act_b = batch["act_b"].to(self.device)
            mask_b = batch["mask_b"].to(self.device)
            instr = batch["instruction"].to(self.device)
            pref = batch["preference"].to(self.device)
            

            if self.use_sentence_encoder:
                reward_a = self.model(obs_a, act_a, instr, mask_a)
                reward_b = self.model(obs_b, act_b, instr, mask_b)
            else:
                reward_a = self.model(obs_a, act_a, instr)
                reward_b = self.model(obs_b, act_b, instr)
                

            loss = compute_preference_loss(
                reward_a, reward_b, pref, self.loss_type
            )
            

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            

            total_loss += loss.item() * len(pref)
            predicted = (reward_b > reward_a).long()
            total_correct += (predicted == pref).sum().item()
            total_samples += len(pref)
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    "loss": loss.item(),
                    "acc": total_correct / total_samples,
                })
                
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
        
    def _validate(self, loader) -> Dict:
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:

                obs_a = batch["obs_a"].to(self.device)
                act_a = batch["act_a"].to(self.device)
                mask_a = batch["mask_a"].to(self.device)
                obs_b = batch["obs_b"].to(self.device)
                act_b = batch["act_b"].to(self.device)
                mask_b = batch["mask_b"].to(self.device)
                instr = batch["instruction"].to(self.device)
                pref = batch["preference"].to(self.device)
                

                if self.use_sentence_encoder:
                    reward_a = self.model(obs_a, act_a, instr, mask_a)
                    reward_b = self.model(obs_b, act_b, instr, mask_b)
                else:
                    reward_a = self.model(obs_a, act_a, instr)
                    reward_b = self.model(obs_b, act_b, instr)
                    

                loss = compute_preference_loss(
                    reward_a, reward_b, pref, self.loss_type
                )
                

                total_loss += loss.item() * len(pref)
                predicted = (reward_b > reward_a).long()
                total_correct += (predicted == pref).sum().item()
                total_samples += len(pref)
                
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
        
    def save(self, path: str):
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
        }, path)
        print(f"Saved checkpoint to {path}")
        
    def load(self, path: str):
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        print(f"Loaded checkpoint from {path}")

