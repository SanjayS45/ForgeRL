#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random

from utils.data_utils import load_dataset, split_dataset, compute_stats
from utils.instruction_encoder import encode_instruction
from reward_model.model import RewardModelWithSentenceEncoder
from reward_model.trainer import RewardModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    
    parser.add_argument(
        "--data", type=str, default="experiments/trajs.pkl",
        help="Path to preference dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3,
        help="Number of MLP layers"
    )
    parser.add_argument(
        "--max-traj-length", type=int, default=100,
        help="Maximum trajectory length"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/reward_model",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--loss-type", type=str, default="cross_entropy",
        choices=["cross_entropy", "hinge", "bce"],
        help="Loss function type"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Training split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split ratio"
    )
    
    args = parser.parse_args()
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    

    print(f"\nLoading dataset from {args.data}...")
    data = load_dataset(args.data)
    
    stats = compute_stats(data)
    print(f"\n{stats}")
    

    if stats.num_pairs < 100:
        print(f"\nWARNING: Only {stats.num_pairs} pairs. Recommend at least 1000 for good results.")
        

    test_ratio = 1.0 - args.train_split - args.val_split
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        test_ratio=test_ratio,
        seed=args.seed,
    )
    print(f"\nSplit: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    

    print("\nCreating reward model...")
    model = RewardModelWithSentenceEncoder(
        obs_dim=stats.obs_dim,
        act_dim=stats.act_dim,
        instr_dim=384,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_attention=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    

    trainer = RewardModelTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        loss_type=args.loss_type,
        use_sentence_encoder=True,
    )
    

    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        instruction_encoder=encode_instruction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_trajectory_length=args.max_traj_length,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.checkpoint_dir,
    )
    

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    
    if history["val_accuracy"]:
        best_val_acc = max(history["val_accuracy"])
        best_epoch = history["val_accuracy"].index(best_val_acc)
        print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
        
    final_train_acc = history["train_accuracy"][-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    

    import json
    history_path = Path(args.checkpoint_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    print(f"Checkpoints saved to {args.checkpoint_dir}/")

if __name__ == "__main__":
    main()
