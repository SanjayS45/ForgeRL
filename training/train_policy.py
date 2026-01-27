#!/usr/bin/env python3
"""
Train RL policy using learned reward model.

Usage:
    python training/train_policy.py --task reach-v3
    python training/train_policy.py --reward-model checkpoints/reward_model/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random

from envs.metaworld_wrapper import make_metaworld_env
from envs.instructions import sample_instruction
from policy.ppo import PPOAgent, PPOTrainer
from reward_model.model import RewardModelWithSentenceEncoder
from utils.instruction_encoder import encode_instruction


def load_reward_model(
    checkpoint_path: str,
    obs_dim: int = 39,
    act_dim: int = 4,
    device: str = "cpu",
) -> RewardModelWithSentenceEncoder:
    """Load trained reward model."""
    model = RewardModelWithSentenceEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        instr_dim=384,
        hidden_dim=256,
        num_layers=3,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded reward model from {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL policy with reward model")
    
    parser.add_argument(
        "--task", type=str, default="reach-v3",
        help="MetaWorld task name"
    )
    parser.add_argument(
        "--reward-model", type=str, default=None,
        help="Path to trained reward model (if None, uses env reward)"
    )
    parser.add_argument(
        "--instruction", type=str, default=None,
        help="Task instruction (random if not specified)"
    )
    parser.add_argument(
        "--total-steps", type=int, default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--steps-per-update", type=int, default=2048,
        help="Steps between policy updates"
    )
    parser.add_argument(
        "--epochs-per-update", type=int, default=10,
        help="PPO epochs per update"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda"
    )
    parser.add_argument(
        "--clip-ratio", type=float, default=0.2,
        help="PPO clip ratio"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="Entropy coefficient"
    )
    parser.add_argument(
        "--use-env-reward", action="store_true",
        help="Combine learned reward with env reward"
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1.0,
        help="Scale factor for learned rewards"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/policy",
        help="Log directory"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10,
        help="Checkpoint frequency (updates)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cpu, cuda)"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create environment
    print(f"\nCreating environment: {args.task}")
    env = make_metaworld_env(task_name=args.task, seed=args.seed)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    
    # Load reward model if specified
    reward_model = None
    instruction = args.instruction
    
    if args.reward_model:
        reward_model = load_reward_model(
            args.reward_model,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )
        
        if instruction is None:
            instruction = sample_instruction(args.task)
        print(f"Instruction: {instruction}")
        
    # Create agent
    print("\nCreating PPO agent...")
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        device=device,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
    )
    
    total_params = sum(p.numel() for p in agent.actor_critic.parameters())
    print(f"Policy parameters: {total_params:,}")
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        reward_model=reward_model,
        instruction=instruction,
        instruction_encoder=encode_instruction,
        use_env_reward=args.use_env_reward or reward_model is None,
        reward_scale=args.reward_scale,
        log_dir=args.log_dir,
    )
    
    # Train
    print(f"\nStarting training for {args.total_steps:,} timesteps...")
    print(f"Updates: {args.total_steps // args.steps_per_update}")
    
    history = trainer.train(
        total_timesteps=args.total_steps,
        steps_per_update=args.steps_per_update,
        epochs_per_update=args.epochs_per_update,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        checkpoint_freq=args.checkpoint_freq,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    
    if history["episode_reward"]:
        final_reward = np.mean(history["episode_reward"][-10:])
        best_reward = max(history["episode_reward"])
        print(f"Final avg reward: {final_reward:.2f}")
        print(f"Best avg reward: {best_reward:.2f}")
        
    print(f"\nSaved to {args.log_dir}/")
    
    env.close()


if __name__ == "__main__":
    main()

