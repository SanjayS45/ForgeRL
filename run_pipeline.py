#!/usr/bin/env python3
"""
Run the complete RLHF pipeline.

This script orchestrates the full pipeline:
1. Generate/load dataset
2. Train reward model
3. Train policy with learned reward

Usage:
    python run_pipeline.py --task reach-v3 --generate-data
    python run_pipeline.py --dataset experiments/trajs.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import random
import torch


def main():
    parser = argparse.ArgumentParser(description="Run complete RLHF pipeline")
    
    # Data options
    parser.add_argument("--task", type=str, default="reach-v3", help="MetaWorld task")
    parser.add_argument("--generate-data", action="store_true", help="Generate new dataset")
    parser.add_argument("--dataset", type=str, default="experiments/trajs.pkl", help="Dataset path")
    parser.add_argument("--num-pairs", type=int, default=1000, help="Number of preference pairs")
    
    # Reward model options
    parser.add_argument("--rm-epochs", type=int, default=50, help="Reward model epochs")
    parser.add_argument("--rm-batch-size", type=int, default=32, help="Reward model batch size")
    parser.add_argument("--rm-lr", type=float, default=1e-4, help="Reward model learning rate")
    
    # Policy options
    parser.add_argument("--policy-steps", type=int, default=100000, help="Policy training steps")
    parser.add_argument("--instruction", type=str, default="reach the target", help="Task instruction")
    parser.add_argument("--use-env-reward", action="store_true", help="Combine with env reward")
    
    # General options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--skip-reward", action="store_true", help="Skip reward model training")
    parser.add_argument("--skip-policy", action="store_true", help="Skip policy training")
    
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
    
    dataset_path = Path(args.dataset)
    
    # Step 1: Generate or validate dataset
    print("\n" + "="*60)
    print("Step 1: Dataset Preparation")
    print("="*60)
    
    if args.generate_data or not dataset_path.exists():
        print(f"\nGenerating dataset with {args.num_pairs} pairs...")
        
        from scripts.generate_trajectories import (
            create_metaworld_env,
            generate_preference_pairs,
        )
        from envs.instructions import get_instructions_for_task
        from utils.data_utils import save_dataset, compute_stats
        
        env, ml1 = create_metaworld_env(args.task, seed=args.seed)
        instructions = get_instructions_for_task(args.task)
        pairs_per_instr = max(1, args.num_pairs // len(instructions))
        
        data = generate_preference_pairs(
            env=env,
            instructions=instructions,
            num_pairs_per_instruction=pairs_per_instr,
            use_policy=True,
        )
        
        save_dataset(data, dataset_path)
        env.close()
        
        stats = compute_stats(data)
        print(f"\n{stats}")
    else:
        print(f"\nLoading existing dataset: {dataset_path}")
        from utils.data_utils import load_dataset, compute_stats
        data = load_dataset(dataset_path)
        stats = compute_stats(data)
        print(f"\n{stats}")
        
    # Step 2: Train reward model
    if not args.skip_reward:
        print("\n" + "="*60)
        print("Step 2: Reward Model Training")
        print("="*60)
        
        from utils.data_utils import load_dataset, split_dataset
        from utils.instruction_encoder import encode_instruction
        from reward_model.model import RewardModelWithSentenceEncoder
        from reward_model.trainer import RewardModelTrainer
        
        data = load_dataset(dataset_path)
        train_data, val_data, _ = split_dataset(data, 0.8, 0.1, 0.1, seed=args.seed)
        
        sample = data[0]
        obs_dim = len(sample[1][0][0])
        act_dim = len(sample[1][0][1])
        
        model = RewardModelWithSentenceEncoder(
            obs_dim=obs_dim,
            act_dim=act_dim,
            instr_dim=384,
            hidden_dim=256,
        ).to(device)
        
        trainer = RewardModelTrainer(
            model=model,
            device=device,
            learning_rate=args.rm_lr,
        )
        
        checkpoint_dir = f"checkpoints/reward_model/{args.task}"
        
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            instruction_encoder=encode_instruction,
            epochs=args.rm_epochs,
            batch_size=args.rm_batch_size,
            checkpoint_dir=checkpoint_dir,
        )
        
        if history["val_accuracy"]:
            best_acc = max(history["val_accuracy"])
            print(f"\nBest validation accuracy: {best_acc:.4f}")
            
        reward_model_path = f"{checkpoint_dir}/best_model.pt"
    else:
        # Try to find existing reward model
        reward_model_path = f"checkpoints/reward_model/{args.task}/best_model.pt"
        if not Path(reward_model_path).exists():
            reward_model_path = None
            print("\nSkipping reward model (no trained model found)")
            
    # Step 3: Train policy
    if not args.skip_policy:
        print("\n" + "="*60)
        print("Step 3: Policy Training")
        print("="*60)
        
        from envs.metaworld_wrapper import make_metaworld_env
        from policy.ppo import PPOAgent, PPOTrainer
        from reward_model.model import RewardModelWithSentenceEncoder
        from utils.instruction_encoder import encode_instruction
        
        env = make_metaworld_env(task_name=args.task, seed=args.seed)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # Load reward model if available
        reward_model = None
        if reward_model_path and Path(reward_model_path).exists():
            print(f"\nLoading reward model: {reward_model_path}")
            reward_model = RewardModelWithSentenceEncoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
            ).to(device)
            checkpoint = torch.load(reward_model_path, map_location=device)
            reward_model.load_state_dict(checkpoint["model_state_dict"])
            reward_model.eval()
        else:
            print("\nNo reward model - using environment reward")
            
        agent = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )
        
        log_dir = f"logs/policy/{args.task}"
        
        trainer = PPOTrainer(
            env=env,
            agent=agent,
            reward_model=reward_model,
            instruction=args.instruction,
            instruction_encoder=encode_instruction,
            use_env_reward=args.use_env_reward or reward_model is None,
            log_dir=log_dir,
        )
        
        history = trainer.train(
            total_timesteps=args.policy_steps,
        )
        
        if history["episode_reward"]:
            final_reward = np.mean(history["episode_reward"][-10:])
            print(f"\nFinal average reward: {final_reward:.2f}")
            
        env.close()
        
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"\nDataset: {dataset_path}")
    if not args.skip_reward:
        print(f"Reward Model: checkpoints/reward_model/{args.task}/best_model.pt")
    if not args.skip_policy:
        print(f"Policy: logs/policy/{args.task}/final_model.pt")
        

if __name__ == "__main__":
    main()

