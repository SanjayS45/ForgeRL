import argparse
import torch
import numpy as np
from pathlib import Path
import json

from envs.metaworld_wrapper import make_metaworld_env
from policy.ppo import PPOAgent
from policy.rollout import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--task", type=str, default="reach-v3", help="MetaWorld task")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = make_metaworld_env(args.task, seed=args.seed)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(args.checkpoint)
    
    def policy_fn(obs, deterministic=True):
        action, _, _ = agent.get_action(obs, deterministic=deterministic)
        return action
    
    print(f"Evaluating policy on {args.task} for {args.num_episodes} episodes...")
    
    results = evaluate_policy(
        env=env,
        policy=policy_fn,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )
    
    print("\n=== Evaluation Results ===")
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Mean Episode Length: {results['mean_length']:.1f}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    env.close()

if __name__ == "__main__":
    main()

