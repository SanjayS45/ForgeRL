#!/usr/bin/env python3
"""
Generate trajectory preference dataset from MetaWorld environments.

This script generates paired trajectories with synthetic preferences
for training reward models. It properly handles MetaWorld V3 environment
instantiation (using train_classes, not train_tasks).

Usage:
    python scripts/generate_trajectories.py --num-pairs 1000 --task reach-v3
    python scripts/generate_trajectories.py --config experiments/config.yaml
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.instructions import get_instructions_for_task, INSTRUCTIONS


def create_metaworld_env(task_name: str = "reach-v3", seed: int = 42):
    """
    Properly create a MetaWorld environment.
    
    IMPORTANT: MetaWorld V3 API requires:
    1. Get the environment CLASS from ml1.train_classes
    2. Instantiate the class to get an actual env
    3. Set a task from ml1.train_tasks
    
    The train_tasks list contains Task objects (goal configurations),
    NOT usable environments!
    """
    import metaworld
    
    # Create ML1 benchmark
    ml1 = metaworld.ML1(task_name, seed=seed)
    
    # Get the environment CLASS (not instance!)
    env_cls = ml1.train_classes[task_name]
    
    # Instantiate the environment
    env = env_cls()
    
    # Set a task (goal configuration)
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    return env, ml1


def rollout(
    env,
    horizon: int = 50,
    noise: float = 0.0,
    policy: Optional[callable] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a single trajectory.
    
    Args:
        env: MetaWorld environment
        horizon: Number of timesteps
        noise: Gaussian noise std for actions
        policy: Optional policy function (obs -> action)
        
    Returns:
        List of (observation, action) tuples
    """
    trajectory = []
    
    # Reset environment
    result = env.reset()
    if isinstance(result, tuple):
        obs = result[0]
    else:
        obs = result
        
    for t in range(horizon):
        # Get action
        if policy is not None:
            action = policy(obs)
        else:
            # Random action
            action = env.action_space.sample()
            
        # Add noise
        if noise > 0:
            action = action + np.random.randn(*action.shape) * noise
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
        # Store transition
        trajectory.append((obs.copy(), action.copy()))
        
        # Step environment
        result = env.step(action)
        
        # Handle both 4-tuple and 5-tuple returns
        if len(result) == 5:
            obs, _, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            obs, _, done, _ = result
            
        if done:
            result = env.reset()
            if isinstance(result, tuple):
                obs = result[0]
            else:
                obs = result
                
    return trajectory


def simple_reaching_policy(obs: np.ndarray) -> np.ndarray:
    """
    Simple heuristic policy for reaching tasks.
    
    Moves gripper toward the goal position.
    """
    # MetaWorld reach observation structure:
    # - gripper position: obs[0:3]
    # - gripper velocity: obs[3:6]  
    # - goal position: obs[-3:]
    
    gripper_pos = obs[0:3]
    goal_pos = obs[-3:]
    
    # Simple proportional control
    direction = goal_pos - gripper_pos
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Action space: [x, y, z, gripper]
    action = np.zeros(4)
    action[0:3] = direction * 0.5
    action[3] = 0.0  # Keep gripper open
    
    return action


def generate_preference_pairs(
    env,
    instructions: List[str],
    num_pairs_per_instruction: int = 100,
    horizon: int = 50,
    good_noise: float = 0.05,
    bad_noise: float = 0.3,
    use_policy: bool = True,
) -> List[Tuple[str, List, List]]:
    """
    Generate preference pairs for multiple instructions.
    
    Args:
        env: MetaWorld environment
        instructions: List of instruction strings
        num_pairs_per_instruction: Pairs per instruction
        horizon: Trajectory length
        good_noise: Noise level for "good" trajectories
        bad_noise: Noise level for "bad" trajectories
        use_policy: Use heuristic policy (True) or random actions (False)
        
    Returns:
        List of (instruction, good_trajectory, bad_trajectory) tuples
    """
    data = []
    
    policy = simple_reaching_policy if use_policy else None
    
    for instr in tqdm(instructions, desc="Instructions"):
        for _ in tqdm(range(num_pairs_per_instruction), desc=f"  Pairs", leave=False):
            # Generate "good" trajectory (low noise)
            traj_good = rollout(env, horizon=horizon, noise=good_noise, policy=policy)
            
            # Generate "bad" trajectory (high noise)
            traj_bad = rollout(env, horizon=horizon, noise=bad_noise, policy=policy)
            
            # Randomly order to avoid position bias
            if random.random() > 0.5:
                data.append((instr, traj_good, traj_bad))
            else:
                data.append((instr, traj_bad, traj_good))
                
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory preference dataset")
    
    parser.add_argument(
        "--task", type=str, default="reach-v3",
        help="MetaWorld task name (default: reach-v3)"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=1000,
        help="Total number of preference pairs to generate"
    )
    parser.add_argument(
        "--horizon", type=int, default=50,
        help="Trajectory length in timesteps"
    )
    parser.add_argument(
        "--good-noise", type=float, default=0.05,
        help="Noise level for good trajectories"
    )
    parser.add_argument(
        "--bad-noise", type=float, default=0.3,
        help="Noise level for bad trajectories"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/trajs.pkl",
        help="Output file path"
    )
    parser.add_argument(
        "--use-policy", action="store_true", default=True,
        help="Use heuristic policy (default: True)"
    )
    parser.add_argument(
        "--random-only", action="store_true",
        help="Use only random actions (overrides --use-policy)"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating dataset for task: {args.task}")
    print(f"Target pairs: {args.num_pairs}")
    
    # Create environment
    print("Creating MetaWorld environment...")
    env, ml1 = create_metaworld_env(args.task, seed=args.seed)
    
    # Get instructions
    instructions = get_instructions_for_task(args.task)
    print(f"Using {len(instructions)} instruction variants")
    
    # Calculate pairs per instruction
    pairs_per_instr = max(1, args.num_pairs // len(instructions))
    actual_total = pairs_per_instr * len(instructions)
    print(f"Generating {pairs_per_instr} pairs per instruction ({actual_total} total)")
    
    # Generate data
    use_policy = args.use_policy and not args.random_only
    data = generate_preference_pairs(
        env=env,
        instructions=instructions,
        num_pairs_per_instruction=pairs_per_instr,
        horizon=args.horizon,
        good_noise=args.good_noise,
        bad_noise=args.bad_noise,
        use_policy=use_policy,
    )
    
    print(f"\nGenerated {len(data)} preference pairs")
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Saved dataset to {output_path}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"  Total pairs: {len(data)}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  Trajectory length: {args.horizon}")
    
    # Sample trajectory info
    sample = data[0]
    print(f"  Observation dim: {len(sample[1][0][0])}")
    print(f"  Action dim: {len(sample[1][0][1])}")
    
    env.close()
    

if __name__ == "__main__":
    main()
