#!/usr/bin/env python3

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.instructions import get_instructions_for_task, INSTRUCTIONS

def create_metaworld_env(task_name: str = "reach-v3", seed: int = 42):
    
    import metaworld
    

    ml1 = metaworld.ML1(task_name, seed=seed)
    

    env_cls = ml1.train_classes[task_name]
    

    env = env_cls()
    

    task = ml1.train_tasks[0]
    env.set_task(task)
    
    return env, ml1

def rollout(
    env,
    horizon: int = 50,
    noise: float = 0.0,
    policy: Optional[callable] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    
    trajectory = []
    

    result = env.reset()
    if isinstance(result, tuple):
        obs = result[0]
    else:
        obs = result
        
    for t in range(horizon):

        if policy is not None:
            action = policy(obs)
        else:

            action = env.action_space.sample()
            

        if noise > 0:
            action = action + np.random.randn(*action.shape) * noise
            action = np.clip(action, env.action_space.low, env.action_space.high)
            

        trajectory.append((obs.copy(), action.copy()))
        

        result = env.step(action)
        

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
    

    
    gripper_pos = obs[0:3]
    goal_pos = obs[-3:]
    

    direction = goal_pos - gripper_pos
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    

    action = np.zeros(4)
    action[0:3] = direction * 0.5
    action[3] = 0.0
    
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
    
    data = []
    
    policy = simple_reaching_policy if use_policy else None
    
    for instr in tqdm(instructions, desc="Instructions"):
        for _ in tqdm(range(num_pairs_per_instruction), desc=f"  Pairs", leave=False):

            traj_good = rollout(env, horizon=horizon, noise=good_noise, policy=policy)
            

            traj_bad = rollout(env, horizon=horizon, noise=bad_noise, policy=policy)
            

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
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating dataset for task: {args.task}")
    print(f"Target pairs: {args.num_pairs}")
    

    print("Creating MetaWorld environment...")
    env, ml1 = create_metaworld_env(args.task, seed=args.seed)
    

    instructions = get_instructions_for_task(args.task)
    print(f"Using {len(instructions)} instruction variants")
    

    pairs_per_instr = max(1, args.num_pairs // len(instructions))
    actual_total = pairs_per_instr * len(instructions)
    print(f"Generating {pairs_per_instr} pairs per instruction ({actual_total} total)")
    

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
    

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Saved dataset to {output_path}")
    

    print("\nDataset Summary:")
    print(f"  Total pairs: {len(data)}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  Trajectory length: {args.horizon}")
    

    sample = data[0]
    print(f"  Observation dim: {len(sample[1][0][0])}")
    print(f"  Action dim: {len(sample[1][0][1])}")
    
    env.close()
    

if __name__ == "__main__":
    main()
