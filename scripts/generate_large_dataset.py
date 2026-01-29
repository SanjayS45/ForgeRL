import pickle
import numpy as np
import random
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.metaworld_wrapper import make_metaworld_env

INSTRUCTIONS = [
    "reach the target",
    "move to the goal position",
    "extend arm to target",
    "navigate to the red marker",
    "position gripper at goal",
    "touch the target point",
    "reach toward the objective",
    "move end effector to destination",
    "approach the target location",
    "guide arm to goal",
]

def generate_random_goal():
    x = np.random.uniform(-0.3, 0.3)
    y = np.random.uniform(-0.3, 0.3)
    z = np.random.uniform(0.02, 0.3)
    return [x, y, z]

def rollout(env, goal, noise=0.1, horizon=50):
    if hasattr(env, 'set_custom_goal'):
        env.set_custom_goal(goal)
    
    obs, _ = env.reset()
    trajectory = []
    
    for step in range(horizon):
        goal_pos = np.array(goal)
        ee_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        
        direction = goal_pos - ee_pos
        dist = np.linalg.norm(direction)
        direction = direction / (dist + 1e-8)
        
        speed = min(0.8, dist * 2)
        action = np.zeros(env.action_space.shape[0])
        action[:3] = direction * speed
        
        action = action + np.random.randn(*action.shape) * noise
        action = np.clip(action, -1, 1)
        
        trajectory.append((obs.copy(), action.copy()))
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
    
    return trajectory

def compute_trajectory_score(trajectory, goal):
    if len(trajectory) == 0:
        return float('inf')
    
    final_obs = trajectory[-1][0]
    ee_pos = final_obs[:3]
    goal_pos = np.array(goal)
    
    final_dist = np.linalg.norm(ee_pos - goal_pos)
    
    total_action_magnitude = sum(np.linalg.norm(t[1]) for t in trajectory)
    smoothness = total_action_magnitude / len(trajectory)
    
    return final_dist + 0.1 * smoothness

def generate_preference_pair(env, instruction, horizon=50):
    goal = generate_random_goal()
    
    traj_good = rollout(env, goal, noise=0.05, horizon=horizon)
    traj_bad = rollout(env, goal, noise=0.3, horizon=horizon)
    
    score_good = compute_trajectory_score(traj_good, goal)
    score_bad = compute_trajectory_score(traj_bad, goal)
    
    if score_good < score_bad:
        return (instruction, traj_good, traj_bad, goal)
    else:
        return (instruction, traj_bad, traj_good, goal)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="reach-v3")
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiments/large_dataset.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_pairs} preference pairs for task '{args.task}'...")
    
    env = make_metaworld_env(args.task, seed=args.seed)
    
    data = []
    goals_used = []
    
    for i in range(args.num_pairs):
        instruction = random.choice(INSTRUCTIONS)
        pair = generate_preference_pair(env, instruction, horizon=args.horizon)
        
        data.append((pair[0], pair[1], pair[2]))
        goals_used.append(pair[3])
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{args.num_pairs} pairs...")
    
    env.close()
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    goals_path = output_path.with_suffix('.goals.pkl')
    with open(goals_path, "wb") as f:
        pickle.dump(goals_used, f)
    
    print(f"Saved {len(data)} pairs to {output_path}")
    print(f"Saved {len(goals_used)} goals to {goals_path}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {len(data)}")
    print(f"  Unique instructions: {len(set(d[0] for d in data))}")
    print(f"  Avg trajectory length: {np.mean([len(d[1]) for d in data]):.1f}")

if __name__ == "__main__":
    main()

