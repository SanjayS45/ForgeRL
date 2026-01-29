import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_dataset(data):
    print(f"\n{'='*50}")
    print("DATASET ANALYSIS")
    print(f"{'='*50}\n")
    
    print(f"Total preference pairs: {len(data)}")
    
    instructions = [d[0] for d in data]
    unique_instructions = list(set(instructions))
    print(f"Unique instructions: {len(unique_instructions)}")
    
    print("\nInstruction distribution:")
    counter = Counter(instructions)
    for instr, count in counter.most_common(10):
        print(f"  - {instr[:50]}... : {count}")
    
    traj_lengths_a = [len(d[1]) for d in data]
    traj_lengths_b = [len(d[2]) for d in data]
    
    print(f"\nTrajectory A lengths:")
    print(f"  Mean: {np.mean(traj_lengths_a):.1f}")
    print(f"  Std: {np.std(traj_lengths_a):.1f}")
    print(f"  Min: {min(traj_lengths_a)}")
    print(f"  Max: {max(traj_lengths_a)}")
    
    print(f"\nTrajectory B lengths:")
    print(f"  Mean: {np.mean(traj_lengths_b):.1f}")
    print(f"  Std: {np.std(traj_lengths_b):.1f}")
    print(f"  Min: {min(traj_lengths_b)}")
    print(f"  Max: {max(traj_lengths_b)}")
    
    if data and data[0][1]:
        obs_dim = len(data[0][1][0][0])
        act_dim = len(data[0][1][0][1])
        print(f"\nObservation dimension: {obs_dim}")
        print(f"Action dimension: {act_dim}")
    
    final_distances_a = []
    final_distances_b = []
    
    for _, traj_a, traj_b in data:
        if traj_a:
            obs = traj_a[-1][0]
            ee_pos = obs[:3]
            goal_pos = obs[36:39] if len(obs) >= 39 else obs[-3:]
            final_distances_a.append(np.linalg.norm(ee_pos - goal_pos))
        
        if traj_b:
            obs = traj_b[-1][0]
            ee_pos = obs[:3]
            goal_pos = obs[36:39] if len(obs) >= 39 else obs[-3:]
            final_distances_b.append(np.linalg.norm(ee_pos - goal_pos))
    
    print(f"\nFinal distance to goal (Trajectory A - preferred):")
    print(f"  Mean: {np.mean(final_distances_a):.4f}")
    print(f"  Std: {np.std(final_distances_a):.4f}")
    
    print(f"\nFinal distance to goal (Trajectory B):")
    print(f"  Mean: {np.mean(final_distances_b):.4f}")
    print(f"  Std: {np.std(final_distances_b):.4f}")
    
    success_rate_a = sum(1 for d in final_distances_a if d < 0.05) / len(final_distances_a) * 100
    success_rate_b = sum(1 for d in final_distances_b if d < 0.05) / len(final_distances_b) * 100
    
    print(f"\nSuccess rate (dist < 0.05):")
    print(f"  Trajectory A: {success_rate_a:.1f}%")
    print(f"  Trajectory B: {success_rate_b:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory preference dataset")
    parser.add_argument("--input", type=str, required=True, help="Input .pkl file")
    args = parser.parse_args()
    
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    analyze_dataset(data)

if __name__ == "__main__":
    main()

