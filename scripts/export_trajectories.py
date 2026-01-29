import argparse
import pickle
import json
import numpy as np
from pathlib import Path

def export_to_json(data, output_path):
    json_data = []
    
    for instruction, traj_a, traj_b in data:
        pair = {
            "instruction": instruction,
            "trajectory_a": [
                {
                    "observation": obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                    "action": act.tolist() if hasattr(act, 'tolist') else list(act),
                }
                for obs, act in traj_a
            ],
            "trajectory_b": [
                {
                    "observation": obs.tolist() if hasattr(obs, 'tolist') else list(obs),
                    "action": act.tolist() if hasattr(act, 'tolist') else list(act),
                }
                for obs, act in traj_b
            ],
        }
        json_data.append(pair)
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Exported {len(json_data)} pairs to {output_path}")

def export_summary(data, output_path):
    summary = {
        "num_pairs": len(data),
        "instructions": list(set(d[0] for d in data)),
        "avg_trajectory_length": np.mean([len(d[1]) for d in data]),
    }
    
    if data:
        summary["obs_dim"] = len(data[0][1][0][0]) if data[0][1] else 0
        summary["act_dim"] = len(data[0][1][0][1]) if data[0][1] else 0
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Exported summary to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export trajectory dataset to different formats")
    parser.add_argument("--input", type=str, required=True, help="Input .pkl file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--format", type=str, choices=["json", "summary"], default="json", help="Output format")
    args = parser.parse_args()
    
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} pairs from {args.input}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "json":
        export_to_json(data, output_path)
    elif args.format == "summary":
        export_summary(data, output_path)

if __name__ == "__main__":
    main()

