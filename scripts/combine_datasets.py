import pickle
import numpy as np
import random
import argparse
from pathlib import Path

def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    combined = []
    
    for input_path in args.inputs:
        print(f"Loading {input_path}...")
        data = load_dataset(input_path)
        print(f"  Loaded {len(data)} pairs")
        combined.extend(data)
    
    print(f"\nTotal combined: {len(combined)} pairs")
    
    if args.shuffle:
        random.shuffle(combined)
        print("Shuffled dataset")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(combined, f)
    
    print(f"Saved to {output_path}")
    
    instructions = set(d[0] for d in combined)
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {len(combined)}")
    print(f"  Unique instructions: {len(instructions)}")

if __name__ == "__main__":
    main()

