#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import (
    load_dataset,
    save_dataset,
    validate_dataset,
    compute_stats,
    preprocess_dataset,
    split_dataset,
    shuffle_dataset,
)

def main():
    parser = argparse.ArgumentParser(description="Preprocess trajectory dataset")
    
    parser.add_argument(
        "input", type=str, nargs="?", default="experiments/trajs.pkl",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: {input}_processed.pkl)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate, don't process"
    )
    parser.add_argument(
        "--normalize-obs", action="store_true",
        help="Normalize observations"
    )
    parser.add_argument(
        "--normalize-actions", action="store_true",
        help="Normalize actions"
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Maximum trajectory length"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Split into train/val/test"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle the dataset"
    )
    
    args = parser.parse_args()
    

    print(f"Loading dataset from {args.input}...")
    data = load_dataset(args.input)
    print(f"Loaded {len(data)} preference pairs")
    

    print("\nValidating dataset...")
    is_valid, errors = validate_dataset(data)
    
    if not is_valid:
        print("Validation FAILED:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        sys.exit(1)
    else:
        print("✓ Dataset is valid")
        

    stats = compute_stats(data)
    print(f"\n{stats}")
    
    if args.validate_only:
        return
        

    if args.normalize_obs or args.normalize_actions or args.max_length:
        print("\nPreprocessing...")
        data = preprocess_dataset(
            data,
            max_length=args.max_length,
            normalize_obs=args.normalize_obs,
            normalize_actions=args.normalize_actions,
        )
        print("✓ Preprocessing complete")
        

    if args.shuffle:
        print("\nShuffling...")
        data = shuffle_dataset(data, seed=args.seed)
        print("✓ Shuffled")
        

    if args.split:
        print("\nSplitting dataset...")
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        train_data, val_data, test_data = split_dataset(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
        print(f"  Train: {len(train_data)} pairs")
        print(f"  Val: {len(val_data)} pairs")
        print(f"  Test: {len(test_data)} pairs")
        

        input_path = Path(args.input)
        base = input_path.stem
        parent = input_path.parent
        
        save_dataset(train_data, parent / f"{base}_train.pkl")
        save_dataset(val_data, parent / f"{base}_val.pkl")
        save_dataset(test_data, parent / f"{base}_test.pkl")
        
    else:

        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_processed.pkl"
            
        save_dataset(data, output_path)
        

if __name__ == "__main__":
    main()

