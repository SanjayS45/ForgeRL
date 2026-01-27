#!/usr/bin/env python3
"""
Test MetaWorld environment setup.

Verifies that MetaWorld V3 environments are correctly instantiated
using the train_classes approach (not train_tasks directly).

Usage:
    python scripts/test_env.py
    python scripts/test_env.py --task push-v3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_metaworld_v3(task_name: str = "reach-v3"):
    """
    Test MetaWorld V3 environment setup.
    
    This demonstrates the CORRECT way to instantiate MetaWorld environments.
    """
    import metaworld
    
    print(f"Testing MetaWorld task: {task_name}")
    print("-" * 50)
    
    # Step 1: Create ML1 benchmark
    print("1. Creating ML1 benchmark...")
    ml1 = metaworld.ML1(task_name, seed=42)
    print(f"   ✓ ML1 created")
    
    # Step 2: Get environment CLASS from train_classes
    print("\n2. Getting environment class from train_classes...")
    print(f"   Available classes: {list(ml1.train_classes.keys())}")
    env_cls = ml1.train_classes[task_name]
    print(f"   ✓ Got class: {env_cls}")
    
    # Step 3: Instantiate the environment
    print("\n3. Instantiating environment...")
    env = env_cls()
    print(f"   ✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Step 4: Set a task (goal configuration)
    print("\n4. Setting task from train_tasks...")
    print(f"   Number of available tasks: {len(ml1.train_tasks)}")
    task = ml1.train_tasks[0]
    env.set_task(task)
    print(f"   ✓ Task set")
    
    # Step 5: Test reset
    print("\n5. Testing reset...")
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
        print(f"   ✓ Reset returns (obs, info) tuple")
    else:
        obs = result
        print(f"   ✓ Reset returns obs only")
    print(f"   Observation shape: {obs.shape}")
    
    # Step 6: Test step
    print("\n6. Testing step...")
    action = env.action_space.sample()
    result = env.step(action)
    
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        print(f"   ✓ Step returns 5-tuple (gymnasium style)")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
    else:
        obs, reward, done, info = result
        print(f"   ✓ Step returns 4-tuple (gym style)")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Done: {done}")
    
    # Step 7: Run a few more steps
    print("\n7. Running 100 steps...")
    total_reward = reward
    for i in range(99):
        action = env.action_space.sample()
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total_reward += reward
        
        if done:
            env.reset()
            
    print(f"   ✓ Completed 100 steps")
    print(f"   Total reward: {total_reward:.4f}")
    
    # Step 8: Test observation structure for reach task
    if "reach" in task_name:
        print("\n8. Analyzing observation structure (reach task)...")
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        print(f"   Full obs shape: {obs.shape}")
        print(f"   Gripper pos (obs[0:3]): {obs[0:3]}")
        print(f"   Goal pos (obs[-3:]): {obs[-3:]}")
        distance = np.linalg.norm(obs[0:3] - obs[-3:])
        print(f"   Distance to goal: {distance:.4f}")
    
    env.close()
    print("\n" + "=" * 50)
    print("✓ All tests passed! Environment is working correctly.")
    print("=" * 50)
    
    return True


def test_env_wrapper():
    """Test the custom environment wrapper."""
    print("\n" + "=" * 50)
    print("Testing custom MetaWorldEnv wrapper...")
    print("=" * 50)
    
    from envs.metaworld_wrapper import MetaWorldEnv, make_metaworld_env
    
    # Test basic wrapper
    print("\n1. Testing MetaWorldEnv wrapper...")
    env = MetaWorldEnv(task_name="reach-v3", seed=42)
    obs, info = env.reset()
    print(f"   ✓ Reset successful, obs shape: {obs.shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Step successful, reward: {reward:.4f}")
    
    # Test task switching
    print("\n2. Testing task switching...")
    env.set_task_by_index(1)
    obs, info = env.reset()
    print(f"   ✓ Task switch successful")
    
    # Test factory function
    print("\n3. Testing make_metaworld_env factory...")
    env2 = make_metaworld_env(task_name="reach-v3", seed=123)
    obs, info = env2.reset()
    print(f"   ✓ Factory function works")
    
    env.close()
    env2.close()
    
    print("\n✓ Wrapper tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test MetaWorld environment setup")
    parser.add_argument(
        "--task", type=str, default="reach-v3",
        help="MetaWorld task name"
    )
    parser.add_argument(
        "--test-wrapper", action="store_true",
        help="Also test custom wrapper"
    )
    
    args = parser.parse_args()
    
    # Test raw MetaWorld
    success = test_metaworld_v3(args.task)
    
    # Test wrapper
    if args.test_wrapper and success:
        test_env_wrapper()
        

if __name__ == "__main__":
    main()
