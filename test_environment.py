#!/usr/bin/env python3
"""
Quick test script to verify Unitree RL Lab environment creation without training.
This will test if the environment can be created and run for a few steps.
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'unitree_rl_lab', 'source', 'unitree_rl_lab'))

from isaaclab.app import AppLauncher

# Launch Isaac Sim in headless mode
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import unitree_rl_lab.tasks  # noqa: F401

def test_environment(task_name, num_steps=5):
    """Test environment creation and basic functionality."""
    print(f"\n🧪 Testing {task_name}...")
    
    try:
        # Create environment
        print(f"  📦 Creating environment: {task_name}")
        env = gym.make(task_name, num_envs=1)
        
        # Reset environment
        print(f"  🔄 Resetting environment...")
        obs, info = env.reset()
        
        print(f"  ✅ Environment created successfully!")
        print(f"  📊 Observation space: {env.observation_space}")
        print(f"  🎮 Action space: {env.action_space}")
        print(f"  📏 Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        
        # Run a few steps
        print(f"  🏃 Running {num_steps} test steps...")
        for i in range(num_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"    Step {i+1}: Reward={reward[0]:.3f}, Terminated={terminated[0]}, Truncated={truncated[0]}")
            
            if terminated[0] or truncated[0]:
                print(f"    🔄 Episode ended, resetting...")
                obs, info = env.reset()
        
        print(f"  ✅ {task_name} test completed successfully!")
        env.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing {task_name}: {str(e)}")
        return False

def main():
    """Run tests for all available Unitree environments."""
    print("🚀 Starting Unitree RL Lab Environment Tests")
    print("=" * 60)
    
    # Test all available environments
    environments = [
        "Unitree-G1-29dof-Velocity",
        "Unitree-Go2-Velocity", 
        "Unitree-H1-Velocity"
    ]
    
    results = {}
    for env_name in environments:
        results[env_name] = test_environment(env_name, num_steps=3)
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    for env_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {env_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready for training!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up
        if 'simulation_app' in locals():
            simulation_app.close()
