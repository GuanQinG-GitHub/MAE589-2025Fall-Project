"""
Training script for G1 Residual RL - Ankle Impedance Scheduling
This script trains a residual network to adapt ankle PD parameters
for uneven terrain while keeping the base policy frozen.
"""

import sys
import os

# Add legged_gym to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'unitree_rl_gym'))

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

import isaacgym
import torch


def train_residual(args):
    """
    Main training function for residual RL.
    
    Args:
        args: Command line arguments
    """
    # ====================================================================
    # SETUP AND CONFIGURATION
    # ====================================================================
    print("=" * 80)
    print("G1 Residual RL Training - Ankle Impedance Scheduling")
    print("=" * 80)
    
    # Get configurations
    # Note: We'll register the task in the script
    from g1_residual_config import G1ResidualCfg, G1ResidualCfgPPO
    
    env_cfg = G1ResidualCfg()
    train_cfg = G1ResidualCfgPPO()
    
    # ====================================================================
    # BASE POLICY PATH CONFIGURATION
    # ====================================================================
    # Set path to pretrained base policy
    # Default: use the motion.pt from deploy folder
    base_policy_path = getattr(args, 'base_policy_path', None)
    if base_policy_path is None:
        # Default path to pretrained policy
        base_policy_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'deploy',
            'pre_train',
            'g1',
            'motion.pt'
        )
    
    # Check if base policy exists
    if not os.path.exists(base_policy_path):
        print(f"Warning: Base policy not found at {base_policy_path}")
        print("Training without base policy (not recommended)")
        base_policy_path = None
    else:
        print(f"Using base policy: {base_policy_path}")
    
    # Add base policy path to config
    env_cfg.base_policy_path = base_policy_path
    
    # ====================================================================
    # ENVIRONMENT CREATION
    # ====================================================================
    print("\n" + "=" * 80)
    print("Creating Environment")
    print("=" * 80)
    
    # Override some parameters from command line if provided
    if hasattr(args, 'num_envs') and args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    
    if hasattr(args, 'headless') and args.headless:
        args.headless = True
    else:
        args.headless = False
    
    # Create simulation parameters
    from legged_gym.utils.helpers import parse_sim_params
    sim_params = parse_sim_params(args, env_cfg)
    
    # Create environment
    from g1_residual_env import G1ResidualRobot
    env = G1ResidualRobot(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless
    )
    
    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"Observation space: {env.num_obs} dimensions")
    print(f"Action space: {env.num_actions} dimensions (ankle impedance modifications)")
    
    # ====================================================================
    # ALGORITHM AND RUNNER SETUP
    # ====================================================================
    print("\n" + "=" * 80)
    print("Setting up PPO Algorithm")
    print("=" * 80)
    
    # Convert config to dictionary
    train_cfg_dict = class_to_dict(train_cfg)
    
    # Create log directory
    from datetime import datetime
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # Create algorithm runner
    from legged_gym.utils import OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    
    # ====================================================================
    # RESUME TRAINING (IF SPECIFIED)
    # ====================================================================
    if train_cfg.runner.resume:
        from legged_gym.utils.helpers import get_load_path
        resume_path = get_load_path(
            log_root,
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint
        )
        if resume_path is not None:
            print(f"Resuming training from: {resume_path}")
            runner.load(resume_path)
        else:
            print("Warning: Resume requested but no checkpoint found")
    
    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Max iterations: {train_cfg.runner.max_iterations}")
    print(f"Steps per env per iteration: {train_cfg.runner.num_steps_per_env}")
    print(f"Total environment steps per iteration: {env.num_envs * train_cfg.runner.num_steps_per_env}")
    print("=" * 80)
    
    # Start training
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Final model saved to: {log_dir}")
    print("=" * 80)


if __name__ == '__main__':
    # ====================================================================
    # COMMAND LINE ARGUMENTS
    # ====================================================================
    from legged_gym.utils import get_args
    
    # Parse arguments
    args = get_args()
    
    # Add custom arguments for residual training
    import argparse
    parser = argparse.ArgumentParser(description='Train G1 Residual RL')
    parser.add_argument(
        '--base_policy_path',
        type=str,
        default=None,
        help='Path to pretrained base policy (default: deploy/pre_train/g1/motion.pt)'
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=None,
        help='Number of parallel environments (default: from config)'
    )
    
    # Merge with existing args
    residual_args = parser.parse_args()
    for key, value in vars(residual_args).items():
        if value is not None:
            setattr(args, key, value)
    
    # ====================================================================
    # RUN TRAINING
    # ====================================================================
    train_residual(args)

