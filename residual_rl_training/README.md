# G1 Residual RL Training - Ankle Impedance Scheduling

This folder contains the implementation of residual reinforcement learning for adapting a pretrained flat-terrain walking policy to uneven terrain through ankle impedance scheduling.

## Overview

The approach freezes a pretrained base policy (trained on flat terrain) and trains a residual network to dynamically adjust ankle PD parameters (stiffness and damping) based on terrain conditions. This allows the robot to adapt to uneven terrain while preserving the base policy's walking behavior.

## Key Components

### 1. `g1_residual_config.py`
Configuration file defining:
- Environment parameters (observations, actions, terrain)
- Control parameters (base PD gains, residual modification ranges)
- Reward function scales
- PPO training hyperparameters

### 2. `g1_residual_env.py`
Environment class that:
- Loads and freezes the pretrained base policy
- Implements residual action processing (ankle impedance modifications)
- Computes observations: contact forces, contact binary, ankle torques
- Defines reward functions for terrain adaptation

### 3. `train_residual.py`
Training script that:
- Sets up the environment and algorithm
- Loads the base policy
- Runs PPO training for the residual network

## Observation Space (12 dimensions)

1. **Contact Forces (6D)**: 3D contact force per foot (normalized by body weight)
2. **Contact Binary (2D)**: Binary indicator per foot (1 if contacting, 0 otherwise)
3. **Ankle Torques (4D)**: Torque applied to each ankle joint (normalized by torque limit)

Additional observations (commented out for future use):
- Foot heights relative to base
- Foot velocities
- Base orientation
- Contact force history

## Action Space (4 dimensions)

Residual actions modify ankle PD parameters:
- `left_ankle_pitch_kp`: Left ankle pitch stiffness modification
- `left_ankle_roll_kp`: Left ankle roll stiffness modification
- `right_ankle_pitch_kp`: Right ankle pitch stiffness modification
- `right_ankle_roll_kp`: Right ankle roll stiffness modification

Actions are in range [-1, 1], scaled to modify kp by ±25% of base value (40 N⋅m/rad).

## Reward Functions

### Primary Rewards:
1. **Terrain Adaptation**: Rewards forward progress and base height maintenance
2. **Impedance Smoothness**: Penalizes rapid impedance changes
3. **Contact Timing**: Rewards proper foot contact during stance phase
4. **Velocity Tracking**: Rewards tracking desired linear/angular velocities

### Secondary Rewards:
- Orientation stability
- Joint smoothness
- Base height maintenance
- Sliding prevention

## Usage

### Basic Training:
```bash
cd residual_rl_training
python train_residual.py --task=g1_residual --headless
```

### With Custom Base Policy:
```bash
python train_residual.py --task=g1_residual --base_policy_path=/path/to/policy.pt
```

### With Custom Number of Environments:
```bash
python train_residual.py --task=g1_residual --num_envs=2048
```

## Requirements

- Isaac Gym (for training)
- PyTorch
- Pretrained base policy (`motion.pt` from `deploy/pre_train/g1/`)

## File Structure

```
residual_rl_training/
├── g1_residual_config.py    # Configuration
├── g1_residual_env.py        # Environment implementation
├── train_residual.py         # Training script
└── README.md                 # This file
```

## Key Design Decisions

1. **Frozen Base Policy**: The base policy is loaded and frozen (no gradients) to preserve flat-terrain behavior
2. **Ankle-Only Modification**: Only ankle joints have adaptive impedance (most critical for terrain adaptation)
3. **Bounded Modifications**: Impedance changes are limited to ±25% to maintain stability
4. **Simple Observations**: Starting with minimal observation set (contact forces, binary, torques) for proof of concept

## Future Improvements

- Add more observations (foot heights, velocities, terrain height map)
- Extend to multiple joints (knee, hip)
- Add residual actions in addition to impedance scheduling
- Implement curriculum learning for terrain difficulty

