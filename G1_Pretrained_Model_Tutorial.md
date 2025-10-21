# Unitree G1 Pre-trained Model Integration with MuJoCo

This tutorial explains how to use the pre-trained Unitree G1 humanoid model from the [Unitree RL Gym repository](https://github.com/unitreerobotics/unitree_rl_gym) with MuJoCo for motion control.

## Overview

The Unitree RL Gym provides pre-trained reinforcement learning models for various Unitree robots, including the G1 humanoid. These models are trained using Isaac Gym and can be deployed to MuJoCo for simulation and testing.

## Pre-trained Model Details

### Model Architecture
- **Type**: LSTM-based Actor-Critic Recurrent Neural Network
- **Policy Class**: `ActorCriticRecurrent`
- **RNN Type**: LSTM
- **Hidden Size**: 64
- **Layers**: 1

### Input/Output Specifications

#### Input (Observations) - 47 dimensions:
1. **Angular Velocity** (3D): `[ωx, ωy, ωz]` - Base angular velocity
2. **Gravity Orientation** (3D): `[gx, gy, gz]` - Projected gravity vector
3. **Commands** (3D): `[vx, vy, ωz]` - Linear velocity commands (x, y) and angular velocity (z)
4. **Joint Positions** (12D): `[q1, q2, ..., q12]` - Joint angles relative to default positions
5. **Joint Velocities** (12D): `[dq1, dq2, ..., dq12]` - Joint angular velocities
6. **Previous Actions** (12D): `[a1, a2, ..., a12]` - Previous control actions
7. **Phase Information** (2D): `[sin(phase), cos(phase)]` - Gait phase for walking

#### Output (Actions) - 12 dimensions:
- **Target Joint Positions** (12D): `[θ1, θ2, ..., θ12]` - Target angles for 12 DOF joints

### Joint Mapping (12 DOF):
1. `left_hip_yaw_joint`
2. `left_hip_roll_joint`
3. `left_hip_pitch_joint`
4. `left_knee_joint`
5. `left_ankle_pitch_joint`
6. `left_ankle_roll_joint`
7. `right_hip_yaw_joint`
8. `right_hip_roll_joint`
9. `right_hip_pitch_joint`
10. `right_knee_joint`
11. `right_ankle_pitch_joint`
12. `right_ankle_roll_joint`

### Control Parameters:
- **Control Type**: Position control (P)
- **Action Scale**: 0.25
- **Control Decimation**: 10 (50Hz control frequency)
- **Simulation DT**: 0.002s (500Hz simulation frequency)

### PD Controller Gains:
```yaml
Stiffness (Kp):
  - Hip joints: 100 N⋅m/rad
  - Knee joints: 150 N⋅m/rad
  - Ankle joints: 40 N⋅m/rad

Damping (Kd):
  - Hip joints: 2 N⋅m⋅s/rad
  - Knee joints: 4 N⋅m⋅s/rad
  - Ankle joints: 2 N⋅m⋅s/rad
```

## Installation Requirements

### 1. Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### 2. Install MuJoCo
```bash
pip install mujoco
```

### 3. Install Additional Dependencies
```bash
pip install numpy pyyaml
```

## Usage with MuJoCo

### Basic Integration

The pre-trained model can be loaded and used with MuJoCo as follows:

```python
import torch
import mujoco
import numpy as np

# Load the pre-trained policy
policy = torch.jit.load("path/to/motion.pt")

# Create observation vector (47 dimensions)
obs = np.zeros(47, dtype=np.float32)

# Get action from policy
obs_tensor = torch.from_numpy(obs).unsqueeze(0)
action = policy(obs_tensor).detach().numpy().squeeze()

# Apply action to robot (after scaling and adding default angles)
target_positions = action * action_scale + default_angles
```

### Observation Construction

The observation vector must be constructed exactly as the training environment:

```python
def construct_observation(qpos, qvel, quat, omega, cmd_vel, prev_action, phase):
    """
    Construct observation vector for G1 pre-trained model
    
    Args:
        qpos: Joint positions (12D)
        qvel: Joint velocities (12D) 
        quat: Base quaternion (4D)
        omega: Base angular velocity (3D)
        cmd_vel: Command velocity [vx, vy, ωz] (3D)
        prev_action: Previous action (12D)
        phase: Gait phase (0-1)
    
    Returns:
        obs: Observation vector (47D)
    """
    obs = np.zeros(47, dtype=np.float32)
    
    # Angular velocity (3D)
    obs[:3] = omega * 0.25  # ang_vel_scale
    
    # Gravity orientation (3D)
    gravity_orientation = get_gravity_orientation(quat)
    obs[3:6] = gravity_orientation
    
    # Commands (3D)
    cmd_scale = np.array([2.0, 2.0, 0.25])
    obs[6:9] = cmd_vel * cmd_scale
    
    # Joint positions (12D) - relative to default
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                              -0.1, 0.0, 0.0, 0.3, -0.2, 0.0])
    obs[9:21] = (qpos - default_angles) * 1.0  # dof_pos_scale
    
    # Joint velocities (12D)
    obs[21:33] = qvel * 0.05  # dof_vel_scale
    
    # Previous actions (12D)
    obs[33:45] = prev_action
    
    # Phase information (2D)
    obs[45] = np.sin(2 * np.pi * phase)
    obs[46] = np.cos(2 * np.pi * phase)
    
    return obs
```

### Command Interface

The model accepts velocity commands in the format:
- `[vx, vy, ωz]` where:
  - `vx`: Forward/backward velocity (m/s)
  - `vy`: Left/right velocity (m/s) 
  - `ωz`: Yaw angular velocity (rad/s)

### Example Commands:
- **Stand Still**: `[0.0, 0.0, 0.0]`
- **Walk Forward**: `[0.5, 0.0, 0.0]`
- **Walk Backward**: `[-0.5, 0.0, 0.0]`
- **Strafe Left**: `[0.0, 0.3, 0.0]`
- **Turn Left**: `[0.0, 0.0, 0.5]`
- **Turn Right**: `[0.0, 0.0, -0.5]`

## Model Files

### Pre-trained Model Location:
```
external/unitree_rl_gym/deploy/pre_train/g1/motion.pt
```

### MuJoCo Scene File:
```
external/unitree_rl_gym/resources/robots/g1_description/scene.xml
```

### Configuration File:
```
external/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml
```

## Performance Characteristics

- **Walking Speed**: Up to 1.5 m/s forward
- **Turning Rate**: Up to 1.0 rad/s
- **Stability**: Robust to small perturbations
- **Gait**: Dynamic walking with natural humanoid motion

## Limitations

1. **Model Specificity**: Trained specifically for 12-DOF G1 configuration
2. **Command Range**: Limited to reasonable walking speeds
3. **Environment**: Optimized for flat terrain
4. **Real-time**: Requires 50Hz control frequency for stable operation

## Troubleshooting

### Common Issues:

1. **Model Loading Error**: Ensure PyTorch version compatibility
2. **Observation Mismatch**: Verify observation vector construction
3. **Unstable Motion**: Check control frequency and PD gains
4. **Poor Tracking**: Verify command scaling and joint limits

### Debug Tips:

1. Monitor observation values during runtime
2. Check action scaling and joint limits
3. Verify phase calculation for gait timing
4. Ensure proper gravity orientation calculation

## References

- [Unitree RL Gym Repository](https://github.com/unitreerobotics/unitree_rl_gym)
- [Unitree G1 Robot Documentation](https://www.unitree.com/products/g1)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
