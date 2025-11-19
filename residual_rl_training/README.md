# G1 Residual RL Training - Ankle Impedance Scheduling

This folder contains the implementation of residual reinforcement learning for adapting a pretrained flat-terrain walking policy to uneven terrain through ankle impedance scheduling. The training script follows the Isaac Lab framework pattern used in the official `train.py` template.

**Robot Model**: Uses G1-12dof (12 degrees of freedom, legs only) for focused legged locomotion training.

## Overview

The approach freezes a pretrained base policy (trained on flat terrain) and trains a residual network to dynamically adjust ankle PD parameters (stiffness and damping) based on terrain conditions. This allows the robot to adapt to uneven terrain while preserving the base policy's walking behavior.

## Prerequisites

### Isaac Sim Installation (Windows)

1. **Install Isaac Sim 5.0.0** following the guide in `ISAAC_SIM_LAB_WINDOWS_INSTALLATION.md`
   - Download Isaac Sim 5.0.0 Windows binary
   - Extract to installation folder (e.g., `D:\software\isaac_sim`)
   - Run `post_install.bat` if present
   - Launch Isaac Sim to verify installation

2. **Install Isaac Lab** in Isaac Sim's Python environment:
   ```powershell
   # Install core Isaac Lab
   & "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab"
   
   # Install Isaac Lab assets
   & "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_assets"
   
   # Install Isaac Lab RL components
   & "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_rl"
   
   # Install Isaac Lab tasks
   & "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_tasks"
   ```

3. **Install Unitree RL Lab**:
   ```powershell
   cd unitree_rl_lab\source\unitree_rl_lab
   & "D:\software\isaac_sim\python.bat" -m pip install -e .
   ```

4. **Install RSL-RL**:
   ```powershell
   & "D:\software\isaac_sim\python.bat" -m pip install rsl-rl-lib>=2.3.1
   ```

### Required Files

- **Pretrained Base Policy**: `trained_models/motion.pt` (or specify path via `--base_policy_path`)
- **Robot Model**: G1-12dof USD model (provided by Unitree RL Lab)

### Unitree RL Lab Modifications

**Important**: This project requires modifications to `unitree_rl_lab` to add the G1-12DOF robot configuration. 

See `../unitree_rl_lab/MODIFICATIONS_G1_12DOF.md` for detailed instructions on how to apply these changes.

**Quick Summary**: Add the `UNITREE_G1_12DOF_CFG` configuration class to `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` after the `UNITREE_G1_29DOF_CFG` definition.

## Key Components

### 1. `train_residual.py`
Main training script that follows the official `train.py` template:
- Uses Isaac Lab's `AppLauncher` for Isaac Sim initialization
- Uses Hydra for configuration management
- Uses RSL-RL's `OnPolicyRunner` for training
- Supports video recording, checkpoint resuming, and distributed training

### 2. `tasks/g1_residual_env_cfg.py`
Configuration file defining:
- Environment parameters (observations, actions, terrain)
- Control parameters (base PD gains, residual modification ranges)
- Reward function scales
- PPO training hyperparameters (via `tasks/rsl_rl_ppo_cfg.py`)

### 3. `tasks/rsl_rl_ppo_cfg.py`
RSL-RL PPO configuration with smaller network for residual policy:
- Actor/Critic: [128, 64] hidden dimensions
- Lower initial noise: 0.5
- Optimized for residual learning task

### 4. `tasks/g1_residual_env.py`
**Custom environment class** that extends Isaac Lab's `ManagerBasedRLEnv` to integrate the frozen base policy with residual RL:

**Key Features:**
- **Base Policy Integration**: Loads and freezes a pretrained base policy (TorchScript format)
- **Residual Action Processing**: Processes residual actions (4 DOF) through Isaac Lab's action manager
- **Base Policy Inference**: Gets base policy actions (12 DOF) by constructing observations and running inference
- **Dynamic PD Modification**: Modifies ankle PD parameters (kp, kd) based on residual actions
- **Custom Torque Computation**: Computes joint torques with modified ankle PD parameters using PD control
- **Direct Torque Application**: Applies computed torques directly to the robot, bypassing normal action application

**How It Works:**
1. In `__init__`: Loads base policy from path specified in config, freezes parameters, identifies ankle joint indices
2. In `step()`: Intercepts residual actions, processes them, gets base policy actions, modifies PD parameters, computes torques, applies torques
3. The parent class (`ManagerBasedRLEnv`) handles simulation stepping, observations, rewards, and terminations

**Technical Details:**
- Base policy observations: 47D (angular velocity, projected gravity, commands, joint positions/velocities, previous actions, phase signal)
- Residual actions: 4D (ankle PD modifications, range [-1, 1])
- Base actions: 12D (all leg joints, desired positions)
- PD modification: `kp_new = clamp(kp_base + residual_action * scale * kp_base, kp_min, kp_max)`
- Torque computation: `τ = kp * (q_target - q_current) - kd * dq_current`

### 5. `g1_residual_env.py` (Legacy)
Original environment implementation using `legged_gym` framework. Kept for reference but not used in the current Isaac Lab implementation.

### 6. `g1_residual_config.py` (Legacy)
Original configuration using `legged_gym` framework. Replaced by `tasks/g1_residual_env_cfg.py` for Isaac Lab compatibility.

## Observation Space

The residual policy observes:
1. **Contact Forces (6D)**: 3D contact force per foot (normalized by body weight)
2. **Contact Binary (2D)**: Binary indicator per foot (1 if contacting, 0 otherwise)
3. **Ankle Torques (4D)**: Torque applied to each ankle joint (normalized by torque limit)

**Note**: The current config uses simplified observations. Full implementation would require custom observation terms in Isaac Lab's manager system.

## Action Space

**Residual Policy Actions (4D)**: Modify ankle PD parameters
- `left_ankle_pitch_kp`: Left ankle pitch stiffness modification
- `left_ankle_roll_kp`: Left ankle roll stiffness modification
- `right_ankle_pitch_kp`: Right ankle pitch stiffness modification
- `right_ankle_roll_kp`: Right ankle roll stiffness modification

Actions are in range [-1, 1], scaled to modify kp by ±25% of base value (40 N⋅m/rad).

**Base Policy Actions (12D)**: Control all leg joints (G1-12dof model)
- Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6 DOF)
- Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6 DOF)

## Usage

### Basic Training (Windows)

```powershell
# Navigate to project root
cd C:\path\to\MAE589_Project

# Run training (headless mode)
& "D:\software\isaac_sim\python.bat" residual_rl_training\train_residual.py --headless --task Unitree-G1-Residual-Velocity
```

### With Custom Base Policy

```powershell
& "D:\software\isaac_sim\python.bat" residual_rl_training\train_residual.py --headless --task Unitree-G1-Residual-Velocity --base_policy_path "C:\path\to\motion.pt"
```

### With Custom Number of Environments

```powershell
& "D:\software\isaac_sim\python.bat" residual_rl_training\train_residual.py --headless --task Unitree-G1-Residual-Velocity --num_envs 2048
```

### With Video Recording

```powershell
& "D:\software\isaac_sim\python.bat" residual_rl_training\train_residual.py --task Unitree-G1-Residual-Velocity --video --video_interval 2000
```

### Resume Training

```powershell
& "D:\software\isaac_sim\python.bat" residual_rl_training\train_residual.py --headless --task Unitree-G1-Residual-Velocity --resume --load_run "2025-01-15_10-30-00"
```

### Additional Options

- `--max_iterations`: Maximum training iterations (default: from config)
- `--seed`: Random seed (use -1 for random)
- `--experiment_name`: Custom experiment name
- `--run_name`: Custom run name suffix
- `--logger`: Logger type (wandb, tensorboard, neptune)
- `--distributed`: Enable multi-GPU training

## File Structure

```
residual_rl_training/
├── train_residual.py          # Main training script (Isaac Lab compatible)
├── cli_args.py                # CLI argument utilities
├── tasks/
│   ├── __init__.py            # Task registration
│   ├── g1_residual_env.py     # Custom environment class (extends ManagerBasedRLEnv)
│   ├── g1_residual_env_cfg.py # Environment configuration (Isaac Lab)
│   └── rsl_rl_ppo_cfg.py      # RSL-RL PPO configuration
├── utils/
│   ├── __init__.py
│   └── export_deploy_cfg.py   # Deployment config export
├── g1_residual_env.py         # Legacy environment (legged_gym, kept for reference)
├── g1_residual_config.py      # Legacy config (deprecated)
└── README.md                   # This file
```

## Training Output

Training logs are saved to:
```
logs/rsl_rl/g1_residual_impedance/YYYY-MM-DD_HH-MM-SS/
├── model_*.pt                  # Model checkpoints
├── progress.csv               # Training progress
├── params/
│   ├── env.yaml               # Environment config
│   ├── agent.yaml              # Agent config
│   ├── env.pkl                 # Environment config (pickle)
│   ├── agent.pkl               # Agent config (pickle)
│   └── deploy.yaml             # Deployment config
└── videos/                     # Training videos (if enabled)
```

## How Base Policy Integration Works

The custom environment class (`G1ResidualRLEnv`) extends Isaac Lab's `ManagerBasedRLEnv` to integrate a frozen base policy with residual RL. Here's how it works:

### 1. **Initialization** (`__init__`)
- Loads the base policy from the path specified in the config (`base_policy_path`)
- Freezes all base policy parameters (sets `requires_grad=False`)
- Identifies ankle joint indices by matching joint names
- Initializes buffers for residual impedance modifications

### 2. **Action Processing** (`step()` → `_apply_residual_action()`)
When the environment receives residual actions (4 DOF) from the RL policy:

1. **Process Residual Actions**: Residual actions go through Isaac Lab's action manager for scaling/clipping
2. **Get Base Policy Actions**: 
   - Constructs base policy observations (47D) from current robot state
   - Runs base policy inference (no gradients) to get 12 DOF actions
3. **Modify PD Parameters**: 
   - Scales residual actions to modify ankle kp: `kp_new = kp_base + residual * scale * kp_base`
   - Computes corresponding kd values proportionally
4. **Compute Torques**: 
   - Uses PD control with modified ankle parameters: `τ = kp * (q_target - q_current) - kd * dq`
   - Non-ankle joints use base PD parameters
5. **Apply Torques**: Applies computed torques directly to the robot

### 3. **Why This Approach?**
- ✅ **Pure Python**: No C++ code needed for training
- ✅ **Full Control**: Can modify PD parameters dynamically
- ✅ **Frozen Base Policy**: Base policy parameters never change during training
- ✅ **Isaac Lab Compatible**: Extends existing framework without breaking changes
- ✅ **Flexible**: Can easily extend to modify other joints or parameters

### 4. **Key Questions Answered**

**Q: Why doesn't this involve C++ changes?**
A: Isaac Lab's Python `ManagerBasedRLEnv` provides all the necessary APIs to access robot state, apply torques, and manage the simulation. The base policy is a PyTorch model (TorchScript), so it runs entirely in Python.

**Q: How are PD parameters modified in the simulator?**
A: We compute torques manually using modified PD parameters and apply them directly via `robot.set_joint_effort_target(torques)`. This bypasses the normal action application (which would use fixed PD parameters) and gives us full control.

**Q: Does the base policy get updated during training?**
A: No. The base policy is loaded once, frozen (no gradients), and used only for inference. Only the residual policy (4 DOF) is trained.

**Q: What happens if the base policy path is not provided?**
A: The environment will warn and use zero actions for the base policy. This is useful for testing but not recommended for actual training.

## Key Design Decisions

1. **Frozen Base Policy**: The base policy is loaded and frozen (no gradients) to preserve flat-terrain behavior
2. **Ankle-Only Modification**: Only ankle joints have adaptive impedance (most critical for terrain adaptation)
3. **Bounded Modifications**: Impedance changes are limited to ±25% to maintain stability
4. **Isaac Lab Framework**: Training script follows official `train.py` template for consistency
5. **Custom Environment Class**: Extends `ManagerBasedRLEnv` to intercept action application and integrate base policy

## Known Limitations

1. **Base Policy Observations**: Currently uses simplified projected gravity computation. Full implementation would compute proper projected gravity from base orientation quaternion.

2. **Command Manager Integration**: Base policy observations currently use zero commands. Full implementation would integrate with Isaac Lab's command manager to get velocity commands.

3. **Observation Terms**: The current implementation uses Isaac Lab's standard observation terms. Custom observation terms for residual RL (contact forces, ankle torques) could be added for better performance.

## Future Improvements

- [x] Refactor environment to fully integrate with Isaac Lab's manager-based system
- [x] Integrate base policy loading into Isaac Lab environment
- [ ] Improve base policy observation construction (projected gravity, commands)
- [ ] Create custom observation terms for residual RL (contact forces, ankle torques)
- [ ] Add more observations (foot heights, velocities, terrain height map)
- [ ] Extend to multiple joints (knee, hip)
- [ ] Implement curriculum learning for terrain difficulty

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. Isaac Lab is installed in Isaac Sim's Python environment (not a separate conda environment)
2. Unitree RL Lab is installed and in the Python path
3. RSL-RL is installed: `rsl-rl-lib>=2.3.1`

### Base Policy Not Found

If the base policy is not found:
- Check that `trained_models/motion.pt` exists
- Or specify path via `--base_policy_path`

### CUDA/GPU Issues

- Ensure NVIDIA GPU drivers are up to date
- Verify CUDA is available: Isaac Sim should detect `cuda:0` automatically
- For multi-GPU training, use `--distributed` flag

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Documentation](https://github.com/leggedrobotics/rsl_rl)
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)
