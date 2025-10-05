# MAE589 Project - Unitree RL Training

This repository contains the setup and code for Unitree robot reinforcement learning training, designed to work across multiple platforms.

## Platform Support

- **Windows 11**: Primary platform for RL training with GPU acceleration
- **macOS**: Development and MuJoCo simulation

## Repository Structure

```
MAE589_Project/
├── unitree_rl_lab/          # Cloned Unitree RL Lab repository
├── guidance.md              # Windows setup guide
├── .gitignore              # Cross-platform ignore rules
└── README.md               # This file
```

## Setup Instructions

### Windows 11 (RL Training)
Follow the detailed guide in `guidance.md` for:
1. Python environment setup
2. MuJoCo installation
3. Unitree RL Lab setup
4. Isaac Lab/Isaac Sim installation

### macOS (MuJoCo Development)
Use the macOS-specific setup for MuJoCo development and testing.

## Version Control Strategy

- **Main branch**: Stable, cross-platform code
- **Platform branches**: Platform-specific configurations and optimizations
- **Shared code**: Core algorithms and models work across platforms

## Getting Started

1. Clone this repository
2. Follow platform-specific setup instructions
3. Refer to `unitree_rl_lab/README.md` for detailed RL training setup
