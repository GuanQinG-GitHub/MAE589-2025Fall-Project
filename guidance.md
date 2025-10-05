# Windows Unitree RL Training Setup Guide

## Overview
This guide provides a streamlined approach to set up Unitree RL training on Windows, focusing on three main components: MuJoCo installation, Unitree RL Lab setup, and Isaac Lab/Isaac Sim installation.

---

## Prerequisites
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+ (32GB recommended for training)
- **Storage**: 10GB+ free space

---

## Step 1: Install Python Environment

### Install Anaconda/Miniconda
1. **Download Anaconda**: https://www.anaconda.com/download
2. **Install Anaconda** with default settings
3. **Open Anaconda Prompt** (not regular Command Prompt)

### Create Python Environment
```bash
# Create a new conda environment
conda create -n unitree_rl python=3.10 -y

# Activate the environment
conda activate unitree_rl

# Verify Python version
python --version
```

---

## Step 2: Install MuJoCo

### Install MuJoCo via pip (Recommended)
```bash
# Install MuJoCo
pip install mujoco

# Install additional MuJoCo utilities
pip install mujoco-viewer

# Test installation
python -c "import mujoco; print('MuJoCo installed successfully!')"
```

### Alternative: Manual MuJoCo Installation
1. **Download MuJoCo**: https://mujoco.org/download
2. **Extract** to `C:\Users\[YourUsername]\.mujoco\mujoco-3.0.0\`
3. **Set environment variable**: `MUJOCO_PATH=C:\Users\[YourUsername]\.mujoco\mujoco-3.0.0\`

---

## Step 3: Setup Unitree RL Lab

### Clone and Setup Unitree RL Lab Repository
1. **Navigate to the official repository**: https://github.com/unitreerobotics/unitree_rl_lab
2. **Read the README.md** for detailed setup instructions
3. **Follow the official installation guide** in the repository

### Basic Setup Commands
```bash
# Navigate to your desired directory
cd C:\Users\[YourUsername]\Documents\

# Clone the repository
git clone https://github.com/unitreerobotics/unitree_rl_lab.git

# Navigate to the repository
cd unitree_rl_lab

# Follow the README.md instructions for:
# - Installing dependencies
# - Setting up the environment
# - Downloading robot models
# - Configuring paths
```

### Key Components from Unitree RL Lab
- **Environment setup**: Follow the conda environment creation
- **Dependencies**: Install all required packages as specified
- **Robot models**: Download Unitree robot USD models
- **Configuration**: Set up proper model paths and settings

---

## Step 4: Install Isaac Lab and Isaac Sim

### Follow NVIDIA Official Documentation
1. **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
2. **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html

### Installation Steps (Follow Official Docs)
1. **Install Isaac Sim** using the official NVIDIA installation guide
2. **Install Isaac Lab** following the official setup instructions
3. **Configure environment variables** as specified in the documentation
4. **Verify installation** using the provided test scripts

### Key Requirements
- **NVIDIA GPU** with CUDA support
- **Windows 11** (recommended) or Windows 10
- **Sufficient RAM** (32GB+ recommended)
- **Proper NVIDIA drivers** installed

---

## Step 5: Integration and Testing

### Test Complete Setup
```bash
# Test MuJoCo
python -c "import mujoco; print('MuJoCo working!')"

# Test Unitree RL Lab (follow their README)
python scripts/list_envs.py

# Test Isaac Lab (follow their documentation)
# Use the verification scripts provided in Isaac Lab docs
```

### Verify All Components
- ✅ **MuJoCo**: Basic physics simulation working
- ✅ **Unitree RL Lab**: Environment registration successful
- ✅ **Isaac Lab**: Isaac Sim integration working
- ✅ **Isaac Sim**: NVIDIA simulation platform operational

---

## Important Notes

### Documentation Priority
1. **Always refer to official documentation first**
2. **Unitree RL Lab README.md** is the primary source for RL setup
3. **NVIDIA Isaac Lab docs** are authoritative for Isaac Sim installation
4. **This guide provides the framework** - follow official docs for details

### Troubleshooting
- **Check official documentation** for common issues
- **Verify system requirements** match NVIDIA specifications
- **Ensure proper GPU drivers** are installed
- **Follow exact installation order** specified in official docs

---

## Resources

### Official Documentation
- **Unitree RL Lab**: https://github.com/unitreerobotics/unitree_rl_lab
- **Isaac Lab**: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- **Isaac Sim**: https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html
- **MuJoCo**: https://mujoco.readthedocs.io/

### Key Points
- **Follow official documentation** for detailed setup instructions
- **This guide provides the high-level framework** only
- **Each component has specific requirements** detailed in their respective docs
- **Integration testing** is crucial for successful setup
