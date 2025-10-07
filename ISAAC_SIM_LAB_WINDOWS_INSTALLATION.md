# Isaac Sim 5.0.0 + Isaac Lab Installation Guide for Windows 11

## Overview
This guide provides step-by-step instructions for installing Isaac Sim 5.0.0 and Isaac Lab on Windows 11, including common issues and solutions encountered during the installation process.

## Prerequisites
- **Operating System**: Windows 11
- **GPU**: NVIDIA GPU with CUDA support (RTX 5070 tested)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space
- **Python**: Will be managed by Isaac Sim

## Part 1: Isaac Sim 5.0.0 Installation

### Step 1: Download Isaac Sim
1. Go to [NVIDIA Isaac Sim Download Page](https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html)
2. Find "Isaac Sim | Version 5.0.0" under "Latest Release" section
3. Download the **Windows binary (zip)** for Isaac Sim 5.0.0
4. Optionally download the **Isaac Sim Compatibility Checker** for Windows 5.0.0

### Step 2: Install Isaac Sim
1. **Create installation folder**: `D:\software\isaac_sim` (or your preferred location)
2. **Extract the zip file** into the installation folder
3. **Run post-install script**: Double-click `post_install.bat` if present
4. **Launch Isaac Sim**: Double-click `isaac-sim.selector.bat`
5. **Choose "Isaac Sim Full"** and click START

### Step 3: Verify Isaac Sim Installation
- First launch may take 5-10 minutes for shader cache initialization
- Verify GUI opens successfully
- Test basic functionality

## Part 2: Isaac Lab Installation

### ⚠️ **CRITICAL: Environment Strategy**

**DO NOT** install Isaac Lab in a separate conda environment. Isaac Lab must be installed in Isaac Sim's Python environment to avoid import conflicts.

### Step 1: Clone Isaac Lab Repository
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### Step 2: Install Isaac Lab in Isaac Sim's Python Environment
```bash
# Install core Isaac Lab
& "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab"

# Install Isaac Lab assets
& "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_assets"

# Install Isaac Lab RL components
& "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_rl"

# Install Isaac Lab tasks
& "D:\software\isaac_sim\python.bat" -m pip install -e "C:\path\to\IsaacLab\source\isaaclab_tasks"
```

### Step 3: Verify Isaac Lab Installation
```bash
# Test basic import
& "D:\software\isaac_sim\python.bat" -c "import isaaclab; print('Isaac Lab imported successfully!')"

# Test AppLauncher import
& "D:\software\isaac_sim\python.bat" -c "from isaaclab.app import AppLauncher; print('AppLauncher imported successfully!')"

# Test demo script
& "D:\software\isaac_sim\python.bat" scripts/demos/quadcopter.py --headless
```

## Expected Output

When running the quadcopter demo, you should see:
```
[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object...
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: ...isaaclab.python.headless.kit
[INFO]: Setup complete...
>>>>>>>> Reset!
```

## Common Issues and Solutions

### Issue 1: Environment Mismatch
**Problem**: `ModuleNotFoundError: No module named 'isaacsim'`
**Cause**: Isaac Lab installed in conda environment, Isaac Sim installed system-wide
**Solution**: 
1. Uninstall Isaac Lab from conda: `pip uninstall isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks -y`
2. Install Isaac Lab in Isaac Sim's Python environment (see Step 2 above)

### Issue 2: EULA Acceptance
**Problem**: Isaac Sim requires EULA acceptance on first run
**Solution**: 
1. Run: `& "D:\software\isaac_sim\python.bat" -c "import isaacsim"`
2. When prompted, type `Yes` and press Enter

### Issue 3: GPU Detection Warnings
**Problem**: Warnings about AMD GPU or unsupported graphics
**Solution**: These are normal warnings. Isaac Sim will use your NVIDIA GPU automatically.

### Issue 4: Symlink Permission Errors
**Problem**: `Administrator privilege required for this operation` when creating symlinks
**Solution**: Not required. Use Isaac Sim's Python directly instead of creating symlinks.

### Issue 5: Missing Simulation Components
**Problem**: `ModuleNotFoundError: No module named 'isaacsim.simulation_app'`
**Cause**: Using pip-installed Isaac Sim instead of official installer
**Solution**: Use the official Isaac Sim installer, not `pip install isaacsim`

## File Structure After Installation

```
D:\software\isaac_sim\
├── python.bat                    # Isaac Sim Python executable
├── isaac-sim.selector.bat        # Isaac Sim launcher
├── kit\                          # Isaac Sim core files
├── exts\                         # Extensions
└── ...

IsaacLab\
├── source\
│   ├── isaaclab\                 # Core Isaac Lab
│   ├── isaaclab_assets\          # Assets
│   ├── isaaclab_rl\              # RL components
│   └── isaaclab_tasks\           # Tasks
├── scripts\                      # Example scripts
└── apps\                         # Application configs
```

## Usage Examples

### Running Isaac Lab Scripts
```bash
# Navigate to Isaac Lab directory
cd IsaacLab

# Run demo script (headless)
& "D:\software\isaac_sim\python.bat" scripts/demos/quadcopter.py --headless

# Run with GUI
& "D:\software\isaac_sim\python.bat" scripts/demos/quadcopter.py
```

### Running Unitree RL Lab
```bash
# After installing Unitree RL Lab
& "D:\software\isaac_sim\python.bat" scripts/train.py --task=Isaac-Velocity-Flat-Unitree-Go2-v0
```

## System Requirements Verification

### GPU Information (Expected Output)
```
| GPU | Name                             | Active | GPU Memory |
| 0   | NVIDIA GeForce RTX 5070          | Yes: 0 | 11855   MB |
```

### CUDA Detection
```
[INFO][AppLauncher]: Using device: cuda:0
```

## Troubleshooting Checklist

- [ ] Isaac Sim GUI opens successfully
- [ ] Isaac Sim Python executable exists: `D:\software\isaac_sim\python.bat`
- [ ] Isaac Lab imports without errors
- [ ] CUDA device detected: `cuda:0`
- [ ] Demo script runs without import errors
- [ ] No conda environment conflicts

## Additional Resources

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab GitHub Repository](https://github.com/isaac-sim/IsaacLab)

## Notes

- **Python Version**: Isaac Sim 5.0.0 uses Python 3.11
- **CUDA Support**: Required for GPU acceleration
- **Memory Usage**: Isaac Sim can use significant GPU memory
- **First Launch**: May take 5-10 minutes for initialization
- **Updates**: Check NVIDIA's website for Isaac Sim updates

---

**Last Updated**: October 2025  
**Tested On**: Windows 11 Pro, RTX 5070, Isaac Sim 5.0.0, Isaac Lab 0.46.3
