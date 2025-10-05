# MAE589 Project - MuJoCo Physics Simulation & Unitree RL Training

**Course**: MAE589 - Fall 2025  
**Student**: [Your Name]  
**Repository**: https://github.com/GuanQinG-GitHub/MAE589-2025Fall-Project.git

## 📚 Project Overview

This repository contains two main components:
1. **MuJoCo Physics Simulation Tutorials** - Comprehensive collection of MuJoCo physics simulation scripts
2. **Unitree RL Training Setup** - Reinforcement learning environment for Unitree robot training

## 🎯 Learning Objectives

- Master MuJoCo physics engine fundamentals
- Implement interactive 3D simulations
- Create advanced visualization effects
- Develop control systems for robotic systems
- Set up reinforcement learning environments for robot training
- Generate video outputs for analysis and presentation

## 📁 Project Structure

```
MAE589_Project/
├── README.md                           # This file
├── VERSION_CONTROL.md                  # Version control strategy
├── guidance.md                         # Windows RL setup guide
├── Mujoco_beginner_tutorial.md         # MuJoCo setup and installation guide
├── Mujoco_TutorialScript_1.py          # Basic box and sphere simulation
├── Mujoco_TutorialScript_2.py          # Tippe top with data collection and plotting
├── Mujoco_TutorialScript_3.py          # Multi-object system with contact visualization
├── Mujoco_TutorialScript_4.py          # Complex multi-body system with actuators and sensors
├── Mujoco_TutorialScript_5.py          # Advanced trajectory visualization with speed-based coloring
├── Mujoco_TutorialScript_6.py          # Humanoid robot simulation with ghost visualization
├── unitree_rl_lab/                     # Unitree RL Lab repository (submodule)
│   ├── source/                         # RL training source code
│   ├── scripts/                        # Training and testing scripts
│   └── unitree_model/                  # Robot models (submodule)
├── output_videos/                      # Generated video outputs
└── .gitignore                          # Cross-platform ignore rules
```

## 🚀 Getting Started

### Platform Support

- **macOS (Apple Silicon)**: Primary platform for MuJoCo tutorials and development
- **Windows 11**: Primary platform for RL training with GPU acceleration

### MuJoCo Tutorials Setup (macOS)

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Miniforge**:
   ```bash
   brew install --cask miniforge
   conda init zsh
   source ~/.zshrc
   ```

3. **Create MuJoCo environment**:
   ```bash
   conda create -n mujoco_env python=3.11
   conda activate mujoco_env
   pip install mujoco
   pip install matplotlib numpy opencv-python
   ```

4. **Clone this repository**:
   ```bash
   git clone https://github.com/GuanQinG-GitHub/MAE589-2025Fall-Project.git
   cd MAE589-2025Fall-Project
   git submodule update --init --recursive
   ```

### Unitree RL Training Setup (Windows 11)

Follow the detailed guide in `guidance.md` for:
1. Python environment setup (Anaconda)
2. MuJoCo installation
3. Unitree RL Lab setup
4. Isaac Lab/Isaac Sim installation

## 📖 MuJoCo Tutorial Scripts

### 1. **Mujoco_TutorialScript_1.py** - Basic Simulation
- Simple box and sphere with hinge joint
- Interactive viewer with joint visualization
- Foundation for understanding MuJoCo basics

### 2. **Mujoco_TutorialScript_2.py** - Data Collection & Analysis
- Tippe top simulation with physics
- Real-time data collection (angular velocity, height)
- Matplotlib plotting and visualization
- PNG output for analysis

### 3. **Mujoco_TutorialScript_3.py** - Contact Visualization
- Multi-object system with contact forces
- Transparent objects for better visualization
- Contact point and force vector display
- Interactive GUI controls

### 4. **Mujoco_TutorialScript_4.py** - Complex Systems
- Pendulum bat with motor control
- Free-flying object with tendon constraints
- Accelerometer sensor integration
- Comprehensive physics simulation

### 5. **Mujoco_TutorialScript_5.py** - Advanced Visualization
- Speed-based trajectory coloring
- Real-time trail generation
- 3D spiral effects
- Video output with OpenCV/Matplotlib

### 6. **Mujoco_TutorialScript_6.py** - Humanoid Simulation
- Standard MuJoCo humanoid model
- Ghost visualization techniques
- Dual control systems
- Interactive humanoid robotics

## 🤖 Unitree RL Training

### Available Robots
- **Go2**: Quadruped robot
- **H1**: Humanoid robot
- **G1-29dof**: Advanced humanoid with 29 degrees of freedom

### Training Commands
```bash
# List available environments
python scripts/list_envs.py

# Train a policy
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# Test trained policy
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

## 🎨 Key Features

### Visualization Techniques
- **Speed-based coloring**: Green (slow) to Red (fast) trails
- **Contact force visualization**: Real-time force vectors
- **Transparency effects**: Ghost objects and overlays
- **Joint visualization**: Skeleton and joint axes
- **Trajectory trails**: Motion history visualization

### Control Systems
- **Sinusoidal control**: Natural movement patterns
- **Dual control**: Independent control of multiple objects
- **Real-time interaction**: Live parameter adjustment
- **Sensor integration**: IMU and accelerometer data
- **Reinforcement learning**: PPO-based policy training

### Output Generation
- **Interactive GUI**: Real-time 3D viewer
- **Video recording**: MP4 output with custom effects
- **Data logging**: CSV-compatible data collection
- **Plot generation**: Matplotlib visualizations
- **Policy checkpoints**: Trained RL models

## 🛠️ Technical Details

### Dependencies
- `mujoco`: Physics simulation engine
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `opencv-python`: Video processing (optional)
- `mediapy`: Video display (optional)
- `isaaclab`: NVIDIA Isaac Lab for RL training
- `rsl_rl`: Reinforcement learning algorithms

### Platform Support
- **macOS**: MuJoCo tutorials and development
- **Windows 11**: RL training with GPU acceleration
- **Viewer**: MuJoCo's native OpenGL viewer

## 📊 Results & Outputs

The project generates various outputs:
- **Interactive simulations**: Real-time 3D physics
- **Video files**: MP4 recordings with special effects
- **Data plots**: PNG files with analysis graphs
- **Documentation**: Comprehensive setup guides
- **Trained policies**: RL models for robot control

## 🔧 Troubleshooting

### Common Issues

1. **"mjpython not found"**: Ensure MuJoCo is installed in the conda environment
2. **Viewer crashes**: Use `mjpython` instead of `python`
3. **Import errors**: Check all dependencies are installed
4. **Video not saving**: Install OpenCV or use matplotlib fallback
5. **Isaac Lab not found**: Follow the Isaac Lab installation guide

### Getting Help

- Check `Mujoco_beginner_tutorial.md` for MuJoCo setup instructions
- Check `guidance.md` for Windows RL training setup
- Review individual script comments for specific functionality
- Consult MuJoCo documentation: https://mujoco.readthedocs.io/
- Consult Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/

## 📈 Future Enhancements

- [x] Add Unitree robot models and RL training
- [ ] Implement more complex robot models
- [ ] Create interactive parameter tuning
- [ ] Add more visualization effects
- [ ] Develop custom control algorithms
- [ ] Sim-to-real transfer learning

## 📝 License

This project is for educational purposes as part of MAE589 coursework.

## 🤝 Contributing

This is a personal academic project, but suggestions and improvements are welcome!

---

**Note**: This project demonstrates proficiency in physics simulation, robotics visualization, reinforcement learning, and scientific computing using state-of-the-art tools and techniques.
