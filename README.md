# MAE589 Project - MuJoCo Physics Simulation Tutorials

**Course**: MAE589 - Fall 2025  
**Student**: [Your Name]  
**Repository**: https://github.com/GuanQinG-GitHub/MAE589-2025Fall-Project.git

## 📚 Project Overview

This repository contains a comprehensive collection of MuJoCo physics simulation tutorials and scripts, demonstrating various aspects of robotics simulation, physics modeling, and visualization techniques.

## 🎯 Learning Objectives

- Master MuJoCo physics engine fundamentals
- Implement interactive 3D simulations
- Create advanced visualization effects
- Develop control systems for robotic systems
- Generate video outputs for analysis and presentation

## 📁 Project Structure

```
MAE589_Project/
├── README.md                           # This file
├── Mujoco_beginner_tutorial.md         # Setup and installation guide
├── Mujoco_TutorialScript_1.py          # Basic box and sphere simulation
├── Mujoco_TutorialScript_2.py          # Tippe top with data collection and plotting
├── Mujoco_TutorialScript_3.py          # Multi-object system with contact visualization
├── Mujoco_TutorialScript_4.py          # Complex multi-body system with actuators and sensors
├── Mujoco_TutorialScript_5.py          # Advanced trajectory visualization with speed-based coloring
├── Mujoco_TutorialScript_6.py          # Humanoid robot simulation with ghost visualization
├── output_videos/                      # Generated video outputs
└── .gitignore                          # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- macOS (Apple Silicon M1/M2/M3)
- Python 3.11+
- Conda/Miniconda

### Installation

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
   ```

### Running the Scripts

All scripts should be run with `mjpython` (not regular `python`) to enable the viewer GUI:

```bash
# Activate the conda environment
conda activate mujoco_env

# Run any tutorial script
mjpython Mujoco_TutorialScript_1.py
mjpython Mujoco_TutorialScript_2.py
# ... etc
```

## 📖 Tutorial Scripts

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

### Output Generation
- **Interactive GUI**: Real-time 3D viewer
- **Video recording**: MP4 output with custom effects
- **Data logging**: CSV-compatible data collection
- **Plot generation**: Matplotlib visualizations

## 🛠️ Technical Details

### Dependencies
- `mujoco`: Physics simulation engine
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `opencv-python`: Video processing (optional)
- `mediapy`: Video display (optional)

### Platform Support
- **Primary**: macOS (Apple Silicon)
- **Compatible**: Linux, Windows (with modifications)
- **Viewer**: MuJoCo's native OpenGL viewer

## 📊 Results & Outputs

The project generates various outputs:
- **Interactive simulations**: Real-time 3D physics
- **Video files**: MP4 recordings with special effects
- **Data plots**: PNG files with analysis graphs
- **Documentation**: Comprehensive setup guides

## 🔧 Troubleshooting

### Common Issues

1. **"mjpython not found"**: Ensure MuJoCo is installed in the conda environment
2. **Viewer crashes**: Use `mjpython` instead of `python`
3. **Import errors**: Check all dependencies are installed
4. **Video not saving**: Install OpenCV or use matplotlib fallback

### Getting Help

- Check `Mujoco_beginner_tutorial.md` for detailed setup instructions
- Review individual script comments for specific functionality
- Consult MuJoCo documentation: https://mujoco.readthedocs.io/

## 📈 Future Enhancements

- [ ] Add more complex robot models
- [ ] Implement reinforcement learning examples
- [ ] Create interactive parameter tuning
- [ ] Add more visualization effects
- [ ] Develop custom control algorithms

## 📝 License

This project is for educational purposes as part of MAE589 coursework.

## 🤝 Contributing

This is a personal academic project, but suggestions and improvements are welcome!

---

**Note**: This project demonstrates proficiency in physics simulation, robotics visualization, and scientific computing using state-of-the-art tools and techniques.
