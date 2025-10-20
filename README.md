# MAE589 Project - MuJoCo Physics Simulation & Unitree RL Training

**Course**: MAE589 - Fall 2025  
**Student**: [Your Name]  
**Repository**: https://github.com/GuanQinG-GitHub/MAE589-2025Fall-Project.git

## 📚 Project Overview

xx
## 🎯 Learning Objectives

xx

## 🏃 Quick Start: G1 Pre-trained Policy (Self-contained)

### 1) Flat Ground Test
```bash
python g1_pretrained_testing.py
```
- Loads `robot_models/g1_description/g1_12dof.xml`
- Uses policy at `trained_models/motion.pt`
- Test scene at `terrains/scene.xml` which includes a flat ground

### 2) Perlin Terrain Test
```bash
python g1_pretrained_perlin_terrain.py
```
- Uses the same robot and policy
- Adds a Perlin heightfield from `terrains/g1_perlin_terrain_1.png`
- Heightfield scaling is controlled in the script’s inline XML via the `<hfield size="L W Hmax Hmin" />` parameters

### 3) Generate/Update Perlin Terrain
```bash
python terrains/g1_perlin_generator.py
```
- Regenerates `terrains/g1_perlin_terrain_1.png` with new random Perlin noise
- Then, repeat step 2 again

Notes:
- To change the terrain random pattern: rerun the generator.
- To change terrain height or position: edit the inline XML in `g1_pretrained_perlin_terrain.py` (hfield `size` for heights; geom `pos` for placement). See `terrains/G1_Perlin_Terrain_Complete_Guide.md` for details.

## 📁 Project Structure

MAE589_Project/
├── README.md                           # This file
├── VERSION_CONTROL.md                  # Version control strategy
├── guidance.md                         # Windows RL setup guide
├── Preliminary/                        # Tutorial scripts and beginner materials
│   ├── Mujoco_beginner_tutorial.md
│   ├── Mujoco_TutorialScript_1.py ...  # Scripts 1–7 (moved here)
│   ├── tippe_top_metrics.png
│   └── trajectory_visualization_*.mp4
├── robot_models/                       # Local Unitree G1 models (with meshes)
│   └── g1_description/
├── terrains/                           # Terrain generator and assets
│   ├── scene.xml                       # Model-independent base scene
│   ├── g1_perlin_generator.py          # Generates Perlin heightmap PNGs
│   └── G1_Perlin_Terrain_Complete_Guide.md
├── trained_models/                     # Pre-trained policies
│   └── motion.pt
├── g1_pretrained_testing.py            # Flat-ground test (self-contained)
├── g1_pretrained_perlin_terrain.py     # Perlin terrain test (self-contained)
├── unitree_rl_lab/                     # Unitree RL Lab repository (submodule)
└── .gitignore
```