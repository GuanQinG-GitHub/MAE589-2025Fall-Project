# Trained Unitree G1-29dof Models

## 🎯 **Model Checkpoints**

This directory contains key model checkpoints from the successful training run of the Unitree G1-29dof humanoid robot.

### **📊 Available Models:**

| Model File | Iterations | Size | Training Time | Description |
|------------|------------|------|---------------|-------------|
| `model_1000.pt` | 1,000 | 10.0 MB | ~1 hour | Early training checkpoint |
| `model_10000.pt` | 10,000 | 10.0 MB | ~7 hours | Mid-training checkpoint |
| `model_25000.pt` | 25,000 | 10.0 MB | ~19 hours | Advanced training checkpoint |
| `model_final.pt` | 49,200 | 10.0 MB | ~37 hours | **Final trained model** |

### **🚀 How to Use:**

#### **Load and Test a Model:**
```python
import torch
from isaaclab.envs import ManagerBasedRLEnv
import unitree_rl_lab.tasks

# Load the trained model
model = torch.load('model_final.pt')

# Create environment
env = ManagerBasedRLEnv(cfg=env_cfg)

# Use the model for inference
obs, _ = env.reset()
action = model['actor'](obs)
```

#### **Resume Training:**
```bash
# Resume training from a checkpoint
& "D:\software\isaac_sim\python.bat" scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity \
  --load_run 2025-10-05_04-03-22 \
  --load_checkpoint model_25000.pt
```

#### **Deploy to Real Robot:**
```bash
# Use the final model for deployment
python scripts/deploy/deploy.py \
  --task Unitree-G1-29dof-Velocity \
  --load_run 2025-10-05_04-03-22 \
  --load_checkpoint model_final.pt
```

### **📈 Training Progress:**

- **Total Training Time**: 37 hours
- **Total Iterations**: 49,200
- **Environments**: 4096 parallel environments
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Neural Network**: 512→256→128 (Actor/Critic)

### **🎯 Performance Metrics:**

The final model (`model_final.pt`) achieved:
- ✅ Stable walking behavior
- ✅ Velocity tracking capability
- ✅ Robust locomotion on various terrains
- ✅ Energy-efficient movement patterns

### **💾 Model Details:**

- **Architecture**: Actor-Critic with ELU activation
- **Input**: 480-dimensional observation space
- **Output**: 29-dimensional action space (joint positions)
- **Training Device**: NVIDIA GeForce RTX 5070 (12GB VRAM)
- **Framework**: Isaac Sim 5.0.0 + Isaac Lab + RSL-RL

### **🔧 Requirements:**

- Isaac Sim 5.0.0
- Isaac Lab 2.2.0
- RSL-RL 3.1.0
- PyTorch 2.7.0
- CUDA 12.8

---
**Training completed successfully on October 6, 2025** 🎉
