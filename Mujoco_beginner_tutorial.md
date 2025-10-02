## MuJoCo Setup on macOS (Apple Silicon, e.g., M3 Pro)
Oct. 1, 2025

### Overview
This document records all steps for installing and testing MuJoCo on macOS with Apple Silicon M3 from very beginning. It uses:

- **Homebrew**: package manager for macOS.
- **Miniforge (Conda)**: environment manager for Python.
- **MuJoCo**: physics engine installed via `pip`.

This tutorial creates an isolated Python environment to avoid conflicts with system Python.

---

### Install and Initialize Necessary Ingredients

#### 1. Install Homebrew (Skip if already have Homebrew)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

---

#### 2. Install Miniforge (Conda)
```bash
brew install --cask miniforge
```

Miniforge installs Conda at:
```
/opt/homebrew/Caskroom/miniforge/base
```

---

#### 3. Initialize Conda for zsh
```bash
conda init zsh
source ~/.zshrc
```

---

### Create and Activate a Conda Environment
```bash
conda create -n mujoco_env python=3.11
conda activate mujoco_env
```

---

### Explanation: Base vs Custom Environments
- **System (no conda)**: Your Mac’s default Python (often `/usr/bin/python3` or `/opt/homebrew/bin/python3`).
- **Base environment**: When you install Miniforge, it creates a default Conda environment called `base`, located at `/opt/homebrew/Caskroom/miniforge/base`. This is activated by default when you open a terminal (after `conda init`).
- **Custom environments**: e.g., `mujoco_env`. These are separate and isolated from base.

Switching hierarchy:

- From system → base:
  ```bash
  conda activate base
  ```
- From base → mujoco_env:
  ```bash
  conda activate mujoco_env
  ```
- To deactivate back step-by-step:
  ```bash
  conda deactivate   # mujoco_env -> base
  conda deactivate   # base -> system
  ```
- From system → mujoco_env:
  ```
  conda activate mujoco_env # when you are in system env
  ```

---

### Install MuJoCo
```bash
pip install mujoco
```

---

### Test MuJoCo
Create `test_mj.py`:
```python
# test_mj.py
import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
    <body>
      <geom name="box" type="box" size="0.1 0.1 0.1" rgba="0.9 0.3 0.3 1"/>
      <joint type="free"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# This keeps the window open and updates it until you close it.
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)  # advance physics
        v.sync()                      # render current state
        time.sleep(0.01)              # don't burn 100% CPU
```

Run with:
```bash
mjpython test_mj.py
```

Note: On macOS, use `mjpython` (not `python`) to enable the viewer GUI.

---

### Environment Management
- **List installed packages**:
  ```bash
  pip list
  ```
- **Leave current env**:
  ```bash
  conda deactivate
  ```
- **Remove env completely**:
  ```bash
  conda remove -n mujoco_env --all
  ```

---