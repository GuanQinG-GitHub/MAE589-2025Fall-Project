## Visualizing the Unitree G1 Humanoid in MuJoCo (Beginner-Friendly Guide)

This guide shows you, step by step, how to download the required robot assets and run a ready-made Python script to open the model in the MuJoCo GUI. It assumes you’ve never used MuJoCo or the terminal before.

Reference model (for context): [Unitree G1 29 DoF XML](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_29dof.xml)

### What you will accomplish
- Download the official Unitree MuJoCo models into this project.
- Install the MuJoCo Python package.
- Run `Mujoco_TutorialScript_7_g1.py` to view the humanoid in a GUI window.

### Requirements
- A Mac with an internet connection.
- Python 3.9 or newer installed (macOS typically includes Python, but you can also install from the official Python website).
- Git installed (you can install via Xcode Command Line Tools or from `https://git-scm.com/downloads`).

If you are unsure whether Python or Git is installed, we’ll show you how to check.

---

## Step 1: Open the Terminal

- Click the Spotlight icon (magnifying glass) in the top-right of your screen.
- Type “Terminal” and press Enter.

You will see a window with a prompt like this:
```
MacBook:~ yourname$
```

This is where you will type commands.

---

## Step 2: Change directory to this project folder

Run this command to go to the project directory. Replace YourUserName with your macOS username if different.

```bash
cd /Users/xinlei/Documents/MAE589_Project
```

Explanation:
- `cd` means “change directory.”
- `/Users/xinlei/Documents/MAE589_Project` is the full path to your project folder.

Tip: If the path is different on your computer, you can drag the folder from Finder into the Terminal to paste its path.

---

## Step 3: (Optional) Create and activate a Python virtual environment

This keeps project dependencies isolated from the rest of your system.

```bash
python3 -m venv .venv
```
- `python3 -m venv .venv`: creates a new virtual environment in a folder named `.venv` inside your project.

```bash
source .venv/bin/activate
```
- `source .venv/bin/activate`: turns on the virtual environment so that `python` and `pip` refer to this isolated environment.

If you want to turn it off later, type `deactivate`.

---

## Step 4: Install the MuJoCo Python package

```bash
pip install --upgrade pip
```
- Updates `pip` (the Python package installer) to the latest version.

```bash
pip install mujoco
```
- Installs the `mujoco` Python package which includes the physics engine and a built-in GUI viewer for macOS.

If installation fails, ensure your internet connection is working and try again.

---

## Step 5: Download the Unitree MuJoCo repository (robot models and meshes)

Run the following commands (you only need to do this once or when updating models):

```bash
mkdir -p external
```
- `mkdir -p external`: creates a folder named `external` if it doesn’t exist.

```bash
cd external
```
- `cd external`: moves into the `external` folder.

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git
```
- `git clone …`: downloads the official Unitree MuJoCo files into a folder called `unitree_mujoco`.
- `--depth 1` downloads only the latest snapshot to save time and space.

```bash
cd ..
```
- `cd ..`: go back to the project root folder.

After this, the G1 model XML and meshes will be here:
```
/Users/xinlei/Documents/MAE589_Project/external/unitree_mujoco/unitree_robots/g1/
```

---

## Step 6: Run the visualization script

From the project root (`/Users/xinlei/Documents/MAE589_Project`), run:

```bash
python Mujoco_TutorialScript_7_g1.py
```

What this does:
- `python`: runs the Python interpreter.
- `Mujoco_TutorialScript_7_g1.py`: runs the provided script that opens the MuJoCo GUI.

What the script does internally (plain-English explanation):
- It looks for the locally cloned Unitree repo at `external/unitree_mujoco`.
- It prefers to load `scene_23dof.xml` (which includes a ground plane), falling back to `g1_29dof.xml` if the scene is missing.
- It builds a MuJoCo model from that XML and opens a viewer window.
- It starts stepping the physics and keeps the window open until you close it.

When the viewer opens:
- You should see the humanoid standing on a ground plane.
- If the robot falls immediately, ensure the script is loading the scene XML (`scene_23dof.xml`) and that the assets exist in `external/unitree_mujoco/unitree_robots/g1/`.

Close the window to stop the script, or press `Ctrl + C` in the Terminal.

---

## Troubleshooting

- If you see an error like “No module named mujoco”: run `pip install mujoco` (inside your virtual environment if you created one).
- If the viewer window does not open: ensure you’re on macOS with GUI access (not an SSH session). Try updating MuJoCo: `pip install --upgrade mujoco`.
- If files are missing: re-run the clone command or check that this folder exists: `/Users/xinlei/Documents/MAE589_Project/external/unitree_mujoco/unitree_robots/g1/`.
- If the robot appears without ground: make sure `scene_23dof.xml` is present in the `g1` folder. The script prefers that scene file because it includes a ground plane.

---

## How to update the Unitree models later

If you want the latest models in the future, run:

```bash
cd /Users/xinlei/Documents/MAE589_Project/external/unitree_mujoco
git fetch --depth 1 origin
git reset --hard origin/main
```

Explanation:
- `git fetch --depth 1 origin`: downloads the latest changes.
- `git reset --hard origin/main`: updates your local copy to match the latest remote version.

Then run the viewer again from the project root:

```bash
cd /Users/xinlei/Documents/MAE589_Project
python Mujoco_TutorialScript_7_g1.py
```

---

You’re done! You now have a working MuJoCo setup that can render the Unitree G1 humanoid using the official assets.



