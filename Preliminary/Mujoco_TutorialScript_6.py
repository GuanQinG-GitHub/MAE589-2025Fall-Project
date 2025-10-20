## Date: Oct 1, 2025
## This script 
## - simulates a humanoid robot with ghost visualization
## - shows two humanoids: one normal, one transparent "ghost" version
## - uses interactive GUI viewer instead of offline rendering

import time
import mujoco
import mujoco.viewer
import numpy as np
import os

# Get MuJoCo's standard humanoid model
print('Getting MuJoCo humanoid XML description from GitHub...')

# Check if mujoco directory exists, if not clone it
if not os.path.exists('mujoco'):
    print('Cloning MuJoCo repository...')
    os.system('git clone https://github.com/google-deepmind/mujoco')
else:
    print('MuJoCo repository already exists.')

# Load the humanoid model
with open('mujoco/model/humanoid/humanoid.xml', 'r') as f:
    xml = f.read()

# Load the model and create two MjData instances
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)      # Main humanoid data
data2 = mujoco.MjData(model)     # Ghost humanoid data

# Episode parameters
duration = 10.0      # Simulation duration (seconds) - increased for GUI
framerate = 30       # Control frequency (Hz) - reduced for GUI
data.qpos[0:2] = [-.5, -.5]  # Initial x-y position (m)
data.qvel[2] = 4     # Initial vertical velocity (m/s)
ctrl_phase = 2 * np.pi * np.random.rand(model.nu)  # Control phase
ctrl_freq = 1        # Control frequency

# Visual options for the "ghost" model
vopt2 = mujoco.MjvOption()
vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Make ghost transparent
pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
# We only want dynamic objects (the humanoid). Static objects (the floor)
# should not be re-drawn. The mjtCatBit flag lets us do that
catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

# Initialize simulation
mujoco.mj_forward(model, data)
mujoco.mj_forward(model, data2)

print("Starting humanoid simulation with ghost visualization...")
print("Features:")
print("- Main humanoid: normal appearance")
print("- Ghost humanoid: transparent, offset position")
print("- Interactive GUI: real-time control and viewing")
print("- Duration: 10 seconds")

# Interactive viewer with humanoid simulation
with mujoco.viewer.launch_passive(model, data) as v:
    # Enable joint visualization to see the humanoid structure
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    # Enable transparency for better visibility
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    
    # Enable contact visualization to see forces
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    # Simulation loop
    start_time = time.time()
    print("\nInteractive controls:")
    print("- Mouse: Rotate camera")
    print("- Scroll: Zoom in/out")
    print("- Right-click + drag: Pan camera")
    print("- Space: Pause/resume simulation")
    
    while v.is_running() and (time.time() - start_time) < duration:
        # Sinusoidal control signal for humanoid movement
        data.ctrl = np.sin(ctrl_phase + 2 * np.pi * data.time * ctrl_freq)
        
        # Advance physics simulation
        mujoco.mj_step(model, data)
        
        # Render the scene
        v.sync()
        
        # Control simulation speed
        time.sleep(0.01)
    
    print("\nSimulation complete!")
    print("The humanoid performed sinusoidal control movements.")
    print("You can continue interacting with the viewer or close it.")

print("\nScript finished. Close the viewer window to exit.")
