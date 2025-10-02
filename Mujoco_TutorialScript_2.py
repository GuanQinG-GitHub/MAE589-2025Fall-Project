## Date: Oct 1, 2025
## This script 
## - creates a tippe-top model and simulates its action for 5 seconds
## - plots the angular velocity and stem height over time
## - saves the plot to a file

import time
import matplotlib
# Use a non-GUI backend to avoid macOS NSWindow-on-non-main-thread crashes under mjpython
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(tippe_top)
data = mujoco.MjData(model)

# Run a single forward pass to initialize the simulation state
mujoco.mj_forward(model, data)

# Reset the state to keyframe 0 so the top starts spinning/moving
mujoco.mj_resetDataKeyframe(model, data, 0)

# Prepare measurement arrays
timevals = []
angular_velocity = []  # base body's angular velocity components (x,y,z)
stem_height = []       # z position of the 'stem' geom

# Simulation duration for data collection (seconds)
duration = 5.0

# Resolve the geom id for 'stem' to read its world position
stem_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stem")

# This keeps the window open and updates it until you close it.
with mujoco.viewer.launch_passive(model, data) as v:
    # enable joint visualization option via viewer's option object (optional)
    # v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    start_time = data.time
    while v.is_running():
        mujoco.mj_step(model, data)  # advance physics

        # record measurements
        timevals.append(data.time)
        angular_velocity.append(data.qvel[3:6].copy())
        stem_height.append(float(data.geom_xpos[stem_geom_id, 2]))

        v.sync()                      # render current state
        time.sleep(0.01)              # don't burn 100% CPU

        # stop after duration seconds of simulated time
        if data.time - start_time >= duration:
            break

# Plot measurements after the viewer closes or duration reached
dpi = 120  # Set the dots per inch for the figure (controls resolution)
width = 600  # Set the width of the figure in pixels
height = 800  # Set the height of the figure in pixels
figsize = (width / dpi, height / dpi)  # Calculate figure size in inches for matplotlib

# Create a figure with 2 vertically stacked subplots, sharing the x-axis
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

# Plot angular velocity (all components) vs time on the first subplot, with legends
angular_velocity = np.array(angular_velocity)  # Ensure it's a numpy array for column slicing
labels = ['x', 'y', 'z']
for i in range(3):
    ax[0].plot(timevals, angular_velocity[:, i], label=labels[i])
ax[0].set_title('angular velocity')  # Set the title for the first subplot
ax[0].set_ylabel('radians / second')  # Label the y-axis for angular velocity
ax[0].legend()  # Add legend for angular velocity components

# Plot stem height vs time on the second subplot
ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')  # Label the x-axis for time
ax[1].set_ylabel('meters')  # Label the y-axis for stem height
_ = ax[1].set_title('stem height')  # Set the title for the second subplot

plt.tight_layout()  # Adjust subplot parameters for a clean layout
plt.savefig('tippe_top_metrics.png', dpi=dpi)  # Save the figure to a file with the specified dpi

