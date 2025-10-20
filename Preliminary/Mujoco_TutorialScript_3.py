## Date: Oct 1, 2025
## This script 
## - simulates a tippe-top model and plots the angular velocity and stem height over time
## - saves the plot to a file

import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

free_body_MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
    rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
    reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1" mode="trackcom"/>
    <geom name="ground" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
    <body name="box_and_sphere" pos="0 0 0">
      <freejoint/>
      <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1" solimp=".99 .99 .01"  solref=".001 1"/>
      <geom name="green_sphere" size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/>
      <camera name="fixed" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2"/>
      <camera name="track" pos="0 -.6 .3" xyaxes="1 0 0 0 1 2" mode="track"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(free_body_MJCF)
data = mujoco.MjData(model)

# Run a single forward pass to initialize the simulation state
mujoco.mj_forward(model, data)

# Reset the state to keyframe 0 so the top starts spinning/moving
# mujoco.mj_resetDataKeyframe(model, data, 0)

# random initial rotational velocity:
mujoco.mj_resetData(model, data)
data.qvel[3:6] = 5*np.random.randn(3)

# This keeps the window open and updates it until you close it.
with mujoco.viewer.launch_passive(model, data) as v:
    # visualize contact frames and forces, make body transparent
    # Apply options directly to the viewer's options object
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # tweak scales of contact visualization elements
    model.vis.scale.contactwidth = 0.1
    model.vis.scale.contactheight = 0.03
    model.vis.scale.forcewidth = 0.05
    model.vis.map.force = 0.3

    while v.is_running():
        mujoco.mj_step(model, data)  # advance physics

        v.sync()                      # render current state
        time.sleep(0.01)              # don't burn 100% CPU