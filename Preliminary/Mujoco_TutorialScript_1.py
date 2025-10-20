## Date: Oct 1, 2025
## This script 
## - simulates a box and sphere model

import time
import mujoco
import mujoco.viewer
# import mediapy as media

# # Model 1: box and sphere
# xml = """
# <mujoco>
#   <worldbody>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """

# Model 2: box and sphere with hinge joint
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

data = mujoco.MjData(model)

# This keeps the window open and updates it until you close it.
with mujoco.viewer.launch_passive(model, data) as v:
    # enable joint visualization option via viewer's option object
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    while v.is_running():
        mujoco.mj_step(model, data)  # advance physics
        v.sync()                      # render current state
        time.sleep(0.01)              # don't burn 100% CPU
