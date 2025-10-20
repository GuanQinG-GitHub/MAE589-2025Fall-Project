## Date: Oct 1, 2025
## This script 
## - simulates a multi-object system with a pendulum bat, free-flying object, and connecting wire
## - demonstrates actuators, sensors, tendons, and complex interactions

# Import required libraries
import time                    # For controlling simulation timing
import mujoco                  # MuJoCo physics engine
import mujoco.viewer           # MuJoCo interactive viewer
import numpy as np             # Numerical computing
import matplotlib.pyplot as plt # Plotting (not used in this script but imported for potential use)

# Define the MuJoCo model using MJCF (MuJoCo XML) format
MJCF = """
<mujoco>
  <!-- Asset definitions: textures and materials for visual appearance -->
  <asset>
    <!-- Create a checkerboard grid texture for the floor -->
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <!-- Apply the grid texture as a material with specific properties -->
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <!-- World body: contains all physical objects in the simulation -->
  <worldbody>
    <!-- Lighting setup: single light source positioned above the scene -->
    <light name="light" pos="0 0 1"/>
    
    <!-- Ground plane: flat surface for objects to rest on -->
    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
    
    <!-- Anchor point: fixed site where the wire/tendon is attached -->
    <site name="anchor" pos="0 0 .3" size=".01"/>
    
    <!-- Camera: positioned to view the scene from a good angle -->
    <camera name="fixed" pos="0 -1.3 .5" xyaxes="1 0 0 0 1 2"/>

    <!-- Pendulum system: pole and swinging bat -->
    <!-- Vertical pole: cylinder extending from floor to bat attachment point -->
    <geom name="pole" type="cylinder" fromto=".3 0 -.5 .3 0 -.1" size=".04"/>
    
    <!-- Bat body: can swing around the pole via hinge joint -->
    <body name="bat" pos=".3 0 -.1">
      <!-- Hinge joint: allows rotation around z-axis with damping -->
      <joint name="swing" type="hinge" damping="1" axis="0 0 1"/>
      <!-- Bat geometry: blue capsule representing the swinging bat -->
      <geom name="bat" type="capsule" fromto="0 0 .04 0 -.3 .04"
       size=".04" rgba="0 0 1 1"/>
    </body>

    <!-- Free-flying object: box with sphere attached -->
    <body name="box_and_sphere" pos="0 0 0">
      <!-- Free joint: allows complete 6DOF motion (3 translation + 3 rotation) -->
      <joint name="free" type="free"/>
      <!-- Red box: main body of the free-flying object -->
      <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
      <!-- Green sphere: attached to the box at an offset position -->
      <geom name="green_sphere"  size=".06" pos=".1 .1 .1" rgba="0 1 0 1"/>
      <!-- Hook site: attachment point for the wire/tendon -->
      <site name="hook" pos="-.1 -.1 -.1" size=".01"/>
      <!-- IMU site: where accelerometer sensor is placed -->
      <site name="IMU"/>
    </body>
  </worldbody>

  <!-- Tendon system: flexible wire connecting anchor to free-flying object -->
  <tendon>
    <!-- Spatial tendon: 3D flexible connection between two sites -->
    <spatial name="wire" limited="true" range="0 0.35" width="0.003">
      <site site="anchor"/>  <!-- Fixed anchor point -->
      <site site="hook"/>    <!-- Moving hook on the free object -->
    </spatial>
  </tendon>

  <!-- Actuator: motor that drives the pendulum bat -->
  <actuator>
    <!-- Motor: applies torque to the swing joint with gear ratio of 1 -->
    <motor name="my_motor" joint="swing" gear="1"/>
  </actuator>

  <!-- Sensor: accelerometer measuring acceleration at the IMU site -->
  <sensor>
    <!-- Accelerometer: measures 3D acceleration vector at the IMU site -->
    <accelerometer name="accelerometer" site="IMU"/>
  </sensor>
</mujoco>
"""

# Create MuJoCo model from the XML string
model = mujoco.MjModel.from_xml_string(MJCF)
# Create data structure to hold simulation state (positions, velocities, etc.)
data = mujoco.MjData(model)

# Run a single forward pass to initialize the simulation state
# This computes initial accelerations, contact forces, etc.
mujoco.mj_forward(model, data)

# Set up initial conditions and control input
# Reset all data to initial state (positions, velocities, etc.)
mujoco.mj_resetData(model, data)
# Apply constant control signal to the motor (40 units of torque)
data.ctrl = 40

# Launch interactive viewer and run simulation loop
with mujoco.viewer.launch_passive(model, data) as v:
    # Optional: visualize contact frames and forces, make body transparent
    # (Currently commented out - uncomment to see contact visualization)
    # Apply options directly to the viewer's options object
    # v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Optional: tweak scales of contact visualization elements
    # (Currently commented out - uncomment to adjust contact visualization)
    # model.vis.scale.contactwidth = 0.1
    # model.vis.scale.contactheight = 0.03
    # model.vis.scale.forcewidth = 0.05
    # model.vis.map.force = 0.3

    # Main simulation loop: continues until viewer window is closed
    while v.is_running():
        # Advance physics simulation by one timestep
        # This updates positions, velocities, accelerations, and contact forces
        mujoco.mj_step(model, data)

        # Update the visual display with current simulation state
        v.sync()
        # Pause briefly to control simulation speed and reduce CPU usage
        time.sleep(0.01)