## Date: Oct 1, 2025
## This script 
## - simulates box-sphere system with fancy trajectory visualizations
## - creates speed-based colored trails that show motion history

import time
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os

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
# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Helper function to calculate the speed of a geometry
def get_geom_speed(model, data, geom_name):
    """Returns the speed of a geom."""
    geom_vel = np.zeros(6)  # 6D velocity vector (3 linear + 3 angular)
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    geom_id = data.geom(geom_name).id
    # Get the 6D velocity of the geometry
    mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
    # Return the magnitude of the velocity vector
    return np.linalg.norm(geom_vel)

# Helper function to add visual capsules to the scene
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return  # Don't add if scene is full
    scene.ngeom += 1  # increment ngeom counter
    # Initialize a new capsule geometry
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    # Create connector between two points
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                         mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         point1, point2)

# Function to modify the scene with trajectory visualization
def modify_scene(scn):
    """Draw position trace, speed modifies width and colors."""
    if len(positions) > 1:
        for i in range(len(positions)-1):
            # Speed-based color: red for fast, green for slow
            rgba = np.array((np.clip(speeds[i]/10, 0, 1),      # Red component (faster = more red)
                            np.clip(1-speeds[i]/10, 0, 1),     # Green component (faster = less green)
                            .5,                                # Blue component (constant)
                            1.))                               # Alpha (opacity)
            
            # Speed-based radius: faster motion = thicker trail
            radius = .003*(1+speeds[i])
            
            # Add time-based offset to create 3D spiral effect
            point1 = positions[i] + offset*times[i]
            point2 = positions[i+1] + offset*times[i+1]
            
            # Add the capsule segment to the scene
            add_visual_capsule(scn, point1, point2, radius, rgba)


# Initialize trajectory tracking variables
times = []          # Store time stamps
positions = []      # Store positions of the green sphere
speeds = []         # Store speeds at each position
offset = model.jnt_axis[0]/16  # Offset along the joint axis for 3D effect

# Initialize simulation
mujoco.mj_forward(model, data)

# Use offline renderer for efficient custom visualization
duration = 8.0  # Simulation duration in seconds
framerate = 30  # Frames per second
frames = []

# Reset simulation state
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("Starting simulation with trajectory visualization...")
print("Speed-based coloring: Green = slow, Red = fast, Thickness = speed")

# Create renderer for offline rendering with trajectory visualization
with mujoco.Renderer(model, height=480, width=640) as renderer:
    frame_count = 0
    while data.time < duration:
        # Record current state for trajectory visualization
        positions.append(data.geom_xpos[data.geom("green_sphere").id].copy())
        times.append(data.time)
        speeds.append(get_geom_speed(model, data, "green_sphere"))
        
        # Advance physics simulation
        mujoco.mj_step(model, data)
        
        # Render frame with trajectory visualization
        renderer.update_scene(data)
        modify_scene(renderer.scene)
        
        # Capture frame if it's time for the next frame
        if len(frames) < data.time * framerate:
            pixels = renderer.render()
            frames.append(pixels)
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every second
                print(f"Rendered {frame_count} frames at t={data.time:.2f}s")

print(f"\nSimulation complete! Generated {len(frames)} frames.")
print("Trajectory visualization features:")
print("- 🟢 Green trails = slow motion")
print("- 🔴 Red trails = fast motion") 
print("- 📏 Thickness = speed magnitude")
print("- 🌪️ 3D spiral effect = time progression")

# Save video using OpenCV
if len(frames) > 0:
    # Create output directory if it doesn't exist
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define video filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{output_dir}/trajectory_visualization_{timestamp}.mp4"
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, framerate, (width, height))
    
    print(f"\nSaving video to: {video_filename}")
    
    # Write frames to video
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
        
        # Show progress
        if (i + 1) % 30 == 0:
            print(f"Writing frame {i+1}/{len(frames)}")
    
    # Release video writer
    video_writer.release()
    
    print(f"✅ Video saved successfully!")
    print(f"📁 Location: {os.path.abspath(video_filename)}")
    print(f"📊 Video specs: {width}x{height}, {len(frames)} frames, {framerate} FPS")
    
    # Calculate video duration
    duration_seconds = len(frames) / framerate
    print(f"⏱️  Duration: {duration_seconds:.2f} seconds")
    
else:
    print("❌ No frames generated to save video.")
