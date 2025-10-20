"""
G1 Perlin Terrain Testing Script

This script tests the G1 humanoid robot on Perlin noise-based terrain.
The terrain features natural height variations generated using Perlin noise algorithms.

Features:
- Single Perlin terrain area with moderate complexity
- Pure height field-based natural terrain variations
- Tests robot's ability to navigate uneven terrain
- Performance monitoring and success zone tracking
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import torch

def load_g1_config():
    """Load G1 configuration parameters."""
    config = {
        # Model paths
        "policy_path": "trained_models/motion.pt",
        "robot_xml_path": "robot_models/g1_description/g1_12dof.xml",
        
        # Simulation parameters
        "simulation_duration": 60.0,  # seconds
        "simulation_dt": 0.002,       # time step
        "control_decimation": 10,     # control frequency = 50Hz
        
        # PD control gains (12 joints: 6 per leg)
        "kps": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
        "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
        
        # Default joint angles (standing pose)
        "default_angles": np.array([
            -0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # Left leg
            -0.1,  0.0,  0.0,  0.3, -0.2, 0.0   # Right leg
        ], dtype=np.float32),
        
        # Scaling factors
        "ang_vel_scale": 0.25,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "action_scale": 0.25,
        "cmd_scale": np.array([2.0, 2.0, 0.25], dtype=np.float32),
        
        # Network parameters
        "num_actions": 12,  # 12 leg joints
        "num_obs": 47,      # observation space size
        
        # Initial command (forward velocity, lateral velocity, angular velocity)
        "cmd_init": np.array([1.5, 0, 0], dtype=np.float32)
    }
    return config

def load_policy(policy_path):
    """Load the pre-trained policy"""
    return torch.jit.load(policy_path)

def get_gravity_orientation(quaternion):
    """Convert quaternion to gravity orientation vector."""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def main():
    print("G1 Perlin Terrain Testing")
    print("=" * 50)
    
    # Load configuration
    config = load_g1_config()
    
    # Load policy
    policy = load_policy(config['policy_path'])
    
    # Read the robot XML and modify it to use the correct mesh path
    robot_xml_path = config["robot_xml_path"]
    with open(robot_xml_path, 'r') as f:
        robot_xml_content = f.read()
    
    # Replace the meshdir directive to use the correct absolute path
    robot_xml_content = robot_xml_content.replace(
        'meshdir="../robot_models/g1_description/meshes/"',
        'meshdir="robot_models/g1_description/meshes/"'
    )
    
    # Create a combined scene by including the robot in the Perlin terrain scene
    combined_xml = f"""
<mujoco model="g1 with perlin terrain">
  <compiler meshdir="robot_models/g1_description/meshes/"/>
  
  <!-- Include robot model content directly -->
  {robot_xml_content.split('<mujoco model="g1_12dof">')[1].split('</mujoco>')[0]}
  
  <statistic center="0 0 0.1" extent="0.8" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-130" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    <hfield name="perlin_hfield_1" size="3.0 2.0 0.08 0.05" file="../../../terrains/g1_perlin_terrain_1.png" />
    <material name="perlin_terrain_1_mat" rgba="0.3 0.5 0.3 1" roughness="0.6" />
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
    <geom name="perlin_terrain_1" type="hfield" hfield="perlin_hfield_1" pos="4.0 0.0 0.0" quat="1 0 0 0" material="perlin_terrain_1_mat" />
  </worldbody>
</mujoco>
"""

    # Load model with combined robot and Perlin terrain
    model = mujoco.MjModel.from_xml_string(combined_xml)
    data = mujoco.MjData(model)
    
    # Initialize simulation
    mujoco.mj_resetData(model, data)
    model.opt.timestep = config['simulation_dt']
    
    # Set initial joint positions
    data.qpos[7:7+12] = config['default_angles']
    
    # Initialize variables
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    counter = 0
    
    # =============================================================================
    # COMMAND MODIFICATION - Change these values to control robot movement
    # =============================================================================
    # Command format: [forward_velocity, lateral_velocity, angular_velocity]
    # Examples:
    # cmd = np.array([0.5, 0, 0.2], dtype=np.float32)    # Forward + slight turn
    # cmd = np.array([0.0, 0.3, 0], dtype=np.float32)    # Sideways movement
    # cmd = np.array([0.8, 0, 0], dtype=np.float32)      # Fast forward
    # cmd = np.array([0.2, 0, 0.5], dtype=np.float32)    # Slow forward + sharp turn
    
    cmd = np.array([1, 0, 0], dtype=np.float32)
    # cmd = config['cmd_init'].copy()  # Default: [0.5, 0, 0.2]
    # =============================================================================
    
    # Performance tracking
    start_time = time.time()
    fall_count = 0
    max_height = 0
    total_distance = 0
    last_pos = data.qpos[0:3].copy()
    
    # Success zone for Perlin terrain area
    success_zones = [
        {"name": "Perlin Terrain", "pos": [5, 0], "radius": 1.5, "description": "Main Perlin terrain area (moderate complexity)"}
    ]
    zone_visits = {zone["name"]: False for zone in success_zones}
    
    print(f"Command: [forward={cmd[0]:.1f}, lateral={cmd[1]:.1f}, angular={cmd[2]:.1f}]")
    print("Starting simulation...")
    
    # Convert config arrays to numpy
    kps = np.array(config['kps'], dtype=np.float32)
    kds = np.array(config['kds'], dtype=np.float32)
    cmd_scale = config['cmd_scale']  # Already a numpy array in config
    
    # Run simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        
        while viewer.is_running():
            step_start = time.time()
            
            # Apply PD control
            tau = (target_dof_pos - data.qpos[7:7+12]) * kps - data.qvel[6:6+12] * kds
            data.ctrl[:12] = tau
            
            # Step simulation
            mujoco.mj_step(model, data)
            counter += 1
            
            # Update policy at control frequency (50Hz)
            if counter % config['control_decimation'] == 0:
                # Create observation (following Script 8's approach)
                qj = data.qpos[7:7+config["num_actions"]]  # Joint positions
                dqj = data.qvel[6:6+config["num_actions"]]  # Joint velocities
                quat = data.qpos[3:7]  # Base orientation quaternion
                omega = data.qvel[3:6]  # Base angular velocity
                
                # Process observations
                qj = (qj - config["default_angles"]) * config["dof_pos_scale"]
                dqj = dqj * config["dof_vel_scale"]
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * config["ang_vel_scale"]
                
                # Create periodic signal for walking
                period = 0.8
                count = counter * config["simulation_dt"]
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                
                # Build observation vector (47 elements)
                obs = np.zeros(config["num_obs"], dtype=np.float32)
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9+config["num_actions"]] = qj
                obs[9+config["num_actions"]:9+2*config["num_actions"]] = dqj
                obs[9+2*config["num_actions"]:9+3*config["num_actions"]] = action
                obs[9+3*config["num_actions"]:9+3*config["num_actions"]+2] = np.array([sin_phase, cos_phase])
                
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = policy(obs_tensor).squeeze().numpy()
                
                # Transform action to target joint positions
                target_dof_pos = action * config['action_scale'] + config['default_angles']
            
            # Performance monitoring
            current_pos_3d = data.qpos[0:3]
            current_height = current_pos_3d[2]
            
            # Track maximum height
            if current_height > max_height:
                max_height = current_height
            
            # Check for falls
            if current_height < 0.3:  # Robot fell
                fall_count += 1
            
            # Calculate distance traveled
            distance = np.linalg.norm(current_pos_3d - last_pos)
            total_distance += distance
            last_pos = current_pos_3d.copy()
            
            # Check success zones
            for zone in success_zones:
                if not zone_visits[zone["name"]]:
                    zone_pos = np.array(zone["pos"])
                    robot_pos_2d = current_pos_3d[:2]
                    if np.linalg.norm(robot_pos_2d - zone_pos) < zone["radius"]:
                        zone_visits[zone["name"]] = True
            
            # Sync viewer
            viewer.sync()
            time.sleep(0.01)
            
            # Stop after reasonable time
            if counter > 5000:  # ~50 seconds
                break
    
    # Final performance report
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("PERLIN TERRAIN TEST RESULTS")
    print("=" * 50)
    print(f"Simulation time: {elapsed_time:.1f} seconds")
    print(f"Total steps: {counter}")
    print(f"Total distance traveled: {total_distance:.2f} meters")
    print(f"Maximum height reached: {max_height:.3f} meters")
    print(f"Number of falls: {fall_count}")
    print()
    print("Success zones reached:")
    for zone_name, reached in zone_visits.items():
        status = "YES" if reached else "NO"
        print(f"  - {zone_name}: {status}")
    print()
    
    # Performance assessment
    if fall_count == 0 and all(zone_visits.values()):
        print("EXCELLENT: Robot successfully navigated the Perlin terrain!")
    elif fall_count <= 2 and sum(zone_visits.values()) >= 1:
        print("GOOD: Robot handled the Perlin terrain challenges well.")
    elif fall_count <= 5:
        print("FAIR: Robot struggled with some Perlin terrain but made progress.")
    else:
        print("POOR: Robot had significant difficulty with the Perlin terrain.")

if __name__ == "__main__":
    main()
