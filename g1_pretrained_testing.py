"""
G1 Pre-trained Model Testing on Flat Ground

This script tests the pre-trained G1 humanoid model on flat ground in MuJoCo.
It demonstrates the robot's walking behavior using the pre-trained policy.

Features:
- Loads pre-trained G1 policy from local trained_models folder
- Uses proper PD control with configurable gains
- Implements observation processing and action scaling
- Real-time visualization with MuJoCo viewer
- Tests on simple flat ground terrain

Requirements:
- MuJoCo installed
- PyTorch installed
- Local robot models and trained policy files
"""

import time
import pathlib
import numpy as np
import torch
import mujoco
import mujoco.viewer


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands using PD control."""
    return (target_q - q) * kp + (target_dq - dq) * kd


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
        "cmd_init": np.array([0.5, 0, 0.2], dtype=np.float32)
    }
    return config


def main():
    """Main function to run G1 with pre-trained model."""
    print("G1 Pre-trained Model Testing on Flat Ground")
    print("=" * 50)
    
    # Load configuration
    config = load_g1_config()
    
    # Verify paths exist
    policy_path = pathlib.Path(config["policy_path"])
    robot_xml_path = pathlib.Path(config["robot_xml_path"])
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not robot_xml_path.exists():
        raise FileNotFoundError(f"Robot XML file not found: {robot_xml_path}")
    
    print(f"Policy loaded from: {policy_path}")
    print(f"Robot model loaded from: {robot_xml_path}")
    
    # Initialize variables
    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    counter = 0
    cmd = config["cmd_init"].copy()
    
    # Read the robot XML and modify it to use the correct mesh path
    with open(robot_xml_path, 'r') as f:
        robot_xml_content = f.read()
    
    # Replace the meshdir directive to use the correct absolute path
    robot_xml_content = robot_xml_content.replace(
        'meshdir="../robot_models/g1_description/meshes/"',
        'meshdir="robot_models/g1_description/meshes/"'
    )
    
    # Create a combined scene by including the robot in the terrain scene
    # This approach keeps the terrain generation model-independent
    combined_xml = f"""
<mujoco model="g1 with flat ground">
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
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
  </worldbody>
</mujoco>
"""

    # Load robot model
    print("Loading MuJoCo model...")
    m = mujoco.MjModel.from_xml_string(combined_xml)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    # Load pre-trained policy
    print("Loading pre-trained policy...")
    policy = torch.jit.load(str(policy_path))
    policy.eval()
    
    print("Starting simulation...")
    print("Controls:")
    print("  - Close viewer window to stop simulation")
    print("  - Simulation will run for 60 seconds")
    print("  - G1 will demonstrate walking behavior on flat ground")
    print()
    
    # Convert config arrays to numpy
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    cmd_scale = config["cmd_scale"]
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set viewer options
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        
        start_time = time.time()
        
        while viewer.is_running() and time.time() - start_time < config["simulation_duration"]:
            step_start = time.time()
            
            # Apply PD control
            tau = pd_control(
                target_dof_pos, 
                d.qpos[7:7+config["num_actions"]],  # Joint positions (skip base pose)
                kps, 
                np.zeros_like(kds),  # Target velocities (zero)
                d.qvel[6:6+config["num_actions"]],  # Joint velocities (skip base)
                kds
            )
            d.ctrl[:] = tau
            
            # Step simulation
            mujoco.mj_step(m, d)
            
            counter += 1
            
            # Update policy at control frequency (50Hz)
            if counter % config["control_decimation"] == 0:
                # Create observation
                qj = d.qpos[7:7+config["num_actions"]]  # Joint positions
                dqj = d.qvel[6:6+config["num_actions"]]  # Joint velocities
                quat = d.qpos[3:7]  # Base orientation quaternion
                omega = d.qvel[3:6]  # Base angular velocity
                
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
                
                # Build observation vector
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9+config["num_actions"]] = qj
                obs[9+config["num_actions"]:9+2*config["num_actions"]] = dqj
                obs[9+2*config["num_actions"]:9+3*config["num_actions"]] = action
                obs[9+3*config["num_actions"]:9+3*config["num_actions"]+2] = np.array([sin_phase, cos_phase])
                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                # Transform action to target joint positions
                target_dof_pos = action * config["action_scale"] + config["default_angles"]
            
            # Update viewer
            viewer.sync()
            
            # Maintain real-time simulation
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Simulation completed!")


if __name__ == "__main__":
    main()
