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
from eval import calculate_mos

def load_g1_config():
    """Load G1 configuration parameters."""
    config = {
        # Model paths
        "policy_path": "trained_models/motion.pt",
        "robot_xml_path": "robot_models/g1_description/g1_12dof.xml",
        
        # Simulation parameters
        "simulation_duration": 10.0,  # seconds
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


def combine_robot_and_terrain(robot_xml_path):
    """Combine robot and terrain XML files into a single Mujoco model."""
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
    <hfield name="perlin_hfield_1" size="15.0 15.0 0.60 0.01" file="../../../terrains/g1_perlin_terrain_1.png" />
    <material name="perlin_terrain_1_mat" rgba="0.3 0.5 0.3 1" roughness="0.5" />
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
    <geom name="perlin_terrain_1" type="hfield" hfield="perlin_hfield_1" pos="4.0 0.0 0.0" quat="1 0 0 0" material="perlin_terrain_1_mat" />
  </worldbody>
</mujoco>
"""
    return combined_xml
def main():
    print("G1 Perlin Terrain Testing")
    print("=" * 50)

    # Load configuration and policy
    config = load_g1_config()
    policy = load_policy(config['policy_path'])

    # Prepare model & data
    combined_xml = combine_robot_and_terrain(config["robot_xml_path"])
    model = mujoco.MjModel.from_xml_string(combined_xml)
    model.opt.timestep = config['simulation_dt']

    # Command setup
    cmd = config.get("cmd_init", np.array([1.0, 0.0, 0.0], dtype=np.float32)).copy()

    # Success zones (shared across episodes)
    success_zones = [
        {"name": "Perlin Terrain", "pos": [5, 0], "radius": 1.5, "description": "Main Perlin terrain area (moderate complexity)"}
    ]

    # Evaluation settings
    num_episodes = 5
    max_steps = int(config['simulation_duration'] / config['simulation_dt'])

    # Aggregation containers
    episode_results = []
    total_falls = 0
    episodes_with_all_zones = 0
    episode_mos_means = []

    # Launch viewer once and reuse; data will be reset each episode
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        for ep in range(1, num_episodes + 1):
            # Initialize per-episode variables (reset simulation state)
            mujoco.mj_resetData(model, data)
            counter = 0
            fall_count = 0
            max_height = -np.inf
            total_distance = 0.0
            last_pos = data.qpos[0:3].copy()
            kps = np.array(config['kps'], dtype=np.float32)
            kds = np.array(config['kds'], dtype=np.float32)
            cmd_scale = config['cmd_scale']
            target_dof_pos = config['default_angles'].copy()
            action = np.zeros(config['num_actions'], dtype=np.float32)
            zone_visits = {zone["name"]: False for zone in success_zones}
            mos_list = []

            print(f"\n=== Episode {ep}/{num_episodes} ===")
            print(f"Command: [forward={cmd[0]:.1f}, lateral={cmd[1]:.1f}, angular={cmd[2]:.1f}]")
            start_time = time.time()

            # Episode simulation loop
            while viewer.is_running() and counter < max_steps:
                # PD control (ensure indices match model DOF layout)
                tau = (target_dof_pos - data.qpos[7:7 + config["num_actions"]]) * kps \
                        - data.qvel[6:6 + config["num_actions"]] * kds
                data.ctrl[: config["num_actions"]] = tau

                # Step
                mujoco.mj_step(model, data)
                counter += 1

                # Measure MoS this step
                mos = calculate_mos(model, data)
                if mos != -np.inf and not np.isnan(mos):
                    mos_list.append(float(mos))

                # Control update at control frequency
                if counter % config['control_decimation'] == 0:
                    # Observations
                    qj = data.qpos[7:7 + config["num_actions"]].copy()
                    dqj = data.qvel[6:6 + config["num_actions"]].copy()
                    quat = data.qpos[3:7].copy()
                    omega = data.qvel[3:6].copy()

                    qj = (qj - config["default_angles"]) * config["dof_pos_scale"]
                    dqj = dqj * config["dof_vel_scale"]
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * config["ang_vel_scale"]

                    period = 0.8
                    sim_time = counter * model.opt.timestep
                    phase = (sim_time % period) / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs = np.zeros(config["num_obs"], dtype=np.float32)
                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:9 + config["num_actions"]] = qj
                    obs[9 + config["num_actions"]:9 + 2 * config["num_actions"]] = dqj
                    # use last action for this slot (safe default)
                    obs[9 + 2 * config["num_actions"]:9 + 3 * config["num_actions"]] = action
                    obs[9 + 3 * config["num_actions"]:9 + 3 * config["num_actions"] + 2] = np.array([sin_phase, cos_phase])

                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = policy(obs_tensor).squeeze().numpy()

                    # Transform action to target joint positions
                    target_dof_pos = action * config['action_scale'] + config['default_angles']

                # Performance tracking
                current_pos_3d = data.qpos[0:3].copy()
                current_height = float(current_pos_3d[2])

                if current_height > max_height:
                    max_height = current_height

                # Fall detection
                if current_height < 0.5:
                    fall_count += 1
                    break  # end episode on fall

                # Distance traveled
                distance = np.linalg.norm(current_pos_3d - last_pos)
                total_distance += float(distance)
                last_pos = current_pos_3d.copy()

                # Success zone checks
                for zone in success_zones:
                    if not zone_visits[zone["name"]]:
                        zone_pos = np.array(zone["pos"], dtype=np.float32)
                        robot_pos_2d = current_pos_3d[:2]
                        if np.linalg.norm(robot_pos_2d - zone_pos) < zone["radius"]:
                            zone_visits[zone["name"]] = True

                # Viewer sync and pacing
                viewer.sync() if counter % 10 == 0 else None
                time.sleep(0.001)

            # Episode summary & aggregation
            ep_elapsed = time.time() - start_time
            ep_mos_mean = float(np.mean(mos_list)) if mos_list else float('nan')
            episode_mos_means.append(ep_mos_mean)
            total_falls += fall_count
            if all(zone_visits.values()):
                episodes_with_all_zones += 1

            episode_results.append({
                "episode": ep,
                "steps": counter,
                "distance": total_distance,
                "max_height": max_height,
                "falls": fall_count,
                "mos_mean": ep_mos_mean,
                "zones": zone_visits,
                "elapsed": ep_elapsed
            })

            # Print episode brief
            print(f"Episode {ep} finished: steps={counter}, distance={total_distance:.2f}, max_h={max_height:.3f}, falls={fall_count}, mos_mean={ep_mos_mean:.2f}")

    # Aggregate results across episodes
    success_rate = episodes_with_all_zones / num_episodes * 100.0
    mos_mean = float(np.nanmean(episode_mos_means)) if episode_mos_means else float('nan')
    mos_std = float(np.nanstd(episode_mos_means)) if episode_mos_means else float('nan')

    print("\n" + "=" * 50)
    print("AGGREGATED PERLIN TERRAIN EVALUATION")
    print("=" * 50)
    print(f"Episodes run: {num_episodes}")
    print(f"Total falls: {total_falls}")
    print(f"Success rate (all zones reached): {success_rate:.1f}%")
    print(f"MoS across episodes: mean={mos_mean:.2f}, std={mos_std:.2f}")
    print("\nPer-episode summary:")
    for r in episode_results:
        zones_str = ", ".join([f"{k}:{'YES' if v else 'NO'}" for k, v in r["zones"].items()])
        print(f"  Ep{r['episode']}: steps={r['steps']}, dist={r['distance']:.2f}, falls={r['falls']}, mos_mean={r['mos_mean']:.2f}, zones=[{zones_str}]")

if __name__ == "__main__":
    main()
