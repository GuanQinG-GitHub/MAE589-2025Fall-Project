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
from utils import load_g1_config, combine_robot_and_terrain, load_policy, get_gravity_orientation

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
