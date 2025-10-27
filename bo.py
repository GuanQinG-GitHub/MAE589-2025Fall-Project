import time
import numpy as np
import mujoco
import mujoco.viewer
import torch
from eval import calculate_mos
from utils import load_g1_config, load_policy, combine_robot_and_terrain, get_gravity_orientation, is_stance
from skopt import gp_minimize
import os

def run_single_episode(config, policy, model, data, cmd, max_steps, success_zones, render = False, viewer = None):
    """
    Run a single episode (either headless or with viewer).
    Returns a tuple: (mean margin of stability (MoS), total_timesteps, total_travel_distance).
    """
    mujoco.mj_resetData(model, data)
    counter = 0
    fall_count = 0
    max_height = -np.inf
    total_distance = 0.0
    last_pos = data.qpos[0:3].copy()
    kps = np.array(config['kps'], dtype=np.float32).copy()
    kds = np.array(config['kds'], dtype=np.float32).copy()
    kp_ankle_pitch = np.array(config['kp_p'], dtype=np.float32)
    kps_ankle_roll = np.array(config['kp_r'], dtype=np.float32)
    cmd_scale = config['cmd_scale']
    target_dof_pos = config['default_angles'].copy()
    action = np.zeros(config['num_actions'], dtype=np.float32)
    zone_visits = {zone["name"]: False for zone in success_zones}
    mos_list = []
    # print("kp ankle pitch:", kp_ankle_pitch)
    # print("kp ankle roll:", kps_ankle_roll)
    while (viewer.is_running() if render else True) and counter < max_steps:
        
        tau = (target_dof_pos - data.qpos[7:7 + config["num_actions"]]) * kps \
              - data.qvel[6:6 + config["num_actions"]] * kds
        data.ctrl[: config["num_actions"]] = tau

        mujoco.mj_step(model, data)
        counter += 1

        mos = calculate_mos(model, data, direction = 'ml')
        if mos != -np.inf and not np.isnan(mos):
            mos_list.append(float(mos))

        if counter % config['control_decimation'] == 0:
            stance_left = is_stance(data, model, side='left')
            stance_right = is_stance(data, model, side='right')
            # print("Stance left:", stance_left, " Stance right:", stance_right)
            if stance_left:
                kps[4], kps[5] = kp_ankle_pitch, kps_ankle_roll  # left ankle pitch and roll
            else:
                kps[4], kps[5] = config['kps'][4], config['kps'][5]  # left ankle swing as usual
            if stance_right:
                kps[10], kps[11] = kp_ankle_pitch, kps_ankle_roll # right ankle pitch and roll
            else:
                kps[10], kps[11] = config['kps'][10], config['kps'][11]  # right ankle swing as usual
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
            obs[9 + 2 * config["num_actions"]:9 + 3 * config["num_actions"]] = action
            obs[9 + 3 * config["num_actions"]:9 + 3 * config["num_actions"] + 2] = np.array([sin_phase, cos_phase])

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor).squeeze().numpy()

            target_dof_pos = action * config['action_scale'] + config['default_angles']

        current_pos_3d = data.qpos[0:3].copy()
        current_height = float(current_pos_3d[2])

        if current_height > max_height:
            max_height = current_height

        if current_height < 0.5:
            fall_count += 1
            break

        distance = np.linalg.norm(current_pos_3d - last_pos)
        total_distance += float(distance)
        last_pos = current_pos_3d.copy()

        for zone in success_zones:
            if not zone_visits[zone["name"]]:
                zone_pos = np.array(zone["pos"], dtype=np.float32)
                robot_pos_2d = current_pos_3d[:2]
                if np.linalg.norm(robot_pos_2d - zone_pos) < zone["radius"]:
                    zone_visits[zone["name"]] = True

        if render and viewer is not None:
            # Viewer sync and pacing
            viewer.sync() if counter % 10 == 0 else None
            time.sleep(0.001)

    ep_mos_mean = float(np.mean(mos_list)) if mos_list else float('nan')
    total_time = counter *model.opt.timestep # in seconds
    return ep_mos_mean, total_time, float(total_distance)

def main(enable_viewer=False, kp_ankle_pitch=30.0, kp_ankle_roll=30.0):
    print("G1 Perlin Terrain Testing")
    print("=" * 50)

    config = load_g1_config()
    config['kp_p'] = kp_ankle_pitch  # ankle pitch stiffness
    config['kp_r'] = kp_ankle_roll   # ankle roll stiffness
    policy = load_policy(config['policy_path'])

    combined_xml = combine_robot_and_terrain(config["robot_xml_path"])
    model = mujoco.MjModel.from_xml_string(combined_xml)
    model.opt.timestep = config['simulation_dt']

    cmd = config.get("cmd_init", np.array([1.0, 0.0, 0.0], dtype=np.float32)).copy()
    success_zones = [
        {"name": "Perlin Terrain", "pos": [5, 0], "radius": 1.5, "description": "Main Perlin terrain area (moderate complexity)"}
    ]
    num_episodes = 5
    max_steps = int(config['simulation_duration'] / config['simulation_dt'])
    episode_mos_means = []

    for ep in range(1, num_episodes + 1):
        data = mujoco.MjData(model)
        if enable_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                print(f"\n=== Episode {ep}/{num_episodes} ===")
                ep_mos_mean, avg_time, avg_distance = run_single_episode(config, policy, model, data, cmd, max_steps, success_zones, render=True, viewer=viewer)
        else:
            ep_mos_mean, avg_time, avg_distance = run_single_episode(config, policy, model, data, cmd, max_steps, success_zones)
        print(f"Episode {ep} MoS mean: {ep_mos_mean:.2f}, Time: {avg_time:.2f}s, Distance: {avg_distance:.2f}m")
        episode_mos_means.append(ep_mos_mean)

    print(f"\nAverage MoS over {num_episodes} episodes: {np.nanmean(episode_mos_means):.2f}")

def evaluate(ankle_stiffness, config):
    config['kp_p'] = ankle_stiffness[0] # ankle pitch stiffness
    config['kp_r'] = ankle_stiffness[1]  # ankle roll stiffness
    policy = load_policy(config['policy_path'])

    combined_xml = combine_robot_and_terrain(config["robot_xml_path"])
    model = mujoco.MjModel.from_xml_string(combined_xml)
    model.opt.timestep = config['simulation_dt']

    cmd = config.get("cmd_init", np.array([1.0, 0.0, 0.0], dtype=np.float32)).copy()
    success_zones = [
        {"name": "Perlin Terrain", "pos": [5, 0], "radius": 1.5, "description": "Main Perlin terrain area (moderate complexity)"}
    ]
    num_episodes = config.get('num_episodes_per_eval', 4)
    max_steps = int(config['simulation_duration'] / config['simulation_dt'])
    episode_mos_means, total_times, total_distances = [], [], []

    for ep in range(1, num_episodes + 1):
        data = mujoco.MjData(model)
        ep_mos_mean, total_time, total_distance = run_single_episode(config, policy, model, data, cmd, max_steps, success_zones)
        episode_mos_means.append(ep_mos_mean)
        total_times.append(total_time)
        total_distances.append(total_distance)

    avg_mos = np.nanmean(episode_mos_means)
    avg_time = np.mean(total_times)
    avg_distance = np.mean(total_distances)
    print(f"Ankle Stiffness: {ankle_stiffness[0]:.2f}, {ankle_stiffness[1]:.2f}, Average MoS: {avg_mos:.2f}, Average Time: {avg_time:.2f}, Average Distance: {avg_distance:.2f}")
    return -avg_mos - avg_time*0.1 # Negative for minimization

def main_pipeline():
    print("G1 Perlin Terrain  BO")
    print("=" * 50)

    import matplotlib.pyplot as plt

    config = load_g1_config()

    param_bounds = [
        (15, 60),  # ankle stiffness range
        (15, 60)   # ankle damping range
    ]

    result = gp_minimize(
        lambda p: evaluate(p, config),
        param_bounds,
        n_calls=40,
        random_state=42,
        acq_func='EI'
    )

    # Extract explored points and their (negative) objective values
    explored = np.array(result.x_iters)          # shape (n_calls, 2)
    func_vals = np.array(result.func_vals)       # these are negative avg_mos (minimization)
    avg_mos_vals = -func_vals                    # convert back to positive avg_mos (higher is better)

    # Log explored points to CSV
    out_dir = os.path.join(os.getcwd(), "bo_results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "explored_points.csv")
    header = "param1,param2,avg_mos"
    np.savetxt(csv_path, np.column_stack([explored[:, 0], explored[:, 1], avg_mos_vals]), delimiter=",", header=header, comments='', fmt="%.6f")
    print(f"Explored points logged to: {csv_path}")

    # Print explored points to stdout
    print("Explored points (param1, param2, avg_mos):")
    for p, m in zip(explored, avg_mos_vals):
        print(f"  {p[0]:.3f}, {p[1]:.3f} -> {m:.4f}")

    # Scatter plot: param1 vs param2 colored by avg_mos
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(explored[:, 0], explored[:, 1], c=avg_mos_vals, cmap="viridis", s=80, edgecolors="k")
    plt.colorbar(sc, label="Average cost")
    plt.scatter([result.x[0]], [result.x[1]], marker="*", color="red", s=200, label="Best")
    plt.plot([b[0] for b in param_bounds], [b[1] for b in param_bounds], 'r--', alpha=0.5)
    plt.xlabel("Ankle stiffness pitch")
    plt.ylabel("Ankle damping roll")
    plt.title("BO explored points: params vs Average cost")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = os.path.join(out_dir, "bo_param_vs_cost.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")
    try:
        plt.show()
    except Exception:
        # In headless environments, showing may fail; continue silently.
        pass

    print("Best stiffness:", result.x)
    print("Best MoS achieved:", -result.fun)



    return result


if __name__ == "__main__":
    # Set enable_viewer=True for rendering, False for headless
    # main(enable_viewer=True, kp_ankle_pitch=54.0, kp_ankle_roll=48.0)
    main(enable_viewer=False, kp_ankle_pitch=20.0, kp_ankle_roll=20.0)
    main(enable_viewer=False, kp_ankle_pitch=54.0, kp_ankle_roll=48.0)
    # main_pipeline()

