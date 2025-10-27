import json
import os
import numpy as np
import torch

def is_stance(data, model, side = 'left'):
    # Determine if the requested side leg is in stance (i.e. contacting ground with sufficient force)
    force_threshold = 0.5
    side = side.lower()
    stance = False

    for i in range(data.ncon):
        contact = data.contact[i]

        # MuJoCo contact force slice
        efc_addr = int(contact.efc_address)
        dim = int(contact.dim)
        contact_force_vector = np.array(data.efc_force[efc_addr: efc_addr + dim])

        if np.linalg.norm(contact_force_vector) <= force_threshold:
            continue

        # check if contact involves ground
        g1 = contact.geom1
        g2 = contact.geom2
        name1 = model.geom(g1).name.lower()
        name2 = model.geom(g2).name.lower()

        ground_hit = ("perlin_terrain_1" in name1 or "floor" in name1 or
                      "perlin_terrain_1" in name2 or "floor" in name2)
        if not ground_hit:
            continue

        # the non-ground geom is the foot
        if "perlin_terrain_1" in name1 or "floor" in name1:
            foot_name = name2
        else:
            foot_name = name1

        # detect a foot geom for the requested side (match your XML naming)
        if side in foot_name:
            stance = True
            break

    return bool(stance)



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

def load_policy(policy_path):
    """Load the pre-trained policy"""
    return torch.jit.load(policy_path)

def load_g1_config(json_path=None):
    """Load G1 configuration parameters from a JSON file with sensible defaults.

    If json_path is None the function will look for 'g1_config.json' next to this script.
    Fields in the JSON override the defaults. Numeric lists are converted to numpy arrays
    where the main script expects them.
    """

    # Default config (same semantics as previous hard-coded dict)
    defaults = {
        "policy_path": "trained_models/motion.pt",
        "robot_xml_path": "robot_models/g1_description/g1_12dof.xml",
        "simulation_duration": 10.0,
        "simulation_dt": 0.002,
        "control_decimation": 10,
        "kps": [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40],
        "kds": [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2],
        "default_angles": [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
        "ang_vel_scale": 0.25,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 0.05,
        "action_scale": 0.25,
        "cmd_scale": [2.0, 2.0, 0.25],
        "num_actions": 12,
        "num_obs": 47,
        "cmd_init": [1.5, 0, 0]
    }

    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "g1_config.json")

    file_cfg = {}
    try:
        with open(json_path, "r") as f:
            file_cfg = json.load(f)
    except FileNotFoundError:
        # fallback to defaults if no file present
        file_cfg = {}
    except Exception as e:
        raise RuntimeError(f"Error reading config file '{json_path}': {e}")

    # Merge and normalize types
    cfg = defaults.copy()
    cfg.update(file_cfg)

    # Ensure correct types and numpy arrays where expected
    cfg["kps"] = list(cfg["kps"])
    cfg["kds"] = list(cfg["kds"])
    cfg["default_angles"] = np.array(cfg["default_angles"], dtype=np.float32)
    cfg["cmd_scale"] = np.array(cfg["cmd_scale"], dtype=np.float32)
    cfg["cmd_init"] = np.array(cfg.get("cmd_init", cfg["cmd_init"]), dtype=np.float32)

    cfg["simulation_duration"] = float(cfg["simulation_duration"])
    cfg["simulation_dt"] = float(cfg["simulation_dt"])
    cfg["control_decimation"] = int(cfg["control_decimation"])
    cfg["num_actions"] = int(cfg["num_actions"])
    cfg["num_obs"] = int(cfg["num_obs"])
    cfg["action_scale"] = float(cfg["action_scale"])
    cfg["dof_pos_scale"] = float(cfg["dof_pos_scale"])
    cfg["dof_vel_scale"] = float(cfg["dof_vel_scale"])
    cfg["ang_vel_scale"] = float(cfg["ang_vel_scale"])

    return cfg



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
    <hfield name="perlin_hfield_1" size="15.0 5.0 0.60 0.01" file="../../../terrains/g1_perlin_terrain_1.png" />
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
