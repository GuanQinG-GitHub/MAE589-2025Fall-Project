"""
Date: Oct 3, 2025

Visualize the Unitree G1 humanoid (29 DoF) model in MuJoCo viewer.

This script follows the structure of `Mujoco_TutorialScript_1.py` and
loads the G1 XML from a locally cloned repository so that relative
asset paths (meshes, textures, includes) resolve without any network I/O.

Source model:
  - https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_29dof.xml

Notes:
  - Ensure you have cloned the repo under `external/unitree_mujoco`.
  - Example:
      git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
"""

import time
import pathlib

import mujoco
import mujoco.viewer


def get_local_model_xml() -> pathlib.Path:
    project_root = pathlib.Path(__file__).parent
    local_repo_root = project_root / "external" / "unitree_mujoco"
    # Prefer the scene file which includes ground/terrain
    scene_xml = local_repo_root / "unitree_robots" / "g1" / "scene_23dof.xml"
    if scene_xml.exists():
        return scene_xml
    # Fallback to robot-only XML (may lack ground)
    model_xml = local_repo_root / "unitree_robots" / "g1" / "g1_29dof.xml"
    if not model_xml.exists():
        raise FileNotFoundError(
            f"Model XML not found at {model_xml}. Clone the repo to 'external/unitree_mujoco'."
        )
    return model_xml


def main() -> None:
    local_xml_path = get_local_model_xml()
    model = mujoco.MjModel.from_xml_path(str(local_xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()


