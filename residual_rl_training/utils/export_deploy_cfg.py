# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified export function for residual RL deployment config."""

import os
import yaml

from isaaclab.envs import ManagerBasedRLEnv


def format_value(x):
    """Format value for YAML output."""
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    """Export deployment configuration for residual RL environment.
    
    Args:
        env: The environment instance.
        log_dir: Directory to save the config file.
    """
    # Create a simplified deployment config
    cfg = {}
    
    # Basic simulation parameters
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    
    # Export base policy path if available
    if hasattr(env.cfg, "base_policy_path") and env.cfg.base_policy_path:
        cfg["base_policy_path"] = env.cfg.base_policy_path
    
    # Export residual control parameters
    if hasattr(env.cfg.control, "residual_kp_scale"):
        cfg["residual_kp_scale"] = env.cfg.control.residual_kp_scale
    if hasattr(env.cfg.control, "kp_min"):
        cfg["kp_min"] = env.cfg.control.kp_min
    if hasattr(env.cfg.control, "kp_max"):
        cfg["kp_max"] = env.cfg.control.kp_max
    
    # Save config file
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)

