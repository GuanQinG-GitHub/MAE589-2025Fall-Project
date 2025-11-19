# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register residual RL task for G1 ankle impedance scheduling."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-Residual-Velocity",
    entry_point=f"{__name__}.g1_residual_env:G1ResidualRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_residual_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_residual_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:ResidualPPORunnerCfg",
    },
)

