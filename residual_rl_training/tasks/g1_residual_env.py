# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom environment for G1 Residual RL with base policy integration."""

import numpy as np
import torch
import re
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


class G1ResidualRLEnv(ManagerBasedRLEnv):
    """
    Custom environment for G1 Residual RL with frozen base policy integration.
    
    This environment extends ManagerBasedRLEnv to:
    1. Load and freeze a pretrained base policy
    2. Get base policy actions (12 DOF) for all leg joints
    3. Get residual policy actions (4 DOF) for ankle PD modifications
    4. Compute torques with modified ankle PD parameters
    5. Apply torques directly to the robot
    
    Key points:
    - Base policy outputs are desired positions (bypass action manager)
    - Residual policy outputs modify ankle PD parameters (go through action manager)
    - Torques are computed manually with dynamic PD parameters
    """

    def __init__(self, cfg, **kwargs):
        """
        Initialize the residual RL environment.
        
        Args:
            cfg: Environment configuration (RobotEnvCfg)
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # ====================================================================
        # ACTION MANAGER TERM LOOKUP
        # ====================================================================
        self._residual_action_term_name = None
        if hasattr(self, "action_manager") and self.action_manager is not None:
            if "JointPositionAction" in self.action_manager.active_terms:
                self._residual_action_term_name = "JointPositionAction"
            elif self.action_manager.active_terms:
                self._residual_action_term_name = self.action_manager.active_terms[0]
            else:
                raise RuntimeError("Action manager has no active terms configured for residual actions.")
        else:
            raise RuntimeError("Action manager is not initialized. Residual actions cannot be processed.")

        # ====================================================================
        # BASE POLICY LOADING
        # ====================================================================
        self.base_policy = None
        self.base_policy_loaded = False
        self.base_policy_path = getattr(cfg, "base_policy_path", None)
        
        if self.base_policy_path:
            try:
                # Load pretrained base policy
                self.base_policy = torch.jit.load(self.base_policy_path, map_location=self.device)
                self.base_policy.eval()  # Set to evaluation mode
                
                # Freeze the base policy (no gradients)
                for param in self.base_policy.parameters():
                    param.requires_grad = False
                
                self.base_policy_loaded = True
                print(f"[INFO] Loaded base policy from: {self.base_policy_path}")
                self._initialize_base_policy_memory()
            except Exception as e:
                print(f"[WARN] Could not load base policy: {e}")
                print("[WARN] Training without base policy (not recommended)")
                self.base_policy = None
        else:
            print("[WARN] No base policy path specified. Training without base policy (not recommended)")
        
        # ====================================================================
        # ANKLE JOINT INDICES
        # ====================================================================
        # Find ankle joint indices for impedance modification
        # Ankle joints: left_ankle_pitch, left_ankle_roll, 
        #               right_ankle_pitch, right_ankle_roll
        robot: Articulation = self.scene["robot"]
        ankle_joint_names = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        
        self.ankle_joint_indices = []
        for name in ankle_joint_names:
            for i, joint_name in enumerate(robot.data.joint_names):
                if name in joint_name:
                    self.ankle_joint_indices.append(i)
                    break
        
        if len(self.ankle_joint_indices) != 4:
            raise ValueError(
                f"Could not find all ankle joints. Found: {self.ankle_joint_indices}. "
                f"Available joints: {robot.data.joint_names}"
            )
        
        print(f"[INFO] Ankle joint indices: {self.ankle_joint_indices}")
        
        # ====================================================================
        # BASE ANKLE PD PARAMETERS
        # ====================================================================
        # Get base PD parameters from robot configuration
        # For G1-12dof, ankle joints use N5020-16-parallel actuator with kp=40, kd=2
        self.base_ankle_kp = 40.0  # Base ankle stiffness
        self.base_ankle_kd = 2.0   # Base ankle damping
        
        # ====================================================================
        # RESIDUAL IMPEDANCE BUFFERS
        # ====================================================================
        # Buffers for residual impedance modifications
        num_envs = self.scene.num_envs
        self.residual_kp_modifications = torch.zeros(
            num_envs, 4, dtype=torch.float, device=self.device
        )
        
        # Current ankle PD parameters (base + residual modifications)
        self.current_ankle_kp = torch.full(
            (num_envs, 4), self.base_ankle_kp, dtype=torch.float, device=self.device
        )
        self.current_ankle_kd = torch.full(
            (num_envs, 4), self.base_ankle_kd, dtype=torch.float, device=self.device
        )
        self.joint_effort_limits = self._build_joint_effort_limits(robot)
        self.debug_log = getattr(cfg, "debug_log", True)
        self.debug_log_interval = getattr(cfg, "debug_log_interval", 200)
        self.debug_max_logs = getattr(cfg, "debug_max_logs", 20)
        self._debug_step_counter = 0
        self._debug_logs_emitted = 0
        
        # ====================================================================
        # BASE POLICY OBSERVATION SCALES
        # ====================================================================
        # Same as training (from g1_residual_env.py)
        self.base_obs_scales = {
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "cmd": [2.0, 2.0, 0.25],
        }
        
        # ====================================================================
        # EPISODE TRACKING FOR PHASE COMPUTATION
        # ====================================================================
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        
        # ====================================================================
        # RESIDUAL CONTROL PARAMETERS
        # ====================================================================
        # Get from config if available
        if hasattr(cfg, "control"):
            self.residual_kp_scale = getattr(cfg.control, "residual_kp_scale", 0.25)
            self.residual_kd_ratio = getattr(cfg.control, "residual_kd_ratio", 0.5)
            self.kp_min = getattr(cfg.control, "kp_min", 20.0)
            self.kp_max = getattr(cfg.control, "kp_max", 60.0)
            self.action_scale = getattr(cfg.control, "action_scale", 0.25)
        else:
            # Default values
            self.residual_kp_scale = 0.25
            self.residual_kd_ratio = 0.5
            self.kp_min = 20.0
            self.kp_max = 60.0
            self.action_scale = 0.25

    def step(self, action: torch.Tensor):
        """
        Override step to inject custom torque computation while preserving Isaac Lab bookkeeping.
        """
        action = action.to(self.device)
        self.action_manager.process_action(action)
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._compute_and_apply_residual_torques()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _compute_and_apply_residual_torques(self):
        """Compute torques from base and residual policies and write them to the robot."""
        processed_residual_actions = (
            self.action_manager.get_term(self._residual_action_term_name).processed_actions.clone()
        )
        num_envs = self.scene.num_envs
        if processed_residual_actions.dim() == 1:
            processed_residual_actions = processed_residual_actions.unsqueeze(0).repeat(num_envs, 1)
        elif processed_residual_actions.shape[0] != num_envs:
            if processed_residual_actions.shape[0] == 1:
                processed_residual_actions = processed_residual_actions.repeat(num_envs, 1)
            else:
                processed_residual_actions = processed_residual_actions[:num_envs]
        # ====================================================================
        # GET BASE POLICY ACTIONS (12 DOF) - BYPASS ACTION MANAGER
        # ====================================================================
        if self.base_policy is not None:
            # Construct base policy observations
            base_obs = self._get_base_policy_observations()
            self._last_base_obs_env0 = base_obs[0].detach().to("cpu")
            
            # Get base policy actions (no gradients)
            with torch.no_grad():
                base_actions = self.base_policy(base_obs)
            
            # Ensure correct shape [num_envs, 12]
            if isinstance(base_actions, torch.Tensor):
                base_actions = base_actions.detach()
            else:
                base_actions = torch.from_numpy(np.array(base_actions, dtype=np.float32)).to(self.device)
            
            if base_actions.dim() == 1:
                base_actions = base_actions.unsqueeze(0).repeat(num_envs, 1)
            elif base_actions.shape[0] != num_envs:
                if base_actions.shape[0] == 1:
                    base_actions = base_actions.repeat(num_envs, 1)
                else:
                    base_actions = base_actions[:num_envs]
            
            # Store for next observation
            self.last_base_actions = base_actions.clone()
        else:
            # No base policy: use zero actions (for testing)
            base_actions = torch.zeros(num_envs, 12, dtype=torch.float32, device=self.device)
            if not hasattr(self, "last_base_actions"):
                self.last_base_actions = base_actions.clone()
        
        # ====================================================================
        # MODIFY ANKLE PD PARAMETERS BASED ON RESIDUAL ACTIONS
        # ====================================================================
        # Residual actions are in range [-1, 1] (after action manager processing)
        # Scale to modify kp: kp_new = kp_base + action * scale * kp_base
        self.residual_kp_modifications = processed_residual_actions * self.residual_kp_scale * self.base_ankle_kp
        
        # Compute new ankle kp values (clamped to [kp_min, kp_max])
        self.current_ankle_kp = torch.clamp(
            self.base_ankle_kp + self.residual_kp_modifications,
            min=self.kp_min,
            max=self.kp_max,
        )
        
        # Compute new ankle kd values (proportional to kp)
        self.current_ankle_kd = (
            self.base_ankle_kd
            * (self.current_ankle_kp / self.base_ankle_kp)
            * self.residual_kd_ratio
            + self.base_ankle_kd * (1 - self.residual_kd_ratio)
        )
        
        # ====================================================================
        # COMPUTE TORQUES WITH MODIFIED ANKLE PD PARAMETERS
        # ====================================================================
        robot: Articulation = self.scene["robot"]
        torques, target_positions = self._compute_torques_with_modified_pd(base_actions, processed_residual_actions)
        self._debug_log_step(
            processed_residual_actions,
            base_actions,
            torques,
            target_positions,
            robot,
        )
        
        # ====================================================================
        # APPLY TORQUES DIRECTLY TO ROBOT
        # ====================================================================
        robot: Articulation = self.scene["robot"]
        robot.set_joint_effort_target(torques)
        
        # Note: We've applied torques directly, so we bypass the normal action application
        # The parent class's step() will handle simulation stepping, observations, rewards, etc.

    def _compute_torques_with_modified_pd(
        self, base_actions: torch.Tensor, residual_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint torques with modified ankle PD parameters.
        
        Args:
            base_actions: Base policy actions [num_envs, 12] (desired positions)
            residual_actions: Residual actions [num_envs, 4] (not used here, already processed)
            
        Returns:
            torques: Joint torques [num_envs, 12]
            target_positions: Position targets sent to PD [num_envs, 12]
        """
        robot: Articulation = self.scene["robot"]
        
        # Get current joint states
        joint_pos = robot.data.joint_pos  # [num_envs, num_joints]
        joint_vel = robot.data.joint_vel  # [num_envs, num_joints]
        default_joint_pos = robot.data.default_joint_pos  # [num_envs, num_joints]
        
        # Scale base actions to get target positions
        actions_scaled = base_actions * self.action_scale
        target_positions = actions_scaled + default_joint_pos
        
        # Get base PD parameters from robot
        base_kp = robot.data.default_joint_stiffness  # [num_envs, num_joints]
        base_kd = robot.data.default_joint_damping   # [num_envs, num_joints]
        
        # Create modified PD parameter arrays
        kp = base_kp.clone()
        kd = base_kd.clone()
        
        # Modify ankle joints with residual PD parameters
        for i, ankle_idx in enumerate(self.ankle_joint_indices):
            kp[:, ankle_idx] = self.current_ankle_kp[:, i]
            kd[:, ankle_idx] = self.current_ankle_kd[:, i]
        
        # Compute torques using PD control: τ = kp * (q_target - q_current) - kd * dq_current
        torques = kp * (target_positions - joint_pos) - kd * joint_vel
        
        # Clip torques to limits if available
        if self.joint_effort_limits is not None:
            torques = torch.clamp(torques, -self.joint_effort_limits, self.joint_effort_limits)
        
        return torques, target_positions

    def _build_joint_effort_limits(self, robot: Articulation):
        """Create joint effort limit tensor per joint using actuator config."""
        joint_names = robot.data.joint_names
        num_envs = self.scene.num_envs
        num_joints = len(joint_names)
        limits = torch.full((num_joints,), float("inf"), dtype=torch.float32, device=self.device)
        actuator_cfgs = getattr(robot.cfg, "actuators", {})
        for actuator_cfg in actuator_cfgs.values():
            effort_limit = getattr(actuator_cfg, "effort_limit_sim", None)
            joint_exprs = getattr(actuator_cfg, "joint_names_expr", [])
            if effort_limit is None or not joint_exprs:
                continue
            for expr in joint_exprs:
                pattern = re.compile(expr)
                for idx, name in enumerate(joint_names):
                    if pattern.fullmatch(name) or pattern.match(name):
                        limits[idx] = float(effort_limit)
        if torch.isinf(limits).all():
            return None
        return limits.unsqueeze(0).repeat(num_envs, 1)

    def _debug_log_step(self, residual_actions, base_actions, torques, target_positions, robot):
        if not self.debug_log or self._debug_logs_emitted >= self.debug_max_logs:
            self._debug_step_counter += 1
            return
        if self._debug_step_counter % max(1, self.debug_log_interval) == 0:
            env_id = 0
            joint_pos = robot.data.joint_pos[env_id].detach().cpu().numpy()
            joint_vel = robot.data.joint_vel[env_id].detach().cpu().numpy()
            base_velocity_cmd = None
            if hasattr(self, "command_manager") and self.command_manager is not None:
                try:
                    base_velocity_cmd = (
                        self.command_manager.get_command("base_velocity")[env_id, :3].detach().cpu().tolist()
                    )
                except KeyError:
                    base_velocity_cmd = None
            base_obs_sample = None
            if hasattr(self, "_last_base_obs_env0"):
                base_obs_sample = self._last_base_obs_env0.detach().cpu().tolist()
            root_pos = robot.data.root_pos_w[env_id].detach().cpu().tolist()
            root_lin_vel = robot.data.root_lin_vel_w[env_id].detach().cpu().tolist()
            root_ang_vel = robot.data.root_ang_vel_w[env_id].detach().cpu().tolist()
            root_quat = robot.data.root_quat_w[env_id].detach().cpu().tolist()
            print(
                "[DEBUG] Step",
                self._debug_step_counter,
                {
                    "residual_actions": residual_actions[env_id].detach().cpu().tolist(),
                    "base_actions": base_actions[env_id].detach().cpu().tolist(),
                    "base_velocity_cmd": base_velocity_cmd,
                    "base_obs_env0": base_obs_sample,
                    "ankle_kp": self.current_ankle_kp[env_id].detach().cpu().tolist(),
                    "ankle_kd": self.current_ankle_kd[env_id].detach().cpu().tolist(),
                    "torques": torques[env_id].detach().cpu().tolist(),
                    "target_pos": target_positions[env_id].detach().cpu().tolist(),
                    "joint_pos": joint_pos.tolist(),
                    "joint_vel": joint_vel.tolist(),
                    "root_pos_w": root_pos,
                    "root_lin_vel_w": root_lin_vel,
                    "root_ang_vel_w": root_ang_vel,
                    "root_quat_w": root_quat,
                },
            )
            self._debug_logs_emitted += 1
        self._debug_step_counter += 1

    # ---------------------------------------------------------------------- #
    # Base policy helper utilities
    # ---------------------------------------------------------------------- #

    def _initialize_base_policy_memory(self):
        """Resize base policy recurrent state to match number of environments."""
        if not self.base_policy_loaded or not hasattr(self.base_policy, "memory"):
            return
        try:
            num_layers = getattr(self.base_policy.memory, "num_layers", 1)
            hidden_size = getattr(self.base_policy.memory, "hidden_size", self.base_policy.hidden_state.shape[-1])
        except AttributeError:
            return
        num_envs = self.scene.num_envs
        device = self.device
        hidden_state = torch.zeros(num_layers, num_envs, hidden_size, device=device)
        cell_state = torch.zeros(num_layers, num_envs, hidden_size, device=device)
        self.base_policy.hidden_state = hidden_state
        self.base_policy.cell_state = cell_state
        if hasattr(self.base_policy, "reset_memory"):
            self.base_policy.reset_memory()

    def _reset_base_policy_memory(self, env_ids: Sequence[int] | None = None):
        """Zero base policy recurrent state for specified environments."""
        if not (self.base_policy_loaded and hasattr(self.base_policy, "hidden_state")):
            return
        if env_ids is None:
            self.base_policy.hidden_state.zero_()
            if hasattr(self.base_policy, "cell_state"):
                self.base_policy.cell_state.zero_()
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(dtype=torch.long, device=self.base_policy.hidden_state.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.base_policy.hidden_state.device, dtype=torch.long)
        self.base_policy.hidden_state[:, env_ids_tensor, :] = 0.0
        if hasattr(self.base_policy, "cell_state"):
            self.base_policy.cell_state[:, env_ids_tensor, :] = 0.0

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and associated base policy memory."""
        super()._reset_idx(env_ids)
        self._reset_base_policy_memory(env_ids)

    def _get_base_policy_observations(self) -> torch.Tensor:
        """
        Construct observations for base policy (47 dimensions).
        
        Uses the same observation format as training:
        - Angular velocity (3D)
        - Projected gravity (3D)
        - Velocity commands (3D)
        - Joint positions relative to default (12D)
        - Joint velocities (12D)
        - Previous actions (12D)
        - Phase signal (2D: sin, cos)
        
        Returns:
            base_obs: Base policy observations [num_envs, 47]
        """
        robot: Articulation = self.scene["robot"]
        
        # Compute phase signal (same as training)
        period = 0.8
        phase_value = (self.episode_length_buf.float() * self.step_dt) % period / period
        self.phase = phase_value
        sin_phase = torch.sin(2 * np.pi * phase_value).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * phase_value).unsqueeze(1)
        
        # Get base angular velocity in body frame
        base_ang_vel = robot.data.root_ang_vel_b  # [num_envs, 3]
        
        # Compute projected gravity by rotating world gravity vector into body frame
        root_quat_w = robot.data.root_quat_w  # [num_envs, 4]
        gravity_vec_w = torch.zeros(self.scene.num_envs, 3, device=self.device, dtype=root_quat_w.dtype)
        gravity_vec_w[:, 2] = -1.0
        projected_gravity = math_utils.quat_apply_inverse(root_quat_w, gravity_vec_w)
        
        # Get velocity commands (from command manager) and scale them
        if hasattr(self, "command_manager") and self.command_manager is not None:
            try:
                commands = self.command_manager.get_command("base_velocity")[:, :3]
            except KeyError:
                commands = torch.zeros(self.scene.num_envs, 3, device=self.device, dtype=base_ang_vel.dtype)
        else:
            commands = torch.zeros(self.scene.num_envs, 3, device=self.device, dtype=base_ang_vel.dtype)
        command_scales = torch.tensor(self.base_obs_scales["cmd"], device=self.device, dtype=commands.dtype)
        scaled_commands = commands * command_scales
        
        # Get joint positions and velocities
        joint_pos = robot.data.joint_pos  # [num_envs, 12]
        joint_vel = robot.data.joint_vel  # [num_envs, 12]
        default_joint_pos = robot.data.default_joint_pos  # [num_envs, 12]
        
        # Get previous actions (last base policy actions)
        if not hasattr(self, "last_base_actions"):
            self.last_base_actions = torch.zeros(self.scene.num_envs, 12, device=self.device)
        
        # Construct base policy observation vector (47D)
        base_obs = torch.cat((
            base_ang_vel * self.base_obs_scales["ang_vel"],  # [0:3]
            projected_gravity,  # [3:6]
            scaled_commands,  # [6:9]
            (joint_pos - default_joint_pos) * self.base_obs_scales["dof_pos"],  # [9:21]
            joint_vel * self.base_obs_scales["dof_vel"],  # [21:33]
            self.last_base_actions,  # [33:45] - previous actions
            sin_phase,  # [45:46]
            cos_phase,  # [46:47]
        ), dim=-1)
        
        # Store current actions for next step
        # (We'll update this after getting base policy actions)
        
        return base_obs

    def reset(self, env_ids=None, **kwargs):
        """
        Reset the environment and update episode tracking.
        
        Args:
            env_ids: Environment IDs to reset (None = all)
            **kwargs: Additional arguments
            
        Returns:
            observations, infos (standard gymnasium format)
        """
        # Reset episode length buffer
        if env_ids is None:
            self.episode_length_buf.zero_()
        else:
            self.episode_length_buf[env_ids] = 0
        
        # Reset phase
        self.phase.zero_()
        
        # Reset last base actions
        if hasattr(self, "last_base_actions"):
            if env_ids is None:
                self.last_base_actions.zero_()
            else:
                self.last_base_actions[env_ids] = 0
        
        # Call parent reset
        return super().reset(env_ids=env_ids, **kwargs)

