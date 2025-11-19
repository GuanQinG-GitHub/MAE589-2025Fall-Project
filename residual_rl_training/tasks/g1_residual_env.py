# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom environment for G1 Residual RL with base policy integration."""

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

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
        Override step function to integrate base policy and residual actions.
        
        This method intercepts the action before it's applied, gets base policy
        actions, computes torques with modified PD parameters, and applies them.
        
        **Important**: We apply torques directly, bypassing the normal action
        application. The parent's step() may still process actions through the
        action manager, but those processed actions should not be applied to
        the robot since we've already applied torques directly.
        
        Args:
            action: Residual actions from RL policy [num_envs, 4] (ankle PD modifications)
            
        Returns:
            observations, rewards, dones, infos (standard gymnasium format)
        """
        # Update episode length for phase computation
        self.episode_length_buf += 1
        
        # Apply our custom action processing (computes and applies torques)
        self._apply_residual_action(action)
        
        # Call parent step to handle simulation stepping, observations, rewards, etc.
        # Note: The parent may process actions through the action manager, but since
        # we've already applied torques directly, those processed actions won't affect
        # the robot. The parent's step() will handle simulation stepping and observation/reward computation.
        return super().step(action)
    
    def _apply_residual_action(self, action: torch.Tensor):
        """
        Override action application to integrate base policy and residual actions.
        
        This method is called internally by Isaac Lab to apply actions to the robot.
        We intercept here to:
        1. Get base policy actions (12 DOF)
        2. Get residual actions (4 DOF) 
        3. Compute torques with modified PD parameters
        4. Apply torques directly
        
        Args:
            action: Residual actions from RL policy [num_envs, 4] (ankle PD modifications)
        """
        # Convert action to numpy if needed for action manager
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
        
        # Process residual actions through action manager (scaling/clipping)
        # This gives us processed residual actions (4 DOF)
        self.action_manager.process_action(action_np)
        processed_residual_actions_list = self.action_manager.processed_actions()
        
        # Convert to tensor
        if isinstance(processed_residual_actions_list, list):
            processed_residual_actions = torch.tensor(
                processed_residual_actions_list, device=self.device, dtype=torch.float32
            )
        else:
            processed_residual_actions = torch.from_numpy(
                np.array(processed_residual_actions_list, dtype=np.float32)
            ).to(self.device)
        
        # Ensure correct shape [num_envs, 4]
        num_envs = self.scene.num_envs
        if processed_residual_actions.dim() == 1:
            processed_residual_actions = processed_residual_actions.unsqueeze(0).repeat(num_envs, 1)
        elif processed_residual_actions.shape[0] != num_envs:
            # If single action, repeat for all environments
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
        torques = self._compute_torques_with_modified_pd(base_actions, processed_residual_actions)
        
        # ====================================================================
        # APPLY TORQUES DIRECTLY TO ROBOT
        # ====================================================================
        robot: Articulation = self.scene["robot"]
        robot.set_joint_effort_target(torques)
        
        # Note: We've applied torques directly, so we bypass the normal action application
        # The parent class's step() will handle simulation stepping, observations, rewards, etc.

    def _compute_torques_with_modified_pd(self, base_actions: torch.Tensor, residual_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute joint torques with modified ankle PD parameters.
        
        Args:
            base_actions: Base policy actions [num_envs, 12] (desired positions)
            residual_actions: Residual actions [num_envs, 4] (not used here, already processed)
            
        Returns:
            torques: Joint torques [num_envs, 12]
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
        
        # Clip torques to limits
        torque_limits = robot.data.joint_effort_limit
        torques = torch.clamp(torques, -torque_limits, torque_limits)
        
        return torques

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
        
        # Get base angular velocity
        base_ang_vel = robot.data.root_ang_vel_b  # [num_envs, 3]
        
        # Get projected gravity (in base frame)
        # Projected gravity is the gravity vector rotated to base frame
        # For now, we'll use a simplified version
        # In full implementation, this would be computed from base orientation
        projected_gravity = torch.zeros(self.scene.num_envs, 3, device=self.device)
        # TODO: Compute proper projected gravity from base orientation
        
        # Get velocity commands (from command manager)
        # For now, use zero commands or get from config
        commands = torch.zeros(self.scene.num_envs, 3, device=self.device)
        if hasattr(self, "command_manager"):
            # Try to get commands from command manager if available
            pass
        
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
            commands * torch.tensor(self.base_obs_scales["cmd"], device=self.device),  # [6:9]
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

