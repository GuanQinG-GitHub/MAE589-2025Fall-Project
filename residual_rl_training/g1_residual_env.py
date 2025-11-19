"""
G1 Residual RL Environment
This environment implements residual RL for ankle impedance scheduling.
The base policy (pretrained on flat terrain) is frozen, and a residual
network learns to adapt ankle PD parameters for uneven terrain.
"""

import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

# Import base class - adjust path as needed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'unitree_rl_gym'))
from legged_gym.envs.base.legged_robot import LeggedRobot


class G1ResidualRobot(LeggedRobot):
    """
    G1 robot environment with residual RL for ankle impedance adaptation.
    
    Key features:
    - Freezes pretrained base policy (flat terrain)
    - Residual network outputs ankle impedance modifications
    - Observations: contact forces, contact binary, ankle torques
    - Rewards: locomotion + terrain adaptation
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        Initialize the residual RL environment.
        
        Args:
            cfg: Configuration object (G1ResidualCfg)
            sim_params: Simulation parameters
            physics_engine: Physics engine type
            sim_device: Device for simulation ('cuda' or 'cpu')
            headless: Whether to run without rendering
        """
        # Initialize base class
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # ====================================================================
        # BASE POLICY LOADING
        # ====================================================================
        # Load and freeze the pretrained base policy
        self.base_policy_path = cfg.base_policy_path  # Path to pretrained policy
        self.base_policy = None  # Will be loaded in _init_buffers
        self.base_policy_loaded = False
        
        # Base policy observation processing (same as training)
        self.base_obs_scales = {
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
            'cmd': [2.0, 2.0, 0.25]
        }
        
        # ====================================================================
        # ANKLE JOINT INDICES
        # ====================================================================
        # Find ankle joint indices for impedance modification
        # Ankle joints: left_ankle_pitch, left_ankle_roll, 
        #               right_ankle_pitch, right_ankle_roll
        self.ankle_joint_indices = []
        ankle_joint_names = [
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint'
        ]
        for name in ankle_joint_names:
            for i, dof_name in enumerate(self.dof_names):
                if name in dof_name:
                    self.ankle_joint_indices.append(i)
                    break
        
        if len(self.ankle_joint_indices) != 4:
            raise ValueError(f"Could not find all ankle joints. Found: {self.ankle_joint_indices}")
        
        # Base ankle PD parameters (from config)
        self.base_ankle_kp = self.p_gains[self.ankle_joint_indices[0]].item()  # Should be 40
        self.base_ankle_kd = self.d_gains[self.ankle_joint_indices[0]].item()  # Should be 2
        
        # ====================================================================
        # RESIDUAL IMPEDANCE BUFFERS
        # ====================================================================
        # Buffers for residual impedance modifications
        # Residual actions are in range [-1, 1], scaled to modify kp
        self.residual_kp_modifications = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )  # One modification per ankle joint
        
        # Current ankle PD parameters (base + residual modifications)
        self.current_ankle_kp = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self.current_ankle_kd = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        
        # Initialize to base values
        self.current_ankle_kp[:] = self.base_ankle_kp
        self.current_ankle_kd[:] = self.base_ankle_kd
        
        # ====================================================================
        # ANKLE TORQUE TRACKING
        # ====================================================================
        # Track ankle joint torques for observations
        self.ankle_torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self.last_ankle_torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
    
    def _init_buffers(self):
        """
        Initialize buffers and load base policy.
        Called after simulation is created.
        """
        # Initialize base class buffers
        super()._init_buffers()
        
        # Initialize foot state tracking
        self._init_foot()
        
        # ====================================================================
        # LOAD BASE POLICY
        # ====================================================================
        if self.base_policy_path and not self.base_policy_loaded:
            try:
                # Load pretrained policy
                self.base_policy = torch.jit.load(self.base_policy_path, map_location=self.device)
                self.base_policy.eval()  # Set to evaluation mode
                
                # Freeze the base policy (no gradients)
                for param in self.base_policy.parameters():
                    param.requires_grad = False
                
                self.base_policy_loaded = True
                print(f"Loaded base policy from: {self.base_policy_path}")
            except Exception as e:
                print(f"Warning: Could not load base policy: {e}")
                print("Training without base policy (not recommended)")
                self.base_policy = None
    
    def _init_foot(self):
        """
        Initialize foot state tracking buffers.
        """
        self.feet_num = len(self.feet_indices)
        
        # Acquire rigid body state tensor for foot tracking
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        
        # Extract foot states
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]  # Foot positions [x, y, z]
        self.feet_vel = self.feet_state[:, :, 7:10]  # Foot velocities [vx, vy, vz]
    
    def update_feet_state(self):
        """
        Update foot state from simulation.
        Called after each physics step.
        """
        # Refresh rigid body states from simulation
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Update foot states
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
    
    def _post_physics_step_callback(self):
        """
        Callback after physics step.
        Updates foot states and computes phase for base policy.
        """
        # Update foot states
        self.update_feet_state()
        
        # Compute phase for base policy (same as training)
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([
            self.phase_left.unsqueeze(1), 
            self.phase_right.unsqueeze(1)
        ], dim=-1)
        
        # Call parent callback
        return super()._post_physics_step_callback()
    
    def _compute_torques(self, actions):
        """
        Compute joint torques from residual actions.
        
        The residual actions modify ankle PD parameters, then we:
        1. Get base policy actions (frozen)
        2. Apply PD control with modified ankle parameters
        
        Args:
            actions: Residual actions [num_envs, 4] - ankle impedance modifications
            
        Returns:
            torques: Joint torques [num_envs, num_dof]
        """
        # ====================================================================
        # PROCESS RESIDUAL ACTIONS (ANKLE IMPEDANCE MODIFICATIONS)
        # ====================================================================
        # Residual actions are in range [-1, 1]
        # Scale to modify kp: kp_new = kp_base + action * scale * kp_base
        residual_scale = self.cfg.control.residual_kp_scale  # 0.25 (25% variation)
        
        # Compute kp modifications: [-0.25*kp_base, +0.25*kp_base]
        self.residual_kp_modifications = actions * residual_scale * self.base_ankle_kp
        
        # Compute new ankle kp values (clamped to [kp_min, kp_max])
        self.current_ankle_kp = torch.clamp(
            self.base_ankle_kp + self.residual_kp_modifications,
            min=self.cfg.control.kp_min,
            max=self.cfg.control.kp_max
        )
        
        # Compute new ankle kd values (proportional to kp)
        kd_ratio = self.cfg.control.residual_kd_ratio
        self.current_ankle_kd = self.base_ankle_kd * (
            self.current_ankle_kp / self.base_ankle_kp
        ) * kd_ratio + self.base_ankle_kd * (1 - kd_ratio)
        
        # ====================================================================
        # GET BASE POLICY ACTIONS
        # ====================================================================
        if self.base_policy is not None:
            # Construct base policy observations
            base_obs = self._get_base_policy_observations()
            
            # Get base policy actions (no gradients)
            with torch.no_grad():
                base_actions = self.base_policy(base_obs)
            
            # Convert to numpy if needed, then back to tensor
            if isinstance(base_actions, torch.Tensor):
                base_actions = base_actions.detach()
            else:
                base_actions = torch.from_numpy(base_actions).to(self.device)
            
            # Ensure correct shape [num_envs, 12]
            if base_actions.dim() == 1:
                base_actions = base_actions.unsqueeze(0).repeat(self.num_envs, 1)
        else:
            # No base policy: use zero actions (for testing)
            base_actions = torch.zeros(
                self.num_envs, self.num_actions, 
                dtype=torch.float, device=self.device
            )
        
        # ====================================================================
        # COMPUTE TORQUES WITH MODIFIED ANKLE PD PARAMETERS
        # ====================================================================
        # Scale base actions
        actions_scaled = base_actions * self.cfg.control.action_scale
        
        # Compute torques for all joints using PD control
        # For non-ankle joints: use base PD parameters
        # For ankle joints: use modified PD parameters
        torques = torch.zeros_like(self.torques)
        
        # Compute torques for each joint
        for i in range(self.num_actions):
            if i in self.ankle_joint_indices:
                # Ankle joint: use modified PD parameters
                ankle_idx = self.ankle_joint_indices.index(i)
                kp = self.current_ankle_kp[:, ankle_idx]
                kd = self.current_ankle_kd[:, ankle_idx]
            else:
                # Non-ankle joint: use base PD parameters
                kp = self.p_gains[i]
                kd = self.d_gains[i]
            
            # PD control: tau = kp * (q_target - q_current) - kd * dq_current
            q_target = actions_scaled[:, i] + self.default_dof_pos[0, i]
            q_current = self.dof_pos[:, i]
            dq_current = self.dof_vel[:, i]
            
            torques[:, i] = kp * (q_target - q_current) - kd * dq_current
        
        # Clip torques to limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _get_base_policy_observations(self):
        """
        Construct observations for base policy.
        Uses the same observation format as training (47 dimensions).
        
        Returns:
            base_obs: Base policy observations [num_envs, 47]
        """
        # Compute phase signal (same as training)
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        
        # Construct base policy observation vector (47D)
        base_obs = torch.cat((
            self.base_ang_vel * self.base_obs_scales['ang_vel'],  # [0:3]
            self.projected_gravity,  # [3:6]
            self.commands[:, :3] * torch.tensor(
                self.base_obs_scales['cmd'], device=self.device
            ),  # [6:9]
            (self.dof_pos - self.default_dof_pos) * self.base_obs_scales['dof_pos'],  # [9:21]
            self.dof_vel * self.base_obs_scales['dof_vel'],  # [21:33]
            self.last_actions,  # [33:45] - previous actions
            sin_phase,  # [45:46]
            cos_phase,  # [46:47]
        ), dim=-1)
        
        return base_obs
    
    def compute_observations(self):
        """
        Compute observations for residual policy.
        
        Observations (12D):
        - Contact forces: 6D (3D per foot, 2 feet)
        - Contact binary: 2D (one per foot)
        - Ankle torques: 4D (one per ankle joint)
        
        Additional observations (commented out for future use):
        - Foot heights: 2D
        - Foot velocities: 6D
        - Base orientation: 2D
        - Contact force history: 6D * N
        """
        # ====================================================================
        # CONTACT FORCES (6D)
        # ====================================================================
        # Extract contact forces on feet (world frame, 3D per foot)
        left_foot_force = self.contact_forces[:, self.feet_indices[0], :]  # [num_envs, 3]
        right_foot_force = self.contact_forces[:, self.feet_indices[1], :]  # [num_envs, 3]
        
        # Normalize contact forces by body weight (~200N for G1)
        # Use log-scale for better distribution
        body_weight = 200.0  # Approximate body weight in Newtons
        left_foot_force_norm = left_foot_force / body_weight
        right_foot_force_norm = right_foot_force / body_weight
        
        # ====================================================================
        # CONTACT BINARY INDICATORS (2D)
        # ====================================================================
        # Binary indicators: 1 if foot is in contact, 0 otherwise
        # Threshold: vertical force > 1N
        left_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 1.0).float()
        right_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 1.0).float()
        
        # ====================================================================
        # ANKLE JOINT TORQUES (4D)
        # ====================================================================
        # Extract torques applied to ankle joints
        # These are computed from PD controller with modified parameters
        ankle_torques = self.torques[:, self.ankle_joint_indices]  # [num_envs, 4]
        
        # Normalize by torque limits (ankle torque limit ~50 N*m)
        ankle_torque_limit = 50.0
        ankle_torques_norm = ankle_torques / ankle_torque_limit
        
        # Store for reward computation
        self.ankle_torques = ankle_torques
        
        # ====================================================================
        # ADDITIONAL OBSERVATIONS (COMMENTED OUT FOR FUTURE USE)
        # ====================================================================
        # Foot heights relative to base (2D)
        # left_foot_height = self.feet_pos[:, 0, 2] - self.base_pos[:, 2]
        # right_foot_height = self.feet_pos[:, 1, 2] - self.base_pos[:, 2]
        # foot_heights = torch.stack([left_foot_height, right_foot_height], dim=1)
        
        # Foot velocities (6D)
        # foot_velocities = self.feet_vel.view(self.num_envs, -1)  # [num_envs, 6]
        
        # Base orientation (pitch and roll) (2D)
        # base_pitch = self.rpy[:, 1]  # Pitch angle
        # base_roll = self.rpy[:, 0]   # Roll angle
        # base_orientation = torch.stack([base_pitch, base_roll], dim=1)
        
        # Contact force history (6D * N timesteps)
        # Would require maintaining a history buffer
        
        # ====================================================================
        # CONCATENATE OBSERVATIONS
        # ====================================================================
        self.obs_buf = torch.cat((
            left_foot_force_norm,      # [0:3]   - Left foot contact force
            right_foot_force_norm,     # [3:6]   - Right foot contact force
            left_contact.unsqueeze(1), # [6:7]   - Left foot contact binary
            right_contact.unsqueeze(1), # [7:8]  - Right foot contact binary
            ankle_torques_norm,        # [8:12]  - Ankle joint torques
        ), dim=-1)
        
        # Clip observations to prevent extreme values
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        
        # Add noise if configured
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _get_noise_scale_vec(self, cfg):
        """
        Set noise scale vector for observations.
        Must match observation structure.
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # Contact forces: small noise
        noise_vec[0:6] = 0.01 * noise_level
        
        # Contact binary: no noise (discrete)
        noise_vec[6:8] = 0.0
        
        # Ankle torques: small noise
        noise_vec[8:12] = 0.01 * noise_level
        
        return noise_vec
    
    # ====================================================================
    # REWARD FUNCTIONS
    # ====================================================================
    
    def _reward_terrain_adaptation(self):
        """
        Reward for successful terrain traversal.
        Encourages the robot to maintain forward progress on uneven terrain.
        """
        # Forward velocity reward (encourages forward progress)
        forward_vel = self.base_lin_vel[:, 0]  # Forward velocity (x-direction)
        desired_vel = self.commands[:, 0]  # Desired forward velocity
        
        # Reward for tracking desired velocity
        vel_error = torch.abs(forward_vel - desired_vel)
        vel_reward = torch.exp(-vel_error / 0.25)  # Exponential reward
        
        # Additional reward for maintaining base height (stability)
        base_height_error = torch.abs(self.base_pos[:, 2] - self.cfg.rewards.base_height_target)
        height_reward = torch.exp(-base_height_error / 0.1)
        
        # Combined terrain adaptation reward
        return vel_reward * height_reward
    
    def _reward_impedance_smoothness(self):
        """
        Penalize rapid changes in impedance.
        Encourages smooth impedance scheduling.
        """
        # Compute impedance change rate
        kp_change = torch.abs(
            self.current_ankle_kp - self.last_ankle_kp
        ) if hasattr(self, 'last_ankle_kp') else torch.zeros_like(self.current_ankle_kp)
        
        # Penalize large changes (smoothness)
        smoothness_penalty = torch.sum(kp_change, dim=1)
        
        # Store current for next step
        self.last_ankle_kp = self.current_ankle_kp.clone()
        
        return -smoothness_penalty  # Negative because it's a penalty
    
    def _reward_contact(self):
        """
        Reward for proper foot contact timing.
        Same as base policy training.
        """
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55  # Expected stance phase
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0  # Vertical force > 1N
            res += ~(contact ^ is_stance).float()  # Reward when contact matches expected phase
        return res
    
    def _reward_contact_no_vel(self):
        """
        Penalize sliding (contact with foot velocity).
        Same as base policy training.
        """
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return -torch.sum(penalize, dim=(1, 2))  # Negative because it's a penalty
    
    def _reward_tracking_lin_vel(self):
        """Reward for tracking desired linear velocity."""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)
    
    def _reward_tracking_ang_vel(self):
        """Reward for tracking desired angular velocity."""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / 0.25)
    
    def _reward_lin_vel_z(self):
        """Penalize vertical velocity."""
        return -torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        """Penalize base angular velocity in x and y."""
        return -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        """Penalize base orientation deviation."""
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_base_height(self):
        """Penalize base height deviation from target."""
        base_height_error = self.base_pos[:, 2] - self.cfg.rewards.base_height_target
        return -torch.square(base_height_error)
    
    def _reward_dof_acc(self):
        """Penalize joint acceleration (smoothness)."""
        dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        return -torch.sum(torch.square(dof_acc), dim=1)
    
    def _reward_dof_vel(self):
        """Penalize joint velocity."""
        return -torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_pos_limits(self):
        """Penalize joint position limit violations."""
        # Get joint limits from URDF (simplified - would need actual limits)
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    
    def _reward_action_rate(self):
        """Penalize action rate (smoothness of impedance changes)."""
        if hasattr(self, 'last_residual_actions'):
            action_rate = torch.sum(
                torch.square(self.actions - self.last_residual_actions), dim=1
            )
            self.last_residual_actions = self.actions.clone()
        else:
            action_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.last_residual_actions = self.actions.clone()
        return -action_rate
    
    def _reward_alive(self):
        """Reward for staying alive."""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
    
    def _reward_hip_pos(self):
        """Penalize hip position deviation."""
        hip_indices = [1, 2, 7, 8]  # Left and right hip roll and yaw
        return -torch.sum(torch.square(self.dof_pos[:, hip_indices]), dim=1)
    
    def _reward_feet_swing_height(self):
        """Penalize low foot swing height."""
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return -torch.sum(pos_error, dim=1)  # Negative because it's a penalty

