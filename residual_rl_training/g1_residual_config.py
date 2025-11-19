"""
Configuration file for G1 Residual RL Training
This config defines the environment and training parameters for learning
ankle impedance scheduling to adapt flat-terrain policy to uneven terrain.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1ResidualCfg(LeggedRobotCfg):
    """
    Configuration for G1 residual RL environment.
    Inherits from LeggedRobotCfg and modifies for residual learning.
    """
    
    # Base policy path (will be set in training script)
    base_policy_path = None
    
    class init_state(LeggedRobotCfg.init_state):
        """Initial state configuration"""
        pos = [0.0, 0.0, 0.8]  # Initial base position [x, y, z] in meters
        
        # Default joint angles (same as base policy training)
        default_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'torso_joint': 0.0
        }
    
    class env(LeggedRobotCfg.env):
        """Environment configuration"""
        # Observation space: contact forces (6D) + contact binary (2D) + ankle torques (4D) = 12D
        # Note: We can add more observations later (commented out)
        num_observations = 12  # Current: contact forces + contact binary + ankle torques
        num_privileged_obs = None  # Not using privileged observations for residual policy
        num_actions = 4  # Residual outputs: [left_ankle_pitch_kp, left_ankle_roll_kp, 
                         #                    right_ankle_pitch_kp, right_ankle_roll_kp]
                         # Note: We modify kp only, kd can be scaled proportionally
    
    class terrain(LeggedRobotCfg.terrain):
        """Terrain configuration - use uneven terrain for training"""
        mesh_type = 'trimesh'  # Use mesh-based terrain
        curriculum = True  # Gradually increase terrain difficulty
        # Terrain parameters for uneven terrain
        measure_heights = True
        measured_points_x = [-0.8, -0.4, 0.0, 0.4, 0.8]  # Measurement points
        measured_points_y = [-0.8, -0.4, 0.0, 0.4, 0.8]
        selected = False  # Don't use selected terrain
        max_init_terrain_level = 5  # Maximum initial terrain difficulty
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # Number of terrain rows
        num_cols = 20  # Number of terrain columns
        horizontal_scale = 0.1  # Horizontal scale of terrain
        vertical_scale = 0.05  # Vertical scale (height variation)
        border_size = 5.0
        slope_threshold = 0.75
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        """Domain randomization for sim-to-real transfer"""
        randomize_friction = True
        friction_range = [0.1, 1.25]  # Friction coefficient range
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]  # Mass variation range
        push_robots = True  # Apply random pushes
        push_interval_s = 5  # Push interval in seconds
        max_push_vel_xy = 1.5  # Maximum push velocity
    
    class control(LeggedRobotCfg.control):
        """Control configuration"""
        control_type = 'P'  # Position control
        
        # Base PD parameters (from flat terrain training)
        # These will be modified by residual network for ankle joints
        stiffness = {
            'hip_yaw': 100,
            'hip_roll': 100,
            'hip_pitch': 100,
            'knee': 150,
            'ankle': 40,  # Base ankle stiffness (will be modified by residual)
        }  # [N*m/rad]
        
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2,  # Base ankle damping (will be modified by residual)
        }  # [N*m*s/rad]
        
        action_scale = 0.25  # Action scale for base policy
        decimation = 4  # Control decimation (policy runs at lower frequency)
        
        # Residual impedance modification parameters
        residual_kp_scale = 0.25  # Scale for residual kp modification (25% variation)
        residual_kd_ratio = 0.5  # kd = kd_base * (kp / kp_base) * ratio (proportional scaling)
        kp_min = 20.0  # Minimum ankle kp (50% of base)
        kp_max = 60.0  # Maximum ankle kp (150% of base)
    
    class asset(LeggedRobotCfg.asset):
        """Asset configuration"""
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"  # Foot body name for contact detection
        penalize_contacts_on = ["hip", "knee"]  # Bodies that should not contact
        terminate_after_contacts_on = ["pelvis"]  # Bodies that cause termination
        self_collisions = 0  # Disable self-collisions
        flip_visual_attachments = False
    
    class rewards(LeggedRobotCfg.rewards):
        """Reward function configuration"""
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78  # Target base height
        
        class scales(LeggedRobotCfg.rewards.scales):
            """Reward scales - tuned for residual learning"""
            # Locomotion rewards (encourage walking)
            tracking_lin_vel = 1.0  # Track desired linear velocity
            tracking_ang_vel = 0.5  # Track desired angular velocity
            lin_vel_z = -2.0  # Penalize vertical velocity
            ang_vel_xy = -0.05  # Penalize base angular velocity
            
            # Stability rewards
            orientation = -1.0  # Penalize base orientation deviation
            base_height = -10.0  # Penalize base height deviation
            
            # Joint rewards
            dof_acc = -2.5e-7  # Penalize joint acceleration (smoothness)
            dof_vel = -1e-3  # Penalize joint velocity
            dof_pos_limits = -5.0  # Penalize joint limit violations
            
            # Contact rewards
            contact = 0.18  # Reward proper contact timing
            contact_no_vel = -0.2  # Penalize sliding (contact with velocity)
            feet_air_time = 0.0  # Not using air time reward
            
            # Terrain adaptation rewards (new for residual learning)
            terrain_adaptation = 1.0  # Reward for successful terrain traversal
            impedance_smoothness = -0.1  # Penalize rapid impedance changes
            
            # General rewards
            action_rate = -0.01  # Penalize action rate (smoothness)
            alive = 0.15  # Reward for staying alive
            hip_pos = -1.0  # Penalize hip position deviation
            feet_swing_height = -20.0  # Penalize low foot swing height
            collision = 0.0  # Not using collision reward


class G1ResidualCfgPPO(LeggedRobotCfgPPO):
    """
    PPO configuration for residual RL training.
    Smaller network since residual policy has simpler task.
    """
    
    class policy:
        """Policy network configuration"""
        init_noise_std = 0.5  # Initial exploration noise (lower than base policy)
        actor_hidden_dims = [128, 64]  # Smaller network for residual policy
        critic_hidden_dims = [128, 64]
        activation = 'elu'  # Activation function
        # Not using RNN for residual policy (simpler, faster)
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO algorithm parameters"""
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2  # PPO clip parameter
        entropy_coef = 0.01  # Entropy coefficient for exploration
        num_learning_epochs = 5  # Number of optimization epochs per update
        num_mini_batches = 4  # Number of mini-batches
        learning_rate = 1.e-3  # Learning rate
        schedule = 'adaptive'  # Learning rate schedule
        gamma = 0.99  # Discount factor
        lam = 0.95  # GAE lambda
        desired_kl = 0.01  # Desired KL divergence
        max_grad_norm = 1.0  # Gradient clipping
    
    class runner(LeggedRobotCfgPPO.runner):
        """Training runner configuration"""
        policy_class_name = 'ActorCritic'  # Not using recurrent policy
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # Steps per environment per iteration
        max_iterations = 5000  # Maximum training iterations
        
        # Logging
        save_interval = 50  # Save checkpoint every N iterations
        experiment_name = 'g1_residual_impedance'  # Experiment name
        run_name = ''  # Run name (will be set automatically)
        
        # Resume training
        resume = False  # Start from scratch
        load_run = -1  # Load latest run
        checkpoint = -1  # Load latest checkpoint

