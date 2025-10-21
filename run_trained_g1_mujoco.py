#!/usr/bin/env python3
"""
Script to run the trained Unitree G1-29dof model with MuJoCo

This script loads your trained model from trained_models/model_final.pt and runs it
in a MuJoCo environment with the 29-DOF G1 robot configuration.

Based on the unitree_rl_lab Sim2Sim deployment workflow.

Usage:
    python run_trained_g1_mujoco.py

Requirements:
    - MuJoCo installed
    - PyTorch installed
    - Trained model in trained_models/model_final.pt
    - unitree_mujoco scene files
"""

import time
import pathlib
import threading
import queue
import sys
import termios
import tty
import select
import yaml

import numpy as np
import torch
import mujoco
import mujoco.viewer


class G1MuJoCoController:
    """Controller for Unitree G1-29dof using trained model with MuJoCo"""
    
    def __init__(self):
        self.project_root = pathlib.Path(__file__).parent
        
        # Model paths - try to find the 29-DOF scene
        self.xml_path = self._find_scene_file()
        
        # Load deployment configuration
        self.deploy_cfg = self._load_deploy_config()
        
        # Control parameters
        self.simulation_dt = 0.002  # 500Hz simulation
        self.control_dt = 0.02      # 50Hz control (from deploy.yaml)
        self.control_decimation = int(self.control_dt / self.simulation_dt)
        
        # Load trained model
        self.policy = self._load_trained_model()
        
        # State variables
        self.cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, ωz]
        self.action = np.zeros(29, dtype=np.float32)
        self.target_dof_pos = np.array(self.deploy_cfg['default_joint_pos'], dtype=np.float32)
        self.obs = np.zeros(480, dtype=np.float32)  # 29-DOF observation space
        self.counter = 0
        
        # Control thread
        self.control_queue = queue.Queue()
        self.running = False
        
        # Load MuJoCo model
        self._load_mujoco_model()
    
    def _find_scene_file(self):
        """Find the appropriate MuJoCo scene file"""
        # Try different possible locations for 29-DOF scene
        possible_paths = [
            self.project_root / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml",
            self.project_root / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene.xml",
            self.project_root / "external" / "unitree_mujoco" / "simulate" / "scene_29dof.xml",
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found scene file: {path}")
                return path
        
        # If not found, use the 12-DOF scene as fallback
        fallback_path = self.project_root / "external" / "unitree_rl_gym" / "resources" / "robots" / "g1_description" / "scene.xml"
        if fallback_path.exists():
            print(f"Warning: Using 12-DOF scene as fallback: {fallback_path}")
            print("Note: This may cause issues with 29-DOF model. Consider using the proper 29-DOF scene.")
            return fallback_path
        
        raise FileNotFoundError("Could not find G1 scene file")
    
    def _load_deploy_config(self):
        """Load deployment configuration"""
        config_path = self.project_root / "unitree_rl_lab" / "deploy" / "robots" / "g1_29dof" / "config" / "policy" / "velocity" / "v0" / "params" / "deploy.yaml"
        
        if not config_path.exists():
            # Use default configuration if file doesn't exist
            return {
                'joint_ids_map': list(range(29)),
                'step_dt': 0.02,
                'stiffness': [100.0] * 12 + [200.0] * 3 + [40.0] * 14,  # 29 DOF
                'damping': [2.0] * 12 + [5.0] * 3 + [10.0] * 14,  # 29 DOF
                'default_joint_pos': [0.0] * 29,
                'actions': {
                    'JointPositionAction': {
                        'scale': [0.25] * 29,
                        'offset': [0.0] * 29
                    }
                },
                'observations': {
                    'base_ang_vel': {'scale': [0.2, 0.2, 0.2]},
                    'projected_gravity': {'scale': [1.0, 1.0, 1.0]},
                    'velocity_commands': {'scale': [1.0, 1.0, 1.0]},
                    'joint_pos_rel': {'scale': [1.0] * 29},
                    'joint_vel_rel': {'scale': [0.05] * 29},
                    'last_action': {'scale': [1.0] * 29}
                }
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_trained_model(self):
        """Load the trained model"""
        model_path = self.project_root / "trained_models" / "model_25000.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract the policy network
        if 'actor' in checkpoint:
            policy = checkpoint['actor']
            print(f"✓ Loaded actor from checkpoint")
        elif 'model_state_dict' in checkpoint:
            # If it's a full checkpoint, we need to reconstruct the model
            from collections import OrderedDict
            state_dict = checkpoint['model_state_dict']
            # Filter for actor parameters
            actor_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'actor' in key or 'policy' in key:
                    actor_state_dict[key.replace('actor.', '').replace('policy.', '')] = value
            
            # Create a simple policy network (you'll need to adjust this based on your actual architecture)
            policy = torch.nn.Sequential(
                torch.nn.Linear(480, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 29)
            )
            policy.load_state_dict(actor_state_dict, strict=False)
            print(f"✓ Reconstructed policy from model_state_dict")
        else:
            # Assume the checkpoint is the policy itself
            policy = checkpoint
            print(f"✓ Loaded checkpoint as policy directly")
        
        policy.eval()
        print(f"✓ Loaded trained model from: {model_path}")
        
        # Test the policy with a dummy input to check for issues
        dummy_obs = torch.zeros(1, 480)
        with torch.no_grad():
            dummy_action = policy(dummy_obs)
            print(f"✓ Policy test - Input shape: {dummy_obs.shape}, Output shape: {dummy_action.shape}")
            print(f"✓ Policy test - Output range: [{dummy_action.min().item():.3f}, {dummy_action.max().item():.3f}]")
            
            # Check for NaN or Inf values
            if torch.isnan(dummy_action).any() or torch.isinf(dummy_action).any():
                print("⚠️  WARNING: Policy output contains NaN or Inf values!")
            else:
                print("✓ Policy output is valid (no NaN/Inf)")
        
        return policy
    
    def _load_mujoco_model(self):
        """Load MuJoCo model"""
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt
        
        print(f"✓ Loaded MuJoCo model: {self.xml_path}")
        print(f"  - DOF: {self.model.nv}")
        print(f"  - Actuators: {self.model.nu}")
    
    def get_gravity_orientation(self, quaternion):
        """Convert quaternion to gravity orientation vector"""
        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        return gravity_orientation
    
    def construct_observation(self):
        """Construct observation vector for the trained model"""
        # Get current state
        qj = self.data.qpos[7:]  # Joint positions (skip base position and quaternion)
        dqj = self.data.qvel[6:]  # Joint velocities (skip base linear and angular velocity)
        quat = self.data.qpos[3:7]  # Base quaternion
        omega = self.data.qvel[3:6]  # Base angular velocity
        
        # Get gravity orientation
        gravity_orientation = self.get_gravity_orientation(quat)
        
        # Scale observations according to deploy config
        obs_scales = self.deploy_cfg['observations']
        
        # Base angular velocity
        omega_scaled = omega * np.array(obs_scales['base_ang_vel']['scale'])
        
        # Projected gravity
        gravity_scaled = gravity_orientation * np.array(obs_scales['projected_gravity']['scale'])
        
        # Velocity commands
        cmd_scaled = self.cmd_vel * np.array(obs_scales['velocity_commands']['scale'])
        
        # Joint positions (relative to default)
        default_pos = np.array(self.deploy_cfg['default_joint_pos'])
        qj_rel = (qj - default_pos) * np.array(obs_scales['joint_pos_rel']['scale'])
        
        # Joint velocities
        dqj_scaled = dqj * np.array(obs_scales['joint_vel_rel']['scale'])
        
        # Last action
        action_scaled = self.action * np.array(obs_scales['last_action']['scale'])
        
        # Construct observation vector (480 dimensions for 29-DOF with history)
        obs = np.zeros(480, dtype=np.float32)
        
        # Fill observation (this is a simplified version - you may need to adjust based on your actual observation structure)
        idx = 0
        obs[idx:idx+3] = omega_scaled
        idx += 3
        obs[idx:idx+3] = gravity_scaled
        idx += 3
        obs[idx:idx+3] = cmd_scaled
        idx += 3
        obs[idx:idx+29] = qj_rel
        idx += 29
        obs[idx:idx+29] = dqj_scaled
        idx += 29
        obs[idx:idx+29] = action_scaled
        idx += 29
        
        # Fill remaining with zeros (for history and other features)
        # You may need to adjust this based on your actual observation structure
        
        # Check for NaN or Inf values in observation
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️  WARNING: Observation contains NaN or Inf values!")
            print(f"  - omega_scaled: {omega_scaled}")
            print(f"  - gravity_scaled: {gravity_scaled}")
            print(f"  - cmd_scaled: {cmd_scaled}")
            print(f"  - qj_rel: {qj_rel}")
            print(f"  - dqj_scaled: {dqj_scaled}")
            print(f"  - action_scaled: {action_scaled}")
            # Replace NaN/Inf with zeros
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def update_policy(self):
        """Update policy and compute new action"""
        if self.counter % self.control_decimation == 0:
            # Construct observation
            self.obs = self.construct_observation()
            
            # Policy inference
            obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
            with torch.no_grad():
                self.action = self.policy(obs_tensor).detach().numpy().squeeze()
            
            # Convert action to target joint positions
            action_scale = np.array(self.deploy_cfg['actions']['JointPositionAction']['scale'])
            action_offset = np.array(self.deploy_cfg['actions']['JointPositionAction']['offset'])
            self.target_dof_pos = self.action * action_scale + action_offset
    
    def apply_control(self):
        """Apply PD control to joints"""
        # Get current joint positions and velocities
        current_q = self.data.qpos[7:]
        current_dq = self.data.qvel[6:]
        
        # PD control
        kps = np.array(self.deploy_cfg['stiffness'])
        kds = np.array(self.deploy_cfg['damping'])
        
        tau = (self.target_dof_pos - current_q) * kps + (0.0 - current_dq) * kds
        
        # Check for NaN or Inf values in control
        if np.isnan(tau).any() or np.isinf(tau).any():
            print(f"⚠️  WARNING: Control torques contain NaN or Inf values!")
            print(f"  - target_dof_pos: {self.target_dof_pos}")
            print(f"  - current_q: {current_q}")
            print(f"  - current_dq: {current_dq}")
            print(f"  - kps: {kps}")
            print(f"  - kds: {kds}")
            print(f"  - tau: {tau}")
            # Replace NaN/Inf with zeros
            tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply torques
        self.data.ctrl[:] = tau
    
    def step_simulation(self):
        """Step the simulation and update control"""
        self.update_policy()
        self.apply_control()
        mujoco.mj_step(self.model, self.data)
        self.counter += 1
    
    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        """Set velocity command"""
        self.cmd_vel = np.array([vx, vy, wz], dtype=np.float32)
        print(f"Command set: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")
    
    def reset_robot(self):
        """Reset robot to default standing pose"""
        default_pos = np.array(self.deploy_cfg['default_joint_pos'])
        self.data.qpos[7:] = default_pos
        self.data.qvel[6:] = 0.0
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.action = np.zeros(29)
        self.target_dof_pos = default_pos.copy()
        print("Robot reset to standing pose")
    
    def print_help(self):
        """Print control commands"""
        print("\n=== G1-29dof MuJoCo Control Commands ===")
        print("w/s: Forward/Backward")
        print("a/d: Strafe Left/Right")
        print("q/e: Turn Left/Right")
        print("space: Stop")
        print("r: Reset to standing")
        print("h: Help")
        print("x: Quit")
        print("==========================================\n")


def get_key():
    """Get a single keypress from stdin"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def control_thread(controller):
    """Thread for handling keyboard input"""
    print("Control thread started. Press 'h' for help.")
    
    while controller.running:
        try:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = get_key().lower()
                
                if key == 'w':
                    controller.set_command(vx=0.5, vy=0.0, wz=0.0)
                elif key == 's':
                    controller.set_command(vx=-0.5, vy=0.0, wz=0.0)
                elif key == 'a':
                    controller.set_command(vx=0.0, vy=0.3, wz=0.0)
                elif key == 'd':
                    controller.set_command(vx=0.0, vy=-0.3, wz=0.0)
                elif key == 'q':
                    controller.set_command(vx=0.0, vy=0.0, wz=0.5)
                elif key == 'e':
                    controller.set_command(vx=0.0, vy=0.0, wz=-0.5)
                elif key == ' ':
                    controller.set_command(vx=0.0, vy=0.0, wz=0.0)
                elif key == 'r':
                    controller.reset_robot()
                elif key == 'h':
                    controller.print_help()
                elif key == 'x':
                    print("Exiting...")
                    controller.running = False
                    break
                    
        except KeyboardInterrupt:
            controller.running = False
            break


def main():
    """Main function"""
    print("=== Unitree G1-29dof Trained Model with MuJoCo ===")
    print("Loading model and policy...")
    
    try:
        # Initialize controller
        controller = G1MuJoCoController()
        controller.running = True
        
        # Start control thread
        control_thread_obj = threading.Thread(target=control_thread, args=(controller,))
        control_thread_obj.daemon = True
        control_thread_obj.start()
        
        # Print initial help
        controller.print_help()
        
        # Launch MuJoCo viewer
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            print("MuJoCo viewer launched. Use keyboard controls to move the robot.")
            
            while viewer.is_running() and controller.running:
                step_start = time.time()
                
                # Step simulation
                controller.step_simulation()
                
                # Sync viewer
                viewer.sync()
                
                # Maintain real-time simulation
                time_until_next_step = controller.simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        controller.running = False
        print("Simulation ended.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the required files exist:")
        print("- trained_models/model_final.pt")
        print("- unitree_mujoco scene files")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
