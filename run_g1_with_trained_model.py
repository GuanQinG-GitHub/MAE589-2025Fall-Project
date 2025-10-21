#!/usr/bin/env python3
"""
Simplified script to run trained G1 model with MuJoCo

This script uses the built MuJoCo directly with your trained model,
following the Unitree approach but simplified for easier setup.

Usage:
    python run_g1_with_trained_model.py

Requirements:
    - MuJoCo built and available
    - Trained model in trained_models/model_final.pt
    - conda mujoco_env activated
"""

import time
import pathlib
import threading
import queue
import sys
import termios
import tty
import select

import numpy as np
import torch
import mujoco
import mujoco.viewer


class SimpleG1Controller:
    """Simplified controller for G1 with trained model"""
    
    def __init__(self):
        self.project_root = pathlib.Path(__file__).parent
        
        # Use the 29-DOF scene from unitree_mujoco
        self.xml_path = self.project_root / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"
        
        # Control parameters
        self.simulation_dt = 0.002  # 500Hz simulation
        self.control_dt = 0.02      # 50Hz control
        self.control_decimation = int(self.control_dt / self.simulation_dt)
        
        # Load trained model
        self.policy = self._load_trained_model()
        
        # State variables
        self.cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, ωz]
        self.action = np.zeros(29, dtype=np.float32)
        self.target_dof_pos = np.zeros(29, dtype=np.float32)
        self.obs = np.zeros(480, dtype=np.float32)
        self.counter = 0
        
        # Control thread
        self.control_queue = queue.Queue()
        self.running = False
        
        # Load MuJoCo model
        self._load_mujoco_model()
    
    def _load_trained_model(self):
        """Load the trained model"""
        model_path = self.project_root / "trained_models" / "model_25000.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✓ Loaded checkpoint from: {model_path}")
        print(f"✓ Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract the policy network
        if 'actor' in checkpoint:
            policy = checkpoint['actor']
            print(f"✓ Loaded actor from checkpoint")
        elif 'model_state_dict' in checkpoint:
            # If it's a full checkpoint, we need to reconstruct the model
            print(f"✓ Found model_state_dict in checkpoint")
            state_dict = checkpoint['model_state_dict']
            
            # Filter for actor parameters
            actor_state_dict = {}
            for key, value in state_dict.items():
                if 'actor' in key or 'policy' in key:
                    new_key = key.replace('actor.', '').replace('policy.', '')
                    actor_state_dict[new_key] = value
            
            print(f"✓ Found {len(actor_state_dict)} actor parameters")
            
            # Create a simple policy network (you may need to adjust this based on your actual architecture)
            policy = torch.nn.Sequential(
                torch.nn.Linear(480, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 29)
            )
            
            # Try to load the state dict
            try:
                policy.load_state_dict(actor_state_dict, strict=False)
                print(f"✓ Loaded actor state dict successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load actor state dict: {e}")
                print(f"✓ Using randomly initialized policy")
        else:
            # Check if it's a direct model
            if hasattr(checkpoint, 'eval'):
                policy = checkpoint
                print(f"✓ Loaded checkpoint as policy directly")
            else:
                print(f"⚠️  Warning: Unknown checkpoint format, using random policy")
                # Create a simple policy network as fallback
                policy = torch.nn.Sequential(
                    torch.nn.Linear(480, 512),
                    torch.nn.ELU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 29)
                )
        
        policy.eval()
        print(f"✓ Policy ready for inference")
        
        # Test the policy with a dummy input
        dummy_obs = torch.zeros(1, 480)
        with torch.no_grad():
            dummy_action = policy(dummy_obs)
            print(f"✓ Policy test - Input shape: {dummy_obs.shape}, Output shape: {dummy_action.shape}")
            print(f"✓ Policy test - Output range: [{dummy_action.min().item():.3f}, {dummy_action.max().item():.3f}]")
            
            if torch.isnan(dummy_action).any() or torch.isinf(dummy_action).any():
                print("⚠️  WARNING: Policy output contains NaN or Inf values!")
            else:
                print("✓ Policy output is valid (no NaN/Inf)")
        
        return policy
    
    def _load_mujoco_model(self):
        """Load MuJoCo model"""
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Scene file not found: {self.xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt
        
        print(f"✓ Loaded MuJoCo model: {self.xml_path}")
        print(f"  - DOF: {self.model.nv}")
        print(f"  - Actuators: {self.model.nu}")
        print(f"  - Bodies: {self.model.nbody}")
        print(f"  - Joints: {self.model.njnt}")
    
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
        
        # Scale observations (simplified scaling)
        omega_scaled = omega * 0.2
        gravity_scaled = gravity_orientation * 1.0
        cmd_scaled = self.cmd_vel * 1.0
        
        # Joint positions (relative to zero)
        qj_rel = qj * 1.0
        
        # Joint velocities
        dqj_scaled = dqj * 0.05
        
        # Last action
        action_scaled = self.action * 1.0
        
        # Construct observation vector (480 dimensions for 29-DOF with history)
        obs = np.zeros(480, dtype=np.float32)
        
        # Fill observation (simplified structure)
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
        
        # Check for NaN or Inf values in observation
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️  WARNING: Observation contains NaN or Inf values!")
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
            
            # Convert action to target joint positions (simplified)
            action_scale = 0.25
            self.target_dof_pos = self.action * action_scale
    
    def apply_control(self):
        """Apply PD control to joints"""
        # Get current joint positions and velocities
        current_q = self.data.qpos[7:]
        current_dq = self.data.qvel[6:]
        
        # Simple PD control with reasonable gains
        kps = np.array([100.0] * 12 + [200.0] * 3 + [40.0] * 14, dtype=np.float32)  # 29 DOF
        kds = np.array([2.0] * 12 + [5.0] * 3 + [10.0] * 14, dtype=np.float32)  # 29 DOF
        
        tau = (self.target_dof_pos - current_q) * kps + (0.0 - current_dq) * kds
        
        # Check for NaN or Inf values in control
        if np.isnan(tau).any() or np.isinf(tau).any():
            print(f"⚠️  WARNING: Control torques contain NaN or Inf values!")
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
        self.data.qpos[7:] = 0.0
        self.data.qvel[6:] = 0.0
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.action = np.zeros(29)
        self.target_dof_pos = np.zeros(29)
        print("Robot reset to standing pose")
    
    def print_help(self):
        """Print control commands"""
        print("\n=== G1-29dof Trained Model Control Commands ===")
        print("w/s: Forward/Backward")
        print("a/d: Strafe Left/Right")
        print("q/e: Turn Left/Right")
        print("space: Stop")
        print("r: Reset to standing")
        print("h: Help")
        print("x: Quit")
        print("================================================\n")


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
        controller = SimpleG1Controller()
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
        print("- external/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
