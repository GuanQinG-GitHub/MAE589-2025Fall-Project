"""
Date: Oct 6, 2025

Controlled Unitree G1 Humanoid with Pre-trained RL Model

This script integrates the pre-trained Unitree G1 model from unitree_rl_gym
with MuJoCo for real-time motion control via command-line interface.

Features:
- Load pre-trained LSTM policy from unitree_rl_gym
- Real-time command control (forward, backward, turn, strafe)
- Interactive terminal interface
- Proper observation construction matching training environment
- PD control with trained gains

Usage:
    python Mujoco_TutorialScript_8_g1_controlled.py

Commands:
    w/s: forward/backward
    a/d: strafe left/right  
    q/e: turn left/right
    space: stop
    r: reset to standing
    h: help
    q: quit
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


class G1Controller:
    """Controller for Unitree G1 humanoid using pre-trained RL model"""
    
    def __init__(self):
        self.project_root = pathlib.Path(__file__).parent
        self.rl_gym_root = self.project_root / "external" / "unitree_rl_gym"
        
        # Model paths
        self.policy_path = self.rl_gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
        self.xml_path = self.rl_gym_root / "resources" / "robots" / "g1_description" / "scene.xml"
        
        # Control parameters (from g1.yaml)
        self.simulation_dt = 0.002
        self.control_decimation = 10
        self.action_scale = 0.25
        self.cmd_scale = np.array([2.0, 2.0, 0.25])
        
        # PD gains
        self.kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
        self.kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
        
        # Default joint angles
        self.default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                       -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
        
        # Observation scaling
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        
        # State variables
        self.cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, ωz]
        self.action = np.zeros(12, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(47, dtype=np.float32)
        self.counter = 0
        self.phase = 0.0
        
        # Control thread
        self.control_queue = queue.Queue()
        self.running = False
        
        # Load model and policy
        self._load_model()
        self._load_policy()
    
    def _load_model(self):
        """Load MuJoCo model"""
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {self.xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt
        
        print(f"✓ Loaded MuJoCo model: {self.xml_path}")
    
    def _load_policy(self):
        """Load pre-trained policy"""
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Pre-trained model not found: {self.policy_path}")
        
        self.policy = torch.jit.load(str(self.policy_path))
        self.policy.eval()
        print(f"✓ Loaded pre-trained policy: {self.policy_path}")
    
    def get_gravity_orientation(self, quaternion):
        """Convert quaternion to gravity orientation vector"""
        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        return gravity_orientation
    
    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """PD controller for joint position control"""
        return (target_q - q) * kp + (target_dq - dq) * kd
    
    def construct_observation(self):
        """Construct observation vector for the pre-trained model"""
        # Get current state
        qj = self.data.qpos[7:]  # Joint positions (skip base position and quaternion)
        dqj = self.data.qvel[6:]  # Joint velocities (skip base linear and angular velocity)
        quat = self.data.qpos[3:7]  # Base quaternion
        omega = self.data.qvel[3:6]  # Base angular velocity
        
        # Scale joint positions relative to default
        qj_scaled = (qj - self.default_angles) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        
        # Get gravity orientation
        gravity_orientation = self.get_gravity_orientation(quat)
        omega_scaled = omega * self.ang_vel_scale
        
        # Calculate phase for gait timing
        period = 0.8
        count = self.counter * self.simulation_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # Construct observation vector (47 dimensions)
        obs = np.zeros(47, dtype=np.float32)
        obs[:3] = omega_scaled  # Angular velocity
        obs[3:6] = gravity_orientation  # Gravity orientation
        obs[6:9] = self.cmd_vel * self.cmd_scale  # Commands
        obs[9:21] = qj_scaled  # Joint positions
        obs[21:33] = dqj_scaled  # Joint velocities
        obs[33:45] = self.action  # Previous actions
        obs[45:47] = [sin_phase, cos_phase]  # Phase information
        
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
            self.target_dof_pos = self.action * self.action_scale + self.default_angles
    
    def apply_control(self):
        """Apply PD control to joints"""
        # Compute torques using PD control
        tau = self.pd_control(
            self.target_dof_pos, 
            self.data.qpos[7:], 
            self.kps, 
            np.zeros_like(self.kds), 
            self.data.qvel[6:], 
            self.kds
        )
        
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
        self.data.qpos[7:] = self.default_angles
        self.data.qvel[6:] = 0.0
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.action = np.zeros(12)
        self.target_dof_pos = self.default_angles.copy()
        print("Robot reset to standing pose")
    
    def print_help(self):
        """Print control commands"""
        print("\n=== G1 Humanoid Control Commands ===")
        print("w/s: Forward/Backward")
        print("a/d: Strafe Left/Right")
        print("q/e: Turn Left/Right")
        print("space: Stop")
        print("r: Reset to standing")
        print("h: Help")
        print("x: Quit")
        print("=====================================\n")


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
    print("=== Unitree G1 Humanoid with Pre-trained RL Model ===")
    print("Loading model and policy...")
    
    try:
        # Initialize controller
        controller = G1Controller()
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
        print("Please ensure the unitree_rl_gym repository is cloned in external/")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
