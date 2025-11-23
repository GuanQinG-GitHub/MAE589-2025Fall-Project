import numpy as np
import os
import torch
from src.env.g1_env import G1MujocoEnv
from src.policy.policy_ct import QuadraticPolicy
import matplotlib.pyplot as plt

def train_lspi():
    print("Starting LSPI Training with G1MujocoEnv")
    
    # Initialize Environment
    env = G1MujocoEnv(terrain_name="ramp", terrain_degree=15)
    
    # Initialize Policy
    # State dim: 4 [MOS_ML, MOS_Forward, speed, jerk]
    # Action dim: 2 [kp_pitch, kp_roll]
    state_dim = 4
    action_dim = 2
    
    policy = QuadraticPolicy(state_dim, action_dim, explore=0.2)
    
    # Set scaling for action
    # We want stiffness in range [15, 60] approx.
    # Policy outputs in [-1, 1] usually? Or raw?
    # QuadraticPolicy usually outputs raw action if not scaled.
    # Let's check QuadraticPolicy implementation if possible, but assuming standard:
    # We'll set offset and scale to map [-1, 1] -> [15, 60]
    # Center = 37.5, Scale = 22.5
    policy.offset_a = np.array([0, 0])
    policy.scale_a = np.array([5, 5])
    
    num_episodes = 10
    max_steps_per_ep = 1000 # Env has its own max_steps but we can limit here too
    
    # Experience buffer
    experience = {'current_state': [], 'action': [], 'stage_cost': [], 'next_state': []}
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            # action = policy.select_action(state)
            action = np.array([0, 0])
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            # Cost = -Reward (LSPI minimizes cost)
            cost = -reward
            
            experience['current_state'].append(state)
            experience['action'].append(action)
            experience['stage_cost'].append(cost)
            experience['next_state'].append(next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update policy periodically (e.g., every 100 steps or end of episode)
            if len(experience['current_state']) >= 200:
                print("Updating policy...")
                policy.cvxw_update(experience, 0, len(experience['current_state']), verbose=False)
                # Clear buffer or keep? LSPI is off-policy, can keep.
                # But for online, maybe clear to avoid stale data dominance?
                # Let's keep it for now, or use a sliding window.
                # For simplicity, we clear after update to emulate "batch" updates
                experience = {'current_state': [], 'action': [], 'stage_cost': [], 'next_state': []}
        
        print(f"Episode {ep+1} finished. Total Reward: {total_reward:.2f}, Steps: {steps}")
        
        # Visualize logs for this episode
        env.render_logs(save_path=f"logs/train_ep_{ep+1}.png")

    print("Training finished.")
    
    # Save policy weights
    os.makedirs("weights", exist_ok=True)
    np.save("weights/lspi_g1_weights.npy", policy.weights)
    print("Policy weights saved.")

if __name__ == "__main__":
    train_lspi()
