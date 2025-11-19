# PD Controller Implementation and Control Pipeline

## Question 1: PD Controller Implementation

### Your Understanding is Correct! ✅

Yes, the pipeline works exactly as you described:
1. **Policy outputs** desired joint angles for all 12 joints
2. **PD controller** drives each joint to the desired position
3. Each joint has **separate PD parameters** (kp and kd)

### PD Controller Implementation Location

The PD controller is implemented in **two places**:

#### 1. In `g1_pretrained_testing.py` (Lines 43-45):

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands using PD control."""
    return (target_q - q) * kp + (target_dq - dq) * kd
```

#### 2. In `deploy_mujoco.py` (Lines 26-28):

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd
```

### How It's Used in the Control Loop

In `g1_pretrained_testing.py` (Lines 187-196):

```python
# Apply PD control
tau = pd_control(
    target_dof_pos,                                    # Desired joint positions (from policy)
    d.qpos[7:7+config["num_actions"]],                 # Current joint positions
    kps,                                               # Proportional gains (array of 12 values)
    np.zeros_like(kds),                                # Target velocities (zero for position control)
    d.qvel[6:6+config["num_actions"]],                 # Current joint velocities
    kds                                                # Derivative gains (array of 12 values)
)
d.ctrl[:] = tau  # Apply torques to MuJoCo actuators
```

### PD Controller Details

**Formula:**
```
τ = kp × (q_target - q_current) + kd × (dq_target - dq_current)
```

Where:
- `τ` = Torque applied to joint
- `kp` = Proportional gain (stiffness)
- `kd` = Derivative gain (damping)
- `q_target` = Desired joint position (from policy)
- `q_current` = Current joint position (from sensors)
- `dq_target` = Desired joint velocity (typically 0 for position control)
- `dq_current` = Current joint velocity (from sensors)

**Key Points:**
- Each of the 12 joints has its own `kp` and `kd` values
- The PD controller runs at the simulation timestep (500 Hz with `dt=0.002`)
- The policy runs at a lower frequency (50 Hz with `control_decimation=10`)
- Between policy updates, the PD controller continuously tracks the last desired position

### PD Parameters for G1 12-DOF

From `g1.yaml` and `g1_pretrained_testing.py`:

```python
kps = [100, 100, 100, 150, 40, 40,    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
       100, 100, 100, 150, 40, 40]    # Right leg: same order

kds = [2, 2, 2, 4, 2, 2,              # Left leg
       2, 2, 2, 4, 2, 2]              # Right leg
```

**Pattern:**
- Hip joints: `kp=100`, `kd=2`
- Knee joints: `kp=150`, `kd=4` (higher stiffness for knee)
- Ankle joints: `kp=40`, `kd=2` (lower stiffness for ankle compliance)

### Training Implementation

In the training code (`external/unitree_rl_gym/legged_gym/envs/base/legged_robot.py`, Lines 308-330):

```python
def _compute_torques(self, actions):
    """Compute torques from actions using PD control."""
    actions_scaled = actions * self.cfg.control.action_scale
    control_type = self.cfg.control.control_type
    
    if control_type == "P":  # Position control (used for G1)
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) \
                 - self.d_gains * self.dof_vel
    # ... other control types ...
    
    return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

This is equivalent to the deployment version, just using PyTorch tensors for batch processing.

---

## Question 2: Dynamic Control Components and Ground Reaction Forces

### Your Observation is Correct! ✅

**Ground reaction forces (GRF) are NOT explicitly computed or used in the control loop**, but they are **implicitly handled** through the physics simulation and learned by the policy.

### How Ground Reaction Forces Work in This Pipeline

#### 1. **Implicit Handling via Physics Simulator**

The ground reaction forces are **automatically computed by MuJoCo** based on:
- Contact between feet and ground
- Joint torques applied by PD controllers
- Robot dynamics (mass, inertia, etc.)

**Flow:**
```
Policy → Desired Joint Angles → PD Controller → Joint Torques → MuJoCo Physics → 
Contact Forces (GRF) → Robot Motion → New Joint States → Policy (next step)
```

The policy **doesn't explicitly command** ground reaction forces. Instead:
- The policy learns to generate joint angles that result in appropriate contact forces
- MuJoCo's physics engine automatically computes the contact forces
- The policy learns this relationship through reinforcement learning

#### 2. **Ground Reaction Forces During Training**

During training, contact forces ARE used (but not in the control loop):

**A. In Rewards** (`external/unitree_rl_gym/legged_gym/envs/g1/g1_env.py`):

```python
def _reward_contact(self):
    """Reward for proper foot contact timing."""
    res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    for i in range(self.feet_num):
        is_stance = self.leg_phase[:, i] < 0.55  # Expected stance phase
        contact = self.contact_forces[:, self.feet_indices[i], 2] > 1  # Vertical force > 1N
        res += ~(contact ^ is_stance)  # Reward when contact matches expected phase
    return res

def _reward_feet_swing_height(self):
    """Reward for lifting feet during swing phase."""
    contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
    return torch.sum(pos_error, dim=(1))

def _reward_contact_no_vel(self):
    """Penalize contact when foot has velocity (sliding)."""
    contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
    return torch.sum(penalize, dim=(1,2))
```

**B. Contact Force Monitoring** (`legged_robot.py`, Line 442):

```python
self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
    self.num_envs, -1, 3
)  # Shape: [num_envs, num_bodies, xyz]
```

**C. In Configuration** (`legged_robot_config.py`, Line 125):

```python
max_contact_force = 100.  # Forces above this value are penalized
```

#### 3. **Why This Works: End-to-End Learning**

The policy is trained **end-to-end** to:
- Generate joint angles that create appropriate ground reaction forces
- Coordinate leg movements to maintain balance
- Adapt to different terrains and disturbances

The policy learns this implicitly because:
- **Rewards** encourage proper contact timing (feet down during stance, up during swing)
- **Rewards** penalize excessive contact forces
- **Rewards** penalize sliding (contact with foot velocity)
- The policy sees the **consequences** of its actions through the physics simulation

#### 4. **Comparison with Explicit Whole-Body Control**

| Aspect | This Approach (RL + PD) | Explicit Whole-Body Control |
|--------|------------------------|----------------------------|
| **GRF Planning** | Learned implicitly | Explicitly computed (QP, MPC) |
| **Control Loop** | Policy → Joint Angles → PD → Torques | Desired GRF → Inverse Dynamics → Torques |
| **Adaptability** | Learns from experience | Requires model and optimization |
| **Complexity** | Simple deployment | Complex real-time optimization |
| **Terrain Adaptation** | Learned during training | Requires terrain estimation |

### What About "Whole-Body Controller"?

You mentioned "whole-body controller" - in this context:

- **The policy IS the whole-body controller** - it coordinates all 12 joints together
- **The PD controllers are low-level joint servos** - they execute the policy's commands
- **Together they form a hierarchical control system:**
  ```
  High-level: Policy (neural network) - decides what to do
       ↓
  Low-level: PD controllers (12 independent) - execute joint commands
       ↓
  Physics: MuJoCo - computes dynamics and contact forces
  ```

### Summary

**Question 1 Answer:**
- ✅ PD controller is at lines 43-45 in `g1_pretrained_testing.py`
- ✅ Each joint has separate kp/kd parameters
- ✅ Formula: `τ = kp × (q_target - q_current) + kd × (dq_target - dq_current)`
- ✅ Applied at every simulation step (500 Hz)

**Question 2 Answer:**
- ✅ Ground reaction forces are **NOT explicitly computed** in the control loop
- ✅ They are **implicitly handled** by MuJoCo's physics engine
- ✅ The policy **learns** to generate joint angles that result in appropriate GRF
- ✅ During training, contact forces are used in **rewards** to guide learning
- ✅ This is an **end-to-end learning approach** rather than explicit model-based control

The beauty of this approach is that the policy learns the complex relationship between joint angles, torques, and ground reaction forces through trial and error, without needing explicit models or optimization solvers!

