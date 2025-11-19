# MuJoCo/Physics Dynamics in Training

## Important Clarification: Training Uses Isaac Gym (PhysX), Not MuJoCo

**Key Point**: The training process uses **Isaac Gym with PhysX physics engine**, not MuJoCo directly. However, the deployment uses MuJoCo. The physics principles are similar, but the APIs differ.

**Training Flow:**
- **Training**: Isaac Gym (PhysX) → Policy Training
- **Deployment**: MuJoCo → Policy Execution

This document explains how physics dynamics work during training (Isaac Gym/PhysX) and what data is extracted.

---

## 1. Physics Simulation Step-by-Step

### Training Loop (`legged_robot.py`, Lines 49-80)

```python
def step(self, actions):
    """Apply actions, simulate, call self.post_physics_step()"""
    
    # 1. Clip actions
    self.actions = torch.clip(actions, -clip_actions, clip_actions)
    
    # 2. Render (if not headless)
    self.render()
    
    # 3. Physics simulation loop (runs multiple times per policy step)
    for _ in range(self.cfg.control.decimation):  # decimation = 4 (typically)
        # 3a. Compute joint torques from actions using PD control
        self.torques = self._compute_torques(self.actions)
        
        # 3b. Apply torques to simulation
        self.gym.set_dof_actuation_force_tensor(self.sim, self.torques)
        
        # 3c. Step physics simulation (THIS IS WHERE DYNAMICS HAPPEN)
        self.gym.simulate(self.sim)
        
        # 3d. Refresh DOF (joint) state tensor
        self.gym.refresh_dof_state_tensor(self.sim)
    
    # 4. Post-physics processing
    self.post_physics_step()
    
    # 5. Return observations, rewards, etc.
    return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
```

### Key Points:
- **Physics steps**: `decimation = 4` means 4 physics steps per policy step
- **Timestep**: `dt = 0.005` seconds (200 Hz physics simulation)
- **Policy frequency**: 200 Hz / 4 = 50 Hz (matches deployment)

---

## 2. What Data is Extracted from Physics Simulation

### A. After Physics Step (`post_physics_step()`, Lines 82-100)

```python
def post_physics_step(self):
    # Refresh state tensors from physics simulation
    self.gym.refresh_actor_root_state_tensor(self.sim)      # Base pose/velocity
    self.gym.refresh_net_contact_force_tensor(self.sim)     # Contact forces
    
    # Extract base state
    self.base_pos[:] = self.root_states[:, 0:3]             # Base position [x, y, z]
    self.base_quat[:] = self.root_states[:, 3:7]            # Base orientation (quaternion)
    self.base_lin_vel[:] = ...                              # Base linear velocity
    self.base_ang_vel[:] = ...                              # Base angular velocity
    
    # Extract joint state (already refreshed in step loop)
    # self.dof_pos = joint positions
    # self.dof_vel = joint velocities
    
    # Extract contact forces
    # self.contact_forces = net contact forces on all bodies
```

### B. Complete Data Available from Physics Simulation

| Data Type | Tensor Name | Shape | Description | Source |
|-----------|-------------|-------|-------------|--------|
| **Base Position** | `self.base_pos` | `[num_envs, 3]` | Robot base (pelvis) position [x, y, z] | `root_states[:, 0:3]` |
| **Base Orientation** | `self.base_quat` | `[num_envs, 4]` | Base quaternion [w, x, y, z] | `root_states[:, 3:7]` |
| **Base Linear Velocity** | `self.base_lin_vel` | `[num_envs, 3]` | Base linear velocity in base frame | `root_states[:, 7:10]` |
| **Base Angular Velocity** | `self.base_ang_vel` | `[num_envs, 3]` | Base angular velocity in base frame | `root_states[:, 10:13]` |
| **Joint Positions** | `self.dof_pos` | `[num_envs, num_dof]` | All joint angles | `dof_state[:, :, 0]` |
| **Joint Velocities** | `self.dof_vel` | `[num_envs, num_dof]` | All joint angular velocities | `dof_state[:, :, 1]` |
| **Contact Forces** | `self.contact_forces` | `[num_envs, num_bodies, 3]` | Net contact force on each body [fx, fy, fz] | `net_contact_force_tensor` |
| **Foot Positions** | `self.feet_pos` | `[num_envs, num_feet, 3]` | Foot positions in world frame | `rigid_body_states` |
| **Foot Velocities** | `self.feet_vel` | `[num_envs, num_feet, 3]` | Foot velocities in world frame | `rigid_body_states` |

### C. Contact Forces Details

**Contact Force Tensor** (`legged_robot.py`, Line 442):
```python
self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
    self.num_envs, -1, 3
)  # Shape: [num_envs, num_bodies, 3]
```

**What it contains:**
- **Net contact force** on each rigid body in the robot
- **3D force vector** [fx, fy, fz] in world frame
- **Includes ground reaction forces** on feet
- **Zero when no contact** (no force applied)

**Usage in rewards** (`g1_env.py`, Lines 98-120):
```python
def _reward_contact(self):
    """Reward for proper foot contact timing."""
    for i in range(self.feet_num):
        is_stance = self.leg_phase[:, i] < 0.55  # Expected stance phase
        contact = self.contact_forces[:, self.feet_indices[i], 2] > 1  # Vertical force > 1N
        res += ~(contact ^ is_stance)  # Reward when contact matches expected phase

def _reward_feet_swing_height(self):
    """Reward for lifting feet during swing phase."""
    contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    # Penalize if foot too low when not in contact
    pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact

def _reward_contact_no_vel(self):
    """Penalize sliding (contact with foot velocity)."""
    contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
    contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
    penalize = torch.square(contact_feet_vel[:, :, :3])
```

---

## 3. Contact Model: Point Contact vs. Surface Contact

### Foot Contact Geometry

Looking at `g1_12dof.xml` (Lines 101-104), the feet use **multiple small spheres** for contact:

```xml
<body name="left_ankle_roll_link">
  <!-- ... -->
  <geom size="0.005" pos="-0.05 0.025 -0.03" rgba="0.2 0.2 0.2 1"/>  <!-- Sphere 1 -->
  <geom size="0.005" pos="-0.05 -0.025 -0.03" rgba="0.2 0.2 0.2 1"/> <!-- Sphere 2 -->
  <geom size="0.005" pos="0.12 0.03 -0.03" rgba="0.2 0.2 0.2 1"/>    <!-- Sphere 3 -->
  <geom size="0.005" pos="0.12 -0.03 -0.03" rgba="0.2 0.2 0.2 1"/>   <!-- Sphere 4 -->
</body>
```

**Contact Model:**
- **4 small spheres** per foot (radius = 0.005m = 5mm)
- **Distributed contact points** approximating a foot surface
- **Each sphere can contact independently**
- **Not a single point contact**, but **multiple point contacts**

### Why Multiple Contact Points?

1. **Stability**: Multiple contact points provide better stability than a single point
2. **Realistic**: Approximates the actual foot contact area
3. **Robustness**: Can handle partial contact (e.g., foot on edge)
4. **Torque generation**: Allows the robot to generate torques about contact points

### Contact Force Computation

**PhysX (Isaac Gym) automatically computes:**
- Contact detection between spheres and ground
- Normal forces (perpendicular to contact surface)
- Friction forces (tangential, based on friction coefficient)
- **Net force** on each rigid body (sum of all contact forces)

**Contact Parameters** (from `legged_robot_config.py`):
```python
class physx:
    contact_offset = 0.01      # [m] Distance for contact detection
    rest_offset = 0.0          # [m] Resting distance
    bounce_threshold_velocity = 0.5  # [m/s] Minimum velocity for bouncing
    max_depenetration_velocity = 1.0  # [m/s] Maximum separation velocity
    contact_collection = 2     # Collect contacts at all sub-steps
```

---

## 4. How Ground Reaction Forces Work

### A. Automatic Computation

**Ground reaction forces are NOT explicitly computed by the code** - they are **automatically computed by PhysX** based on:

1. **Contact geometry**: Spheres on feet vs. ground plane
2. **Applied torques**: From PD controllers on joints
3. **Robot dynamics**: Mass, inertia, joint constraints
4. **Physics laws**: Newton's laws, friction, etc.

### B. Physics Flow

```
Policy Action → Joint Torques (PD) → Physics Simulation (PhysX)
                                              ↓
                                    Contact Detection
                                              ↓
                                    Force Computation
                                    (Normal + Friction)
                                              ↓
                                    Ground Reaction Forces
                                    (automatically computed)
                                              ↓
                                    Robot Motion
                                    (acceleration, velocity, position)
                                              ↓
                                    New State → Observations
```

### C. Ground Reaction Force Characteristics

**What PhysX computes:**
- **Normal force**: Perpendicular to contact surface (prevents penetration)
- **Friction force**: Tangential to contact surface (prevents sliding)
- **Total GRF**: Vector sum of all contact forces on foot
- **Direction**: Points upward (opposes gravity) when foot is on ground

**In the code:**
```python
# Contact force on foot (vertical component)
contact_force_z = self.contact_forces[:, foot_index, 2]  # Z-component (vertical)

# Check if foot is in contact
is_contacting = contact_force_z > 1.0  # Threshold: 1 Newton

# Total contact force magnitude
total_force = torch.norm(self.contact_forces[:, foot_index, :3], dim=-1)
```

---

## 5. Physics Simulation Details

### A. Simulation Parameters

From `legged_robot_config.py`:
```python
class sim:
    dt = 0.005              # Physics timestep: 5ms (200 Hz)
    substeps = 1            # Sub-steps per timestep
    gravity = [0., 0., -9.81]  # Gravity: 9.81 m/s² downward

class physx:
    solver_type = 1         # TGS (Temporal Gauss-Seidel) solver
    num_position_iterations = 4  # Position constraint solver iterations
    num_velocity_iterations = 0  # Velocity constraint solver iterations
```

### B. What Happens in `gym.simulate(self.sim)`

1. **Constraint solving**: Resolves joint constraints, contact constraints
2. **Force integration**: Applies joint torques, contact forces, gravity
3. **State update**: Updates positions, velocities, orientations
4. **Contact detection**: Detects new contacts, removes broken contacts
5. **Contact force computation**: Computes normal and friction forces

### C. Batch Processing

**Key advantage of Isaac Gym:**
- **Parallel simulation**: Thousands of environments run simultaneously
- **GPU acceleration**: Physics computation on GPU
- **Efficient data access**: Tensors directly on GPU (no CPU-GPU transfer)

```python
# All environments processed in parallel
self.contact_forces.shape  # [4096, num_bodies, 3] for 4096 parallel environments
self.dof_pos.shape         # [4096, 12] for 4096 robots
```

---

## 6. Comparison: Training (Isaac Gym) vs. Deployment (MuJoCo)

| Aspect | Training (Isaac Gym/PhysX) | Deployment (MuJoCo) |
|--------|---------------------------|---------------------|
| **Physics Engine** | PhysX (NVIDIA) | MuJoCo (Google DeepMind) |
| **Contact Model** | Multiple spheres per foot | Multiple spheres per foot |
| **Contact Forces** | `net_contact_force_tensor` | `data.efc_force` (contact forces) |
| **State Access** | GPU tensors (batch) | CPU arrays (single robot) |
| **Frequency** | 200 Hz (dt=0.005s) | 500 Hz (dt=0.002s) typical |
| **Parallelization** | Thousands of envs | Single environment |
| **API** | `gym.simulate()`, `gym.refresh_*_tensor()` | `mujoco.mj_step()`, `data.qpos`, etc. |

### Similarities:
- Both compute contact forces automatically
- Both use similar contact models (multiple contact points)
- Both provide joint positions, velocities, contact forces
- Both use PD control for joint actuation

### Differences:
- **Isaac Gym**: Optimized for parallel training (batch processing)
- **MuJoCo**: Optimized for single-robot simulation (faster, more accurate)

---

## 7. Summary

### What Physics Simulation Provides:

1. **Joint States**: Positions and velocities of all joints
2. **Base State**: Position, orientation, linear/angular velocity of robot base
3. **Contact Forces**: Ground reaction forces on all bodies (especially feet)
4. **Foot States**: Positions and velocities of feet
5. **Robot Motion**: All resulting from physics laws and applied torques

### Key Points:

✅ **Ground reaction forces ARE included** - automatically computed by PhysX  
✅ **Contact model**: Multiple small spheres (4 per foot), not single point contact  
✅ **Forces are extracted** via `net_contact_force_tensor` after each physics step  
✅ **Used in rewards** to guide policy learning (contact timing, force magnitude)  
✅ **Not explicitly commanded** - policy learns to generate appropriate GRF through joint torques  

The physics simulation is the "ground truth" that the policy learns to interact with. The policy doesn't directly control ground reaction forces - it controls joint torques, and the physics engine computes the resulting contact forces and motion.

