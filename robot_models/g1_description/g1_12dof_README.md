# G1 12-DOF Model Documentation

## Overview

The `g1_12dof` model is a simplified version of the Unitree G1 humanoid robot that includes **only the leg actuators** (12 degrees of freedom total), while the upper body (torso, arms, head) is present visually but not actuated.

## Model Specifications

### Actuated Joints (12 DOF)

The model has 12 controllable actuators, all located in the legs:

**Left Leg (6 DOF):**
- `left_hip_pitch_joint` - Hip pitch rotation
- `left_hip_roll_joint` - Hip roll rotation  
- `left_hip_yaw_joint` - Hip yaw rotation
- `left_knee_joint` - Knee flexion/extension
- `left_ankle_pitch_joint` - Ankle pitch rotation
- `left_ankle_roll_joint` - Ankle roll rotation

**Right Leg (6 DOF):**
- `right_hip_pitch_joint` - Hip pitch rotation
- `right_hip_roll_joint` - Hip roll rotation
- `right_hip_yaw_joint` - Hip yaw rotation
- `right_knee_joint` - Knee flexion/extension
- `right_ankle_pitch_joint` - Ankle pitch rotation
- `right_ankle_roll_joint` - Ankle roll rotation

### Visual Components (Non-Actuated)

The model includes visual meshes for the complete robot body:
- Pelvis and torso
- Head
- Left and right arms (shoulder, elbow, wrist)
- All leg components

However, **only the leg joints are actuated**. The upper body components are present for visual realism but do not have motors/actuators.

## Usage

### In Reinforcement Learning

The `g1_12dof` model is specifically designed for **legged locomotion tasks** where only leg control is needed. This makes it:

- **Simpler**: Fewer action dimensions (12 vs 23+ for full-body models)
- **Faster**: Reduced computational complexity
- **Focused**: Optimized for walking, running, and balancing tasks

### Configuration

When using this model with the `deploy_mujoco.py` script:

- **XML Path**: `resources/robots/g1_description/scene.xml` (which includes `g1_12dof.xml`)
- **Number of Actions**: 12
- **Number of Observations**: 47
- **Control Frequency**: 50 Hz (with `control_decimation: 10` and `simulation_dt: 0.002`)

### Observation Space (47 dimensions)

The observation vector contains the following components, in order:

| Index Range | Component | Dimension | Description | Scaling |
|-------------|-----------|-----------|-------------|---------|
| `obs[0:3]` | Base angular velocity | 3 | Base orientation angular velocity (roll, pitch, yaw rates) in base frame | `ang_vel_scale = 0.25` |
| `obs[3:6]` | Gravity orientation | 3 | Gravity vector in base frame, derived from base quaternion orientation | None (unit vector) |
| `obs[6:9]` | Velocity command | 3 | Desired velocity commands: `[forward_vel, lateral_vel, angular_vel]` | `cmd_scale = [2.0, 2.0, 0.25]` |
| `obs[9:21]` | Joint positions | 12 | Normalized joint positions relative to default angles | `(qj - default_angles) * dof_pos_scale` where `dof_pos_scale = 1.0` |
| `obs[21:33]` | Joint velocities | 12 | Joint angular velocities | `dqj * dof_vel_scale` where `dof_vel_scale = 0.05` |
| `obs[33:45]` | Previous action | 12 | Action from previous control timestep (for temporal consistency) | None |
| `obs[45:47]` | Phase signal | 2 | Periodic signal `[sin(2π·phase), cos(2π·phase)]` encoding gait cycle position (0.0-1.0) | Period = 0.8 seconds, cycles continuously |

**Detailed Breakdown:**

1. **Base Angular Velocity (obs[0:3])**: 
   - The 3D angular velocity of the robot's base (pelvis) in the base frame
   - Scaled by `ang_vel_scale = 0.25` to normalize the values
   - Extracted from `d.qvel[3:6]` (MuJoCo data structure)

2. **Gravity Orientation (obs[3:6])**:
   - A 3D unit vector representing the direction of gravity in the base frame
   - Computed from the base quaternion using the `get_gravity_orientation()` function
   - Helps the policy understand the robot's orientation relative to gravity
   - Formula: `[2*(-qz*qx + qw*qy), -2*(qz*qy + qw*qx), 1 - 2*(qw*qw + qz*qz)]`

3. **Velocity Command (obs[6:9])**:
   - Desired motion commands: forward velocity, lateral velocity, and angular velocity
   - Scaled by `cmd_scale = [2.0, 2.0, 0.25]` to normalize different velocity ranges
   - Default initial command: `[0.5, 0, 0]` m/s (forward walking)

4. **Joint Positions (obs[9:21])**:
   - 12 normalized joint angles (6 per leg)
   - Order: `[left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll, right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll]`
   - Normalized by subtracting default angles and scaling by `dof_pos_scale = 1.0`
   - Extracted from `d.qpos[7:19]` (skipping base position/orientation)

5. **Joint Velocities (obs[21:33])**:
   - 12 joint angular velocities in the same order as joint positions
   - Scaled by `dof_vel_scale = 0.05` to normalize the values
   - Extracted from `d.qvel[6:18]` (skipping base linear/angular velocity)

6. **Previous Action (obs[33:45])**:
   - The action output from the policy at the previous control timestep
   - Provides temporal information to help the policy generate smooth, consistent motions
   - Same 12-dimensional structure as the current action space

7. **Phase Signal (obs[45:47])**:
   - A periodic signal `[sin(2π·phase), cos(2π·phase)]` that encodes the current phase within a gait cycle
   - **Purpose**: Provides temporal rhythm information to help the policy coordinate leg movements in a rhythmic walking pattern
   - **Period**: 0.8 seconds (one complete gait cycle)
   - **Phase calculation**: `phase = (elapsed_time % 0.8) / 0.8`, which cycles from 0.0 to 1.0
   - **Encoding**: Uses sin/cos pair instead of raw phase value to avoid discontinuity at phase wrap-around (0.0 ↔ 1.0)
   
   **Intuitive Explanation:**
   
   Think of walking as a rhythmic cycle, like a metronome. The phase signal tells the policy "where we are" in the current walking cycle. Just like a clock hand that completes a full rotation, the phase cycles from 0.0 to 1.0 every 0.8 seconds.
   
   **Example - Phase Progression Through a Gait Cycle:**
   
   | Time (s) | Phase | sin(2π·phase) | cos(2π·phase) | Gait Stage (Intuitive) |
   |---------|-------|---------------|---------------|------------------------|
   | 0.0     | 0.0   | 0.0           | 1.0           | Start of cycle - Left foot lift |
   | 0.1     | 0.125 | 0.707         | 0.707         | Left leg swinging forward |
   | 0.2     | 0.25  | 1.0           | 0.0           | Mid-swing - Left foot highest |
   | 0.3     | 0.375 | 0.707         | -0.707        | Left leg descending |
   | 0.4     | 0.5   | 0.0           | -1.0          | Left foot contact - Right foot lift |
   | 0.5     | 0.625 | -0.707        | -0.707        | Right leg swinging forward |
   | 0.6     | 0.75  | -1.0          | 0.0           | Mid-swing - Right foot highest |
   | 0.7     | 0.875 | -0.707        | 0.707         | Right leg descending |
   | 0.8     | 0.0   | 0.0           | 1.0           | Cycle complete - Back to start |
   
   **Why Sin/Cos Encoding?**
   
   - If we used raw phase (0.0 to 1.0), there would be a discontinuity: phase 0.999 is very different from phase 0.001, even though they're adjacent in the cycle
   - Sin/cos encoding creates a smooth circular representation: `(sin, cos)` forms a unit circle
   - The policy can learn smooth transitions: as phase increases, the (sin, cos) point smoothly moves around the circle
   - This helps the neural network learn continuous, rhythmic patterns without sudden jumps
   
   **Visual Representation:**
   ```
   Phase 0.0:   (0.0,  1.0)  → Top of circle
   Phase 0.25:  (1.0,  0.0)  → Right of circle
   Phase 0.5:   (0.0, -1.0)  → Bottom of circle
   Phase 0.75:  (-1.0, 0.0)  → Left of circle
   Phase 1.0:   (0.0,  1.0)  → Back to top (smooth wrap-around)
   ```
   
   The policy uses this phase information to coordinate which leg should be lifting, swinging, or contacting the ground at any given moment, creating natural, rhythmic walking patterns.

**Observation Processing Example** (from `g1_pretrained_testing.py`):
```python
# Extract raw state
qj = d.qpos[7:19]  # Joint positions (skip base pose)
dqj = d.qvel[6:18]  # Joint velocities (skip base)
quat = d.qpos[3:7]  # Base orientation quaternion
omega = d.qvel[3:6]  # Base angular velocity

# Process observations
qj = (qj - default_angles) * dof_pos_scale  # Normalize joint positions
dqj = dqj * dof_vel_scale  # Scale joint velocities
gravity_orientation = get_gravity_orientation(quat)  # Compute gravity vector
omega = omega * ang_vel_scale  # Scale angular velocity

# Build phase signal
period = 0.8
phase = (counter * simulation_dt) % period / period
sin_phase = np.sin(2 * np.pi * phase)
cos_phase = np.cos(2 * np.pi * phase)

# Construct observation vector
obs[:3] = omega
obs[3:6] = gravity_orientation
obs[6:9] = cmd * cmd_scale
obs[9:21] = qj
obs[21:33] = dqj
obs[33:45] = action  # Previous action
obs[45:47] = [sin_phase, cos_phase]
```

### Action Space (12 dimensions)

The action space consists of 12 normalized joint position targets, one for each actuated joint:

| Index | Joint Name | Description | Action Range | Transformation |
|-------|------------|-------------|--------------|----------------|
| 0 | `left_hip_pitch_joint` | Left hip pitch target | [-1, 1] | `target = action * action_scale + default_angle` |
| 1 | `left_hip_roll_joint` | Left hip roll target | [-1, 1] | where `action_scale = 0.25` |
| 2 | `left_hip_yaw_joint` | Left hip yaw target | [-1, 1] | |
| 3 | `left_knee_joint` | Left knee target | [-1, 1] | |
| 4 | `left_ankle_pitch_joint` | Left ankle pitch target | [-1, 1] | |
| 5 | `left_ankle_roll_joint` | Left ankle roll target | [-1, 1] | |
| 6 | `right_hip_pitch_joint` | Right hip pitch target | [-1, 1] | |
| 7 | `right_hip_roll_joint` | Right hip roll target | [-1, 1] | |
| 8 | `right_hip_yaw_joint` | Right hip yaw target | [-1, 1] | |
| 9 | `right_knee_joint` | Right knee target | [-1, 1] | |
| 10 | `right_ankle_pitch_joint` | Right ankle pitch target | [-1, 1] | |
| 11 | `right_ankle_roll_joint` | Right ankle roll target | [-1, 1] | |

**Action Processing:**
- Actions are output from the policy as normalized values in the range [-1, 1]
- These are transformed to target joint positions using: `target_dof_pos = action * action_scale + default_angles`
- The `action_scale = 0.25` limits the maximum deviation from default angles to ±0.25 radians
- Target positions are then tracked using PD control with the gains specified in the PD Control Parameters section

**Example** (from `g1_pretrained_testing.py`):
```python
# Policy outputs normalized action [-1, 1] for each joint
action = policy(obs_tensor).detach().numpy().squeeze()  # Shape: (12,)

# Transform to target joint positions
target_dof_pos = action * action_scale + default_angles
# Example: if action[0] = 0.5, then target_dof_pos[0] = 0.5 * 0.25 + (-0.1) = 0.025 rad
```

### Default Joint Angles

The default standing pose configuration:
```python
default_angles = [-0.1,  0.0,  0.0,  0.3, -0.2, 0.0,  # Left leg
                  -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]  # Right leg
```

Corresponding to:
- Hip pitch: -0.1 rad
- Hip roll: 0.0 rad
- Hip yaw: 0.0 rad
- Knee: 0.3 rad
- Ankle pitch: -0.2 rad
- Ankle roll: 0.0 rad

### PD Control Parameters

The model uses position control with the following PD gains:

**Stiffness (kp):**
- Hip joints: 100 N⋅m/rad
- Knee: 150 N⋅m/rad
- Ankle: 40 N⋅m/rad

**Damping (kd):**
- Hip joints: 2 N⋅m⋅s/rad
- Knee: 4 N⋅m⋅s/rad
- Ankle: 2 N⋅m⋅s/rad

## Comparison with Other G1 Models

| Model | Total DOF | Leg DOF | Waist DOF | Arm DOF | Use Case |
|-------|-----------|---------|-----------|---------|----------|
| `g1_12dof` | **12** | **6×2** | **0** | **0** | **Legged locomotion only** |
| `g1_23dof_rev_1_0` | 23 | 6×2 | 1 | 5×2 | Full body (no hands) |
| `g1_29dof_rev_1_0` | 29 | 6×2 | 3 | 7×2 | Full body with waist |
| `g1_29dof_with_hand_rev_1_0` | 43 | 6×2 | 3 | 7×2 + 7×2 | Full body with hands |

## Files

- **MJCF Model**: `g1_12dof.xml`
- **URDF Model**: `g1_12dof.urdf`
- **Scene File**: `scene.xml` (includes the robot model with environment)

## Notes

- The model uses a **floating base** (6 DOF for base position/orientation), but only the 12 leg joints are actuated
- The upper body is **passive** - it will move due to **dynamics** but cannot be directly controlled
- This model is optimized for **reinforcement learning** training and deployment of locomotion policies
- The pre-trained policy (`motion.pt`) is specifically trained for this 12-DOF configuration

