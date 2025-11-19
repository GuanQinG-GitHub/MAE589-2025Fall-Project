# Unitree RL Lab Modifications: G1-12DOF Configuration

This document describes the modifications made to `unitree_rl_lab` to add support for the G1-12DOF robot model (legs only) for residual RL training.

## Overview

**Purpose**: Add a 12 DOF (degrees of freedom) configuration for the Unitree G1 robot, focusing on leg joints only. This is used for residual RL training where we train a policy to adapt ankle impedance parameters.

**Robot Model**: G1-12DOF
- **Total DOF**: 12 (legs only)
- **Left Leg**: 6 DOF (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
- **Right Leg**: 6 DOF (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)

## File Modified

**File Path**: `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

**Location**: After `UNITREE_G1_29DOF_CFG` (around line 509) and before `UNITREE_G1_23DOF_CFG` (around line 598)

## Change Summary

- **Lines Added**: 86 lines
- **Type**: New configuration class `UNITREE_G1_12DOF_CFG`
- **Commit**: `98dbec5` - "Add G1-12dof robot configuration for residual RL training"

## Step-by-Step Instructions

### Step 1: Locate the Insertion Point

1. Open the file: `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
2. Find the `UNITREE_G1_29DOF_CFG` configuration (ends around line 509)
3. Look for the closing parenthesis and comma: `),` after the `joint_sdk_names` list
4. The new code should be inserted **right after** this closing parenthesis, before the next configuration

### Step 2: Find the Exact Location

Look for this pattern in the file:

```python
    ],
)
```

This should be the end of `UNITREE_G1_29DOF_CFG`. The next line should be a blank line, followed by a docstring for `UNITREE_G1_23DOF_CFG`.

### Step 3: Insert the New Configuration

Insert the following code block **after** the closing `)` of `UNITREE_G1_29DOF_CFG` and **before** the `UNITREE_G1_23DOF_CFG` configuration:

```python
"""Configuration for the Unitree G1 12DOF Humanoid robot (legs only)."""

UNITREE_G1_12DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Note: USD path assumes 12 DOF USD file exists at this location
        # If not available, may need to use 23 DOF USD and configure only leg joints
        usd_path=f"{UNITREE_MODEL_DIR}/G1/12dof/usd/g1_12dof_rev_1_0/g1_12dof_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*"],  # 4 joints
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
            },
            damping={
                ".*_hip_.*": 2.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],  # 4 joints
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],  # 4 joints
            effort_limit_sim=35,
            velocity_limit_sim=30,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    # fmt: on
)
```

### Step 4: Verify the Change

After inserting the code, verify:

1. The file still has valid Python syntax (no indentation errors)
2. The new `UNITREE_G1_12DOF_CFG` is defined
3. It appears between `UNITREE_G1_29DOF_CFG` and `UNITREE_G1_23DOF_CFG`
4. All parentheses and brackets are properly closed

## Configuration Details

### Actuator Configuration

The G1-12DOF uses three actuator types:

1. **N7520-14.3** (4 joints):
   - `left_hip_pitch_joint`, `right_hip_pitch_joint`
   - `left_hip_yaw_joint`, `right_hip_yaw_joint`
   - Stiffness: 100.0 N⋅m/rad
   - Damping: 2.0 N⋅m⋅s/rad

2. **N7520-22.5** (4 joints):
   - `left_hip_roll_joint`, `right_hip_roll_joint`
   - `left_knee_joint`, `right_knee_joint`
   - Stiffness: 100.0 (hip_roll), 150.0 (knee) N⋅m/rad
   - Damping: 2.0 (hip_roll), 4.0 (knee) N⋅m⋅s/rad

3. **N5020-16-parallel** (4 joints):
   - `left_ankle_pitch_joint`, `left_ankle_roll_joint`
   - `right_ankle_pitch_joint`, `right_ankle_roll_joint`
   - Stiffness: 40.0 N⋅m/rad (base value, can be modified by residual policy)
   - Damping: 2.0 N⋅m⋅s/rad

### Joint SDK Names

The 12 joint names (in order):
1. `left_hip_pitch_joint`
2. `left_hip_roll_joint`
3. `left_hip_yaw_joint`
4. `left_knee_joint`
5. `left_ankle_pitch_joint`
6. `left_ankle_roll_joint`
7. `right_hip_pitch_joint`
8. `right_hip_roll_joint`
9. `right_hip_yaw_joint`
10. `right_knee_joint`
11. `right_ankle_pitch_joint`
12. `right_ankle_roll_joint`

### Initial State

- **Base Position**: (0.0, 0.0, 0.8) meters (standing pose)
- **Hip Pitch**: -0.1 radians
- **Knee**: 0.3 radians
- **Ankle Pitch**: -0.2 radians
- **All Joint Velocities**: 0.0 rad/s

## Usage

After adding this configuration, you can import and use it in your environment configurations:

```python
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_12DOF_CFG as ROBOT_CFG
```

This is used in the residual RL training environment (`residual_rl_training/tasks/g1_residual_env_cfg.py`).

## Important Notes

1. **USD File Path**: The configuration assumes a 12 DOF USD file exists at:
   ```
   {UNITREE_MODEL_DIR}/G1/12dof/usd/g1_12dof_rev_1_0/g1_12dof_rev_1_0.usd
   ```
   If this file doesn't exist, you may need to:
   - Use the 23 DOF USD file and configure only leg joints
   - Or create/obtain the 12 DOF USD file

2. **Ankle Stiffness**: The ankle stiffness (40.0 N⋅m/rad) is the base value. In residual RL training, this value is dynamically modified by the residual policy (typically ±25% variation).

3. **Compatibility**: This configuration is compatible with Isaac Lab's `ManagerBasedRLEnv` framework.

## Verification

To verify the modification was applied correctly:

1. Check that `UNITREE_G1_12DOF_CFG` can be imported:
   ```python
   from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_12DOF_CFG
   ```

2. Verify the configuration has 12 joints:
   ```python
   assert len(UNITREE_G1_12DOF_CFG.joint_sdk_names) == 12
   ```

3. Check that the residual RL environment can use it:
   ```python
   from residual_rl_training.tasks.g1_residual_env_cfg import RobotEnvCfg
   # Should use UNITREE_G1_12DOF_CFG
   ```

## Troubleshooting

### Import Error
If you get an import error, make sure:
- The code was inserted in the correct location
- Python syntax is valid (check for missing commas, parentheses)
- The file was saved

### USD File Not Found
If the USD file path doesn't exist:
- Check if `UNITREE_MODEL_DIR` is set correctly
- Verify the USD file exists at the specified path
- Consider using the 23 DOF USD file as a fallback

### Configuration Not Working
If the configuration doesn't work:
- Verify all actuator names match the joint names
- Check that joint_sdk_names are in the correct order
- Ensure actuator limits are reasonable

## Related Files

This modification is used by:
- `residual_rl_training/tasks/g1_residual_env_cfg.py` - Environment configuration
- `residual_rl_training/tasks/g1_residual_env.py` - Custom environment class

## Git Information

- **Commit Hash**: `98dbec5`
- **Commit Message**: "Add G1-12dof robot configuration for residual RL training"
- **Date**: (Check your local git log for exact date)
- **Files Changed**: 1 file, 86 insertions

## Summary

This modification adds a 12 DOF configuration for the G1 robot, focusing on leg joints only. It's a straightforward addition of a new configuration class that follows the same pattern as the existing G1 configurations (29DOF, 23DOF). The configuration is specifically designed for residual RL training where ankle impedance parameters are dynamically modified.

