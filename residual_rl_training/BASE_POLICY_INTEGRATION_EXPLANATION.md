# Base Policy Integration - Explanation

## Key Clarifications

### 1. **No C++ Code Needed for Training! ✅**

**Important**: The C++ code you saw (`unitree_rl_lab/deploy/`) is **only for real robot deployment**, not for training.

- **Training**: Pure Python using Isaac Lab framework
- **Deployment**: C++ code for real robot control
- **`train.py`**: 100% Python, no C++ involved

### 2. **How Isaac Lab Works (Python)**

Isaac Lab's `ManagerBasedRLEnv` is a **Python class** that:
1. Uses managers (ActionManager, ObservationManager) to process actions/observations
2. Applies actions to the robot through Isaac Sim's Python API
3. PD parameters are set in the robot's actuator configuration

### 3. **How PD Parameters Work**

In Isaac Lab, PD parameters (stiffness/damping) are set in the robot configuration:

```python
# In unitree.py - actuator configuration
actuators={
    "N5020-16-parallel": ImplicitActuatorCfg(
        joint_names_expr=[".*ankle.*"],  # 4 ankle joints
        stiffness=40.0,  # Base kp value
        damping=2.0,     # Base kd value
    ),
}
```

**Key Point**: These are **config values**, but we can **compute torques manually** with modified PD parameters!

## The Solution: Custom Torque Computation

### What We Need to Do

1. **Load base policy** (frozen, no gradients)
2. **Get base policy actions** (12 DOF for all leg joints)
3. **Get residual actions** (4 DOF for ankle impedance modifications)
4. **Modify ankle PD parameters** based on residual actions
5. **Compute torques manually** using modified PD parameters
6. **Apply torques** directly to the robot

### Why This Works

Instead of relying on Isaac Lab's built-in PD control (which uses fixed PD parameters), we:
- **Bypass** the default torque computation
- **Compute torques ourselves** with dynamic PD parameters
- **Apply torques directly** to the robot

This is exactly what your current `g1_residual_env.py` does (lines 201-296)!

## Implementation Approach

### **Recommended: Custom Environment Class (Extend ManagerBasedRLEnv)**

**Why this is best**:
- ✅ Pure Python (no C++ needed)
- ✅ Full control over torque computation
- ✅ Can modify PD parameters dynamically
- ✅ Matches your current `g1_residual_env.py` structure

**How it works**:

```python
class ResidualRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        # Load base policy
        self.base_policy = torch.jit.load(cfg.base_policy_path)
        self.base_policy.eval()
        for param in self.base_policy.parameters():
            param.requires_grad = False
        
        # Store ankle joint indices
        self.ankle_joint_indices = [4, 5, 10, 11]  # Ankle joints
    
    def _apply_action(self, action):
        """Override action application to use base policy + residual."""
        # 1. Get residual actions (4 DOF) from RL policy
        residual_actions = action  # Already processed by action manager
        
        # 2. Get base policy actions (12 DOF)
        base_obs = self._get_base_policy_observations()
        with torch.no_grad():
            base_actions = self.base_policy(base_obs)
        
        # 3. Modify ankle PD parameters based on residual actions
        residual_scale = 0.25
        base_kp = 40.0
        modified_kp = base_kp + residual_actions * residual_scale * base_kp
        modified_kp = torch.clamp(modified_kp, 20.0, 60.0)
        
        # 4. Compute torques with modified PD parameters
        torques = self._compute_torques_with_modified_pd(
            base_actions,  # 12 DOF actions
            modified_kp,   # Modified ankle kp
            self.ankle_joint_indices
        )
        
        # 5. Apply torques directly
        self.scene["robot"].set_joint_effort_target(torques)
    
    def _compute_torques_with_modified_pd(self, actions, modified_kp, ankle_indices):
        """Compute torques with modified ankle PD parameters."""
        robot = self.scene["robot"]
        default_kp = robot.data.default_joint_stiffness
        default_kd = robot.data.default_joint_damping
        
        # Create modified kp/kd arrays
        kp = default_kp.clone()
        kd = default_kd.clone()
        
        # Modify ankle joints
        for i, ankle_idx in enumerate(ankle_indices):
            kp[:, ankle_idx] = modified_kp[:, i]
            kd[:, ankle_idx] = modified_kp[:, i] * 0.05  # kd proportional to kp
        
        # Compute torques: τ = kp * (q_target - q_current) - kd * dq_current
        q_target = actions + robot.data.default_joint_pos
        q_current = robot.data.joint_pos
        dq_current = robot.data.joint_vel
        
        torques = kp * (q_target - q_current) - kd * dq_current
        return torch.clip(torques, -robot.data.joint_effort_limit, robot.data.joint_effort_limit)
```

## Summary

### What You Need:
1. ✅ **Base policy loading** - Load frozen policy in `__init__`
2. ✅ **Base policy inference** - Get 12 DOF actions from base policy
3. ✅ **Residual action processing** - Modify ankle PD parameters
4. ✅ **Custom torque computation** - Compute torques with modified PD
5. ✅ **Direct torque application** - Apply torques to robot

### What You DON'T Need:
- ❌ C++ code
- ❌ Modifying Isaac Lab's core
- ❌ Custom action terms (unless you want to)
- ❌ Complex wrappers

### The Simplest Path:

**Extend `ManagerBasedRLEnv`** and override the action application method to:
1. Get base policy actions
2. Compute torques with modified PD parameters
3. Apply torques directly

This is essentially what your current `g1_residual_env.py` does, but adapted to Isaac Lab's structure!

