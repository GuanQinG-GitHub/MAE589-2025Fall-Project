# Version Control Strategy

## Repository Structure

This repository is set up for cross-platform development with the following branch strategy:

### Branches

- **`master`**: Stable, production-ready code (merged from both platforms)
- **`macos-mujoco-tutorials`**: macOS-specific MuJoCo physics simulation tutorials
- **`windows-rl-training`**: Windows-specific RL training code and configurations
- **`development`**: General development and integration testing

### Current Branch: `development`

You are currently on the `development` branch, which is perfect for:
- Testing cross-platform compatibility
- Integration work between macOS and Windows components
- General development and experimentation

### Platform-Specific Branches

**`macos-mujoco-tutorials`** - For macOS development:
- MuJoCo physics simulation tutorials
- Interactive 3D visualizations
- Data collection and analysis scripts
- Video generation and plotting

**`windows-rl-training`** - For Windows RL training:
- Isaac Lab/Isaac Sim installation and configuration
- RL training experiments
- Windows-specific optimizations
- GPU-accelerated training

## Submodules

- **`unitree_rl_lab`**: Main Unitree RL Lab repository
- **`unitree_model`**: Unitree robot USD models (via Hugging Face)

## Platform-Specific Workflows

### Windows RL Training (Current)
```bash
git checkout windows-rl-training
# Work on RL training, Isaac Lab setup, etc.
git add .
git commit -m "Windows RL training updates"
git push origin windows-rl-training
```

### macOS MuJoCo Development
```bash
git checkout macos-mujoco-tutorials
# Work on MuJoCo tutorials, testing, etc.
git add .
git commit -m "macOS MuJoCo tutorial updates"
git push origin macos-mujoco-tutorials
```

### Merging Changes
```bash
# Merge development work to main
git checkout main
git merge development
git push origin main

# Merge platform-specific work
git checkout master
git merge windows-rl-training
git merge macos-mujoco-tutorials
```

## File Organization

- **Cross-platform files**: Core algorithms, models, documentation
- **Platform-specific files**: Environment configs, installation scripts
- **Shared resources**: Robot models, training data, checkpoints

## Best Practices

1. **Always work on feature branches** before merging to main
2. **Test on both platforms** before merging to main
3. **Keep platform-specific configs** in separate files
4. **Document platform differences** in README files
5. **Use descriptive commit messages** with platform context

## Current Status

- ✅ Repository initialized with proper branch structure
- ✅ Unitree RL Lab cloned and configured
- ✅ Robot models downloaded via submodule
- ✅ Windows RL training branch active
- 🔄 Ready for Isaac Lab/Isaac Sim installation
