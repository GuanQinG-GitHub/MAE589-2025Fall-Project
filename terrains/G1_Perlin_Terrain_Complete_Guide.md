# G1 Perlin Terrain Complete Guide

This comprehensive guide covers everything you need to know about testing and customizing Perlin noise-based terrain for the G1 humanoid robot in MuJoCo simulation.

## Overview

The Perlin terrain system provides:
1. **Pure Height Field Terrain**: Natural-looking terrain using Perlin noise
2. **Single Terrain Area**: Focused testing on one Perlin terrain with moderate complexity
3. **Realistic Natural Variations**: Organic terrain patterns for challenging robot navigation
4. **Easy Customization**: Simple parameter adjustment for different terrain characteristics
5. **Real-time Performance Monitoring**: Tracks falls, distance, height, and success zones
6. **Model-Independent Design**: Terrain generation is separate from robot models

## Files Structure

```
├── terrains/
│   ├── g1_perlin_generator.py              # Main terrain generator
│   ├── g1_perlin_terrain_1.png             # Height field image (main area)
│   ├── g1_perlin_terrain_2.png             # Height field image (side area) - optional
│   ├── g1_perlin_terrain_3.png             # Height field image (challenging area) - optional
│   ├── scene.xml                           # Base scene template (model-independent)
│   └── G1_Perlin_Terrain_Complete_Guide.md # This guide
├── robot_models/
│   └── g1_description/
│       ├── g1_12dof.xml                    # G1 robot model (12 DoF)
│       └── meshes/                          # Robot mesh files
│           ├── pelvis.STL
│           ├── head_link.STL
│           └── [other mesh files...]
├── trained_models/
│   └── motion.pt                           # Pre-trained policy
├── g1_pretrained_testing.py               # Flat ground testing script
└── g1_pretrained_perlin_terrain.py        # Perlin terrain testing script
```

## Quick Start

### 1. Install Dependencies
```bash
conda activate unitree_rl
pip install opensimplex opencv-python pyyaml
```

### 2. Generate Terrain (if needed)
```bash
cd terrains
python g1_perlin_generator.py
cd ..
```

### 3. Run the Test
```bash
# Test on flat ground
python g1_pretrained_testing.py

# Test on Perlin terrain
python g1_pretrained_perlin_terrain.py
```

## Terrain Types

### Available Terrain Types

1. **Flat Ground** (`g1_pretrained_testing.py`)
   - Simple flat plane for basic validation
   - Tests robot's basic walking capabilities
   - No height variations

2. **Perlin Terrain** (`g1_pretrained_perlin_terrain.py`)
   - Natural height variations using Perlin noise
   - Single terrain area with moderate complexity
   - Position: (4.0, 0.0, -0.07) - in front of robot
   - Size: 3x2m with 0.2m max height
   - Resolution: 128x128 pixels
   - Scale: 40.0 (moderate detail)
   - Octaves: 6 (moderate complexity)

### Other Terrain Types (Available for Extension)

The system can be extended to include:

3. **Rough Terrain**: Multiple height variations with sharp edges
4. **Stairs**: Regular step patterns for structured challenges
5. **Suspended Stairs**: Elevated platforms requiring jumping
6. **Mixed Terrain**: Combination of different terrain types
7. **Custom Height Fields**: User-defined terrain from images

## Manual Terrain Customization

### 1. Understanding HField (Height Field) Parameters

#### What is an HField?
An HField in MuJoCo is a **height field** - a 2D grid where each pixel represents a height value. It's like a "height map" that defines the terrain elevation at each point.

#### HField Parameters Breakdown:
```xml
<hfield name="perlin_hfield_1" size="3.0 2.0 0.08 0.05" file="../../../terrains/g1_perlin_terrain_1.png" />
```

The `size` parameter has **4 components**:
- **`3.0`** = **Length** (X-axis) in meters
- **`2.0`** = **Width** (Y-axis) in meters  
- **`0.08`** = **Maximum height** in meters (highest point)
- **`0.05`** = **Minimum height** in meters (lowest point)

#### Why Only the PNG File is Sufficient:
The PNG file contains **all the terrain data** because:
1. **Pixel Values = Height Data**: Each pixel's grayscale value (0-255) represents the height at that location
2. **Automatic Scaling**: MuJoCo automatically maps the 0-255 pixel values to the height range (0.05 to 0.08 meters)
3. **Grid Resolution**: The PNG dimensions (128x128) define the terrain resolution

#### Height Mapping Logic:
```
PNG Pixel Value → Height Mapping
0 (black)       → 0.05m (minimum height)
128 (gray)      → 0.065m (middle height)  
255 (white)     → 0.08m (maximum height)
```

### 2. Terrain Modification Workflow

#### **Two-Step Modification Process:**

**Step 1: Regenerate Random Pattern (if needed)**
```bash
python terrains/g1_perlin_generator.py
```
- This regenerates the PNG height map with new random Perlin noise
- Only run this if you want a completely different terrain pattern
- The PNG file contains the actual height data

**Step 2: Adjust Height Scale (if needed)**
- Edit the `hfield` definition in `g1_pretrained_perlin_terrain.py`:
```xml
<hfield name="perlin_hfield_1" size="3.0 2.0 0.08 0.05" file="../../../terrains/g1_perlin_terrain_1.png" />
```
- Change the height values (3rd and 4th parameters) to adjust terrain height
- No need to regenerate the PNG file

#### **Common Height Scale Examples:**
```xml
<!-- Gentle terrain -->
<hfield size="3.0 2.0 0.05 0.02" file="../../../terrains/g1_perlin_terrain_1.png" />

<!-- Steep terrain -->
<hfield size="3.0 2.0 0.2 0.0" file="../../../terrains/g1_perlin_terrain_1.png" />

<!-- Very flat terrain -->
<hfield size="3.0 2.0 0.02 0.01" file="../../../terrains/g1_perlin_terrain_1.png" />
```

### 3. Understanding Perlin Noise Parameters

The terrain generator uses these key parameters:

#### Basic Parameters
- **`width`, `height`**: Height map resolution (e.g., 128x128, 64x64)
- **`scale`**: Noise frequency (lower = smoother, higher = more detailed)
- **`octaves`**: Number of noise layers (more = more complex)
- **`persistence`**: Amplitude reduction per octave (0.0-1.0)
- **`lacunarity`**: Frequency increase per octave (usually 2.0)
- **`seed`**: Random seed for reproducible terrain

#### Terrain Dimensions
- **`size`**: Physical terrain size in meters [length, width]
- **`height_scale`**: Maximum terrain height in meters
- **`negative_height`**: Depth below ground level

### 2. Modifying Terrain Characteristics

#### Method 1: Edit the Generator Script
Open `terrains/g1_perlin_generator.py` and modify the terrain generation calls:

```python
# Example: Modifying Perlin Terrain 1
terrain_1 = generate_perlin_heightmap(
    width=128, height=128,     # Increase for higher resolution
    scale=40.0,                # Lower = smoother, higher = more detailed
    octaves=6,                 # More octaves = more complex terrain
    persistence=0.5,           # 0.0-1.0, higher = more variation
    lacunarity=2.0,            # Usually 2.0, higher = denser detail
    seed=42                    # Change for different terrain
)
```

#### Key Parameters to Adjust:
- **`scale`**: Controls terrain smoothness (40.0 = moderate, 20.0 = very detailed, 80.0 = very smooth)
- **`octaves`**: Controls complexity (6 = moderate, 10 = very complex, 3 = simple)
- **`persistence`**: Controls height variation (0.3 = uniform, 0.7 = very varied)
- **`lacunarity`**: Controls detail density (1.8 = gradual, 2.5 = dense)

#### Method 2: Modify Terrain Position in Script
**Important**: You can directly modify the terrain position in `g1_pretrained_perlin_terrain.py`:

```python
# In the combined_xml string, find this line:
<geom name="perlin_terrain_1" type="hfield" hfield="perlin_hfield_1" pos="4.0 0.0 -0.07" quat="1 0 0 0" material="perlin_terrain_1_mat" />

# Modify the pos attribute to change terrain position:
# pos="X Y Z" where:
# X = forward/backward position (positive = in front of robot)
# Y = left/right position (positive = to robot's left)
# Z = up/down position (negative = below ground level)
```

**Common Position Examples:**
- `pos="2.0 0.0 0.0"` - Close to robot, at ground level
- `pos="5.0 0.0 -0.1"` - Far ahead, slightly below ground
- `pos="0.0 3.0 0.0"` - To the side of robot
- `pos="-2.0 0.0 0.0"` - Behind robot

#### Method 3: Modify Terrain Size and Properties
In the same script, you can also adjust:
```python
# Terrain size: size="3.0 2.0 0.2 0.1" = [length, width, max_height, min_height]
<hfield name="perlin_hfield_1" size="3.0 2.0 0.2 0.1" file="../../../terrains/g1_perlin_terrain_1.png" />

# Material properties: rgba="0.3 0.5 0.3 1" = [red, green, blue, alpha]
<material name="perlin_terrain_1_mat" rgba="0.3 0.5 0.3 1" roughness="0.6" />
```

### 3. Complete Modification Workflow

#### **Two-Step Modification Process:**

**Step 1: Regenerate Random Pattern (if needed)**
```bash
python terrains/g1_perlin_generator.py
```
- This regenerates the PNG height map with new random Perlin noise
- Only run this if you want a completely different terrain pattern
- The PNG file contains the actual height data

**Step 2: Adjust Height Scale (if needed)**
- Edit the `hfield` definition in `g1_pretrained_perlin_terrain.py`:
```xml
<hfield name="perlin_hfield_1" size="3.0 2.0 0.08 0.05" file="../../../terrains/g1_perlin_terrain_1.png" />
```
- Change the height values (3rd and 4th parameters) to adjust terrain height
- No need to regenerate the PNG file

#### **Common Height Scale Examples:**
```xml
<!-- Gentle terrain -->
<hfield size="3.0 2.0 0.05 0.02" file="../../../terrains/g1_perlin_terrain_1.png" />

<!-- Steep terrain -->
<hfield size="3.0 2.0 0.2 0.0" file="../../../terrains/g1_perlin_terrain_1.png" />

<!-- Very flat terrain -->
<hfield size="3.0 2.0 0.02 0.01" file="../../../terrains/g1_perlin_terrain_1.png" />
```

#### **Quick Modification Examples:**

**For Pattern Changes:**
```bash
# 1. Regenerate terrain pattern
python terrains/g1_perlin_generator.py

# 2. Test the new terrain
python g1_pretrained_perlin_terrain.py
```

**For Height/Position Changes:**
```bash
# 1. Edit g1_pretrained_perlin_terrain.py (modify hfield size or geom pos)
# 2. Test immediately (no regeneration needed)
python g1_pretrained_perlin_terrain.py
```

## Performance Monitoring

The testing script tracks:
- **Distance traveled**: How far the robot moves
- **Maximum height**: Highest point reached
- **Fall count**: Number of times robot falls
- **Success zones**: Which areas the robot reaches
- **Final position**: Where the robot ends up

### Expected Performance
The Perlin terrain is designed to be challenging:
- **Expected Performance**: Fair to Good (robot may struggle initially)
- **Falls**: 0-10 falls are normal for moderate terrain
- **Purpose**: Test robot's ability to handle natural, uneven terrain
- **Success Zone**: Robot should reach the Perlin terrain area (position 5,0)

## Troubleshooting

### Common Issues:

1. **`ModuleNotFoundError`**: 
   ```bash
   conda activate unitree_rl
   pip install opensimplex opencv-python pyyaml
   ```

2. **XML Errors (`Error opening file '.../meshes/...'`)**:
   - Ensure the `g1_perlin_terrain_*.png` files are in the `terrains/` directory
   - Verify that the `robot_models/g1_description/meshes/` directory contains all required STL files
   - The scripts use `<compiler meshdir="robot_models/g1_description/meshes/"/>` to locate mesh files

3. **Robot falls immediately/Poor performance**:
   - Reduce `scale` or increase `smoothness` in `g1_perlin_generator.py`
   - Adjust terrain position in `g1_pretrained_perlin_terrain.py` to start on flatter section
   - The pre-trained policy might not be robust enough for highly uneven terrain

4. **File path errors**:
   - Ensure you're running scripts from the project root directory
   - Check that all files are in the correct `terrains/` folder
   - Verify that `trained_models/motion.pt` exists

### Tips:
- Start with smoother Perlin terrain and gradually increase complexity
- Use the MuJoCo viewer to inspect terrain before testing
- Monitor the performance metrics to track progress
- The terrain uses the `g1_12dof.xml` model (12 actuators) which is compatible with the pre-trained policy
- **Quick terrain position adjustment**: Modify the `pos` attribute in `g1_pretrained_perlin_terrain.py` for immediate testing

## Requirements

- MuJoCo simulation environment
- G1 robot model and pre-trained policy
- Python environment with required dependencies
- Windows PowerShell (for running commands)

## Advanced Customization

### Creating New Terrain Types

You can create additional terrain areas by:
1. Adding new `generate_perlin_heightmap()` calls in `g1_perlin_generator.py`
2. Adding corresponding `hfield` and `material` elements in the script's combined XML
3. Adding new `geom` elements for the terrain placement in `g1_pretrained_perlin_terrain.py`

### Quick Terrain Position Testing

For rapid testing of different terrain positions:
1. Open `g1_pretrained_perlin_terrain.py`
2. Find the `combined_xml` string
3. Modify the `pos` attribute in the terrain geom
4. Run the script immediately (no regeneration needed)

### Batch Terrain Generation

For testing multiple terrain configurations:
1. Create multiple generator scripts with different parameters
2. Use different output file names
3. Create corresponding test scripts for each terrain type
4. Or modify the existing script to cycle through different positions

## Conclusion

This Perlin terrain testing system provides a robust and realistic environment for evaluating the G1 robot's locomotion capabilities on natural, uneven ground. The system is designed to be easily customizable, with three main methods for terrain modification:

1. **Generator Script**: Modify `g1_perlin_generator.py` for terrain characteristics
2. **Position Adjustment**: Directly edit `g1_pretrained_perlin_terrain.py` for quick position changes
3. **Property Modification**: Adjust size, material, and other properties in the script

The model-independent design allows for easy extension to other robot models and terrain types, making it ideal for comprehensive robot testing and development.
