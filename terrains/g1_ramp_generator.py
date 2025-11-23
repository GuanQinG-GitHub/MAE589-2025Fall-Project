"""
G1 ramp Terrain Generator

This script creates pure ramp noise-based terrain specifically for G1 humanoid robot testing.
It generates multiple terrain areas with different noise characteristics.

Usage:
    cd external/unitree_mujoco/terrain_tool
    python g1_ramp_generator.py
"""

# Import required libraries for XML manipulation, image processing, and noise generation
import xml.etree.ElementTree as xml_et  # For creating and modifying XML elements
import xml.etree.ElementTree as ET      # Alternative import for XML parsing
import numpy as np                      # For numerical operations and array handling
import cv2                             # OpenCV for image processing and saving height maps

import os                              # For file system operations (creating directories)

# Configuration constants for the G1 robot terrain generation
ROBOT = "g1"                                    # Robot type identifier
INPUT_SCENE_PATH = "./terrains/scene.xml"       # Path to the base MuJoCo scene template
# Note: No longer generating scene_ramp_terrain.xml as it's not used by the scripts

def generate_ramp_heightmap(width, height, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate a heightmap using a linear ramp.
    
    Parameters:
        width (int): Width of the heightmap in pixels
        height (int): Height of the heightmap in pixels
        scale (float): Unused for ramp
        octaves (int): Unused for ramp
        persistence (float): Unused for ramp
        lacunarity (float): Unused for ramp
        seed (int): Unused for ramp
    
    Returns:
        np.ndarray: 2D heightmap array with values 0-255 (grayscale image)
    """
    # Initialize empty heightmap array with 8-bit unsigned integers (0-255 range)
    heightmap = np.zeros((height, width), dtype=np.uint8)
    
    # Create a linear ramp along the x-axis
    # Value increases from 0 to 255 as x goes from 0 to width
    for x in range(width):
        value = int((x / width) * 255)
        heightmap[:, x] = value
            
    return heightmap

def create_ramp_terrain_1(asset, worldbody, output_dir):
    """
    Create ramp terrain 1: Main area (moderate complexity).
    
    Parameters:
        asset: XML asset element to add height field and material
        worldbody: XML worldbody element to add terrain geometry
        output_dir: Directory to save the height map image
    """
    print("  - Generating ramp terrain 1 (main area)...")
    
    # Generate height map for terrain 1
    terrain_1 = generate_ramp_heightmap(
        width=128 * 2, height=128 * 2,     # 128x128 pixel resolution
        scale=40.0,                # Moderate detail level
        octaves=6,                 # 6 noise layers for moderate complexity
        persistence=0.5,           # 50% amplitude reduction per octave
        lacunarity=2.0,            # Double frequency each octave
        seed=42                    # Fixed seed for reproducible terrain
    )
    cv2.imwrite(output_dir + "g1_ramp_terrain_1.png", terrain_1)  # Save as PNG image
    
    # Create height field asset
    hfield1 = xml_et.SubElement(asset, "hfield")
    hfield1.set("name", "ramp_hfield_1")
    hfield1.set("size", "3.0 2.0 0.1 0.05")       # Physical size: length, width, max_height, min_height
    hfield1.set("file", "g1_ramp_terrain_1.png")
    
    # Create material
    material1 = xml_et.SubElement(asset, "material")
    material1.set("name", "ramp_terrain_1_mat")
    material1.set("rgba", "0.3 0.5 0.3 1")       # Greenish color
    material1.set("roughness", "0.6")
    
    # Create terrain geometry
    terrain1 = xml_et.SubElement(worldbody, "geom")
    terrain1.set("name", "ramp_terrain_1")
    terrain1.set("type", "hfield")
    terrain1.set("hfield", "ramp_hfield_1")
    terrain1.set("pos", "2.0 0.0 0.0")          # Position: right of robot
    terrain1.set("quat", "1 0 0 0")
    terrain1.set("material", "ramp_terrain_1_mat")

def create_g1_ramp_terrain():
    """
    Create terrain with pure ramp noise (model-independent).
    
    This function:
    1. Loads the base MuJoCo scene template (no robot model included)
    2. Sets up the scene with terrain-specific settings
    3. Calls individual terrain creation functions
    4. Saves the complete terrain scene to file
    
    To customize which terrains are created, comment out the corresponding function calls below.
    """
    print("Generating ramp Terrain (Model-Independent)...")
    
    # Load the base MuJoCo scene XML file (generic terrain template)
    scene = ET.parse(INPUT_SCENE_PATH)
    root = scene.getroot()                    # Get the root element of the XML tree
    worldbody = root.find("worldbody")        # Find the worldbody element (contains all physical objects)
    asset = root.find("asset")                # Find the asset element (contains textures, materials, etc.)
    
    # Update the scene XML model name to reflect ramp terrain
    root.set("model", "ramp terrain scene")
    
    # Note: No robot model is included in this template - it's purely for terrain generation
    # Robot models should be included by the specific scene files that use this terrain
    
    # Define output directory for generated files (relative to root directory)
    output_dir = "./terrains/"  # Save files in the terrains directory
    os.makedirs(output_dir, exist_ok=True)    # Create directory if it doesn't exist (exist_ok prevents error if exists)
    
    print("  - Creating XML scene...")
    
    # =============================================================================
    # TERRAIN CREATION - Comment out any terrain you don't want to include
    # =============================================================================
    
    # Terrain 1: Main area (moderate complexity) - positioned right of robot
    create_ramp_terrain_1(asset, worldbody, output_dir)
    
    
    # =============================================================================
    # END TERRAIN CREATION
    # =============================================================================
    
    # Note: No longer saving scene_ramp_terrain.xml as it's not used by the scripts
    # The terrain images are saved directly and used by g1_pretrained_ramp_terrain.py
    
    # Print summary information about the generated terrain
    print("ramp terrain images generated successfully!")
    print("Generated files:")
    print("  - g1_ramp_terrain_1.png: Main area (3x2m, moderate complexity)")
    print("  - Pure height field-based terrain with natural variations")
    print("  - Model-independent terrain (no robot model included)")
    print("\nTo customize terrain characteristics, edit the terrain creation section in create_g1_ramp_terrain().")
    print("To use this terrain, run: python g1_pretrained_ramp_terrain.py")

# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    create_g1_ramp_terrain()  # Call the main terrain generation function
