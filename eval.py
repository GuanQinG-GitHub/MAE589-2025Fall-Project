

from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import numpy as np

def calculate_mos(model, data):
    """
    Calculates the Margin of Stability (MoS) for a Unitree H1 in MuJoCo.

    Args:
        model: MjModel object.
        data: MjData object with up-to-date simulation state.

    Returns:
        The Margin of Stability (MoS) as a float.
        Returns -inf if no ground contacts are detected.
    """
    # Step 1 & 2: Calculate the Center of Mass (CoM) and its velocity
    com_pos = data.subtree_com[0]
    com_vel = data.subtree_linvel[1] # velocity

    # Step 3: Calculate the Extrapolated Center of Mass (XCoM)
    g = abs(model.opt.gravity[2])
    z_com = com_pos[2]
    if z_com < 1e-6:
        # Handle cases where CoM is at or below ground level
        return -np.inf
    
    omega0 = np.sqrt(g / z_com)
    xcom = com_pos + com_vel / omega0

    # Step 4: Determine the Base of Support (BoS)
    contact_points = []
    # Loop through MuJoCo contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Define a threshold to filter out weak or transient contacts
        force_threshold = 0.5
        # MuJoCo stores forces in a compressed format; get the force vector
        contact_force_vector = data.efc_force[contact.efc_address : contact.efc_address + contact.dim]
        contact_force_magnitude = np.linalg.norm(contact_force_vector)

        if contact_force_magnitude > force_threshold:
            # Assuming ground is the second geom in the contact pair
            if model.geom(contact.geom2).name == "perlin_terrain_1" or model.geom(contact.geom1).name == "perlin_terrain_1" or \
                model.geom(contact.geom2).name == "floor" or model.geom(contact.geom1).name == "floor":
                contact_points.append(contact.pos)
    
    if len(contact_points) < 3:
        # If there are fewer than 3 stable contact points, the BoS is undefined
        return -np.inf

    # Project contact points onto the ground plane
    ground_contacts = np.array([p[:2] for p in contact_points])
    
    # Compute the convex hull of the ground contacts
    hull = ConvexHull(ground_contacts)
    support_polygon = ground_contacts[hull.vertices]

    # Step 5: Calculate the Margin of Stability (MoS)
    xcom_point = Point(xcom[0], xcom[1])
    support_polygon_shape = Polygon(support_polygon)
    
    # Check if XCoM is inside the polygon
    if support_polygon_shape.contains(xcom_point):
        # Find the distance to the closest edge
        nearest_geom = nearest_points(xcom_point, support_polygon_shape.boundary)[1]
        mos = xcom_point.distance(nearest_geom)
    else:
        # If outside, the margin is negative
        nearest_geom = nearest_points(xcom_point, support_polygon_shape.boundary)[1]
        mos = -xcom_point.distance(nearest_geom)
    
    return mos

def calculate_com_jerk(data, prev_com_acc, dt):
    """
    Calculates the jerk of the Center of Mass (CoM) for a Unitree H1 in MuJoCo.

    Args:
        model: MjModel object.
        data: MjData object with up-to-date simulation state.
        prev_com_acc: Previous CoM acceleration as a numpy array.
        dt: Time step duration.

    Returns:
        The CoM jerk as a float.
    """
    # Calculate current CoM position and velocity
    com_pos = data.subtree_com[0]
    com_vel = data.subtree_linvel[1]  # velocity

    # Calculate current CoM acceleration
    com_acc = data.subtree_acc[1]

    # Calculate jerk as the change in acceleration over time
    com_jerk = (com_acc - prev_com_acc) / dt

    # Return the magnitude of the jerk vector
    return np.linalg.norm(com_jerk), com_acc

