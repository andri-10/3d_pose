"""
Geometry and Object Sampling for Laser Triangulation System
FIXED VERSION - Uses correct laser triangulation geometry
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ==============================================================
# OBJECT SURFACE SAMPLING
# ==============================================================

def sample_cube_surface(size=0.15, num_points=2000):
    """
    Sample points uniformly on cube surface.
    
    Args:
        size: Cube side length (meters)
        num_points: Number of points to sample
        
    Returns:
        points: (N, 3) array of points on cube surface
    """
    points = []
    s = size / 2.0  # Half-size
    
    # Sample each face uniformly
    points_per_face = num_points // 6
    
    for _ in range(points_per_face):
        # Random position on face
        u = np.random.uniform(-s, s)
        v = np.random.uniform(-s, s)
        
        # Randomly select face
        face = np.random.randint(0, 6)
        
        if face == 0:   # +X face (right)
            points.append([s, u, v])
        elif face == 1: # -X face (left)
            points.append([-s, u, v])
        elif face == 2: # +Y face (top)
            points.append([u, s, v])
        elif face == 3: # -Y face (bottom)
            points.append([u, -s, v])
        elif face == 4: # +Z face (front)
            points.append([u, v, s])
        else:           # -Z face (back)
            points.append([u, v, -s])
    
    return np.array(points)


def sample_cylinder_surface(radius=0.075, height=0.20, num_points=2000):
    """
    Sample points uniformly on cylinder surface.
    
    Args:
        radius: Cylinder radius (meters)
        height: Cylinder height (meters)
        num_points: Number of points to sample
        
    Returns:
        points: (N, 3) array of points on cylinder surface
    """
    points = []
    h = height / 2.0  # Half-height
    
    # Distribute points: 50% side, 25% top, 25% bottom
    n_side = num_points // 2
    n_cap = num_points // 4
    
    # Side surface
    for _ in range(n_side):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-h, h)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    # Top cap
    for _ in range(n_cap):
        r = np.sqrt(np.random.uniform(0, radius**2))
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, h])
    
    # Bottom cap
    for _ in range(n_cap):
        r = np.sqrt(np.random.uniform(0, radius**2))
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, -h])
    
    return np.array(points)


# ==============================================================
# POSE GENERATION AND TRANSFORMATION
# ==============================================================

def random_pose(position_range, rotation_range_deg):
    """
    Generate random 6D pose within specified ranges.
    
    Args:
        position_range: dict with 'x', 'y', 'z' ranges
        rotation_range_deg: dict with 'roll', 'pitch', 'yaw' ranges
        
    Returns:
        position: (3,) translation vector
        quaternion: (4,) quaternion [qw, qx, qy, qz]
    """
    # Random position
    position = np.array([
        np.random.uniform(*position_range['x']),
        np.random.uniform(*position_range['y']),
        np.random.uniform(*position_range['z'])
    ])
    
    # Random rotation (Euler angles → quaternion)
    roll = np.radians(np.random.uniform(*rotation_range_deg['roll']))
    pitch = np.radians(np.random.uniform(*rotation_range_deg['pitch']))
    yaw = np.radians(np.random.uniform(*rotation_range_deg['yaw']))
    
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # [qw, qx, qy, qz]
    
    return position, quaternion


def apply_pose(points, position, quaternion):
    """
    Apply pose transformation to points.
    
    Args:
        points: (N, 3) array of points
        position: (3,) translation
        quaternion: (4,) quaternion [qw, qx, qy, qz]
        
    Returns:
        transformed_points: (N, 3) array
    """
    # Convert quaternion to rotation matrix
    qw, qx, qy, qz = quaternion
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    R = rotation.as_matrix()
    
    # Apply rotation then translation
    transformed = (R @ points.T).T + position
    return transformed


# ==============================================================
# NOISE AND REALISM
# ==============================================================

def add_noise(points, sigma=0.002):
    """
    Add Gaussian noise to point cloud.
    
    Args:
        points: (N, 3) array
        sigma: Standard deviation (meters)
        
    Returns:
        noisy_points: (N, 3) array
    """
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise


def random_dropout(points, dropout_rate=0.15):
    """
    Randomly remove points from point cloud.
    
    Args:
        points: (N, 3) array
        dropout_rate: Fraction of points to remove
        
    Returns:
        reduced_points: (M, 3) array where M < N
    """
    keep_mask = np.random.rand(len(points)) > dropout_rate
    return points[keep_mask]


def add_outliers(points, outlier_rate=0.02, outlier_scale=0.05):
    """
    Add random outlier points.
    
    Args:
        points: (N, 3) array
        outlier_rate: Fraction of points to replace with outliers
        outlier_scale: Scale of random displacement (meters)
        
    Returns:
        points_with_outliers: (N, 3) array
    """
    n_outliers = int(len(points) * outlier_rate)
    outlier_indices = np.random.choice(len(points), n_outliers, replace=False)
    
    points_copy = points.copy()
    points_copy[outlier_indices] += np.random.uniform(-outlier_scale, outlier_scale, (n_outliers, 3))
    
    return points_copy


def simulate_occlusion(points, camera_position=np.array([0, 0, 0]), occlusion_threshold=0.1):
    """
    Remove points not visible from camera (simple occlusion model).
    
    Args:
        points: (N, 3) array
        camera_position: (3,) camera position
        occlusion_threshold: Random occlusion probability
        
    Returns:
        visible_points: (M, 3) array
    """
    # Simple model: remove points randomly behind object center
    center = points.mean(axis=0)
    to_camera = camera_position - points
    to_center = center - points
    
    # Dot product: negative means point facing away from camera
    visibility = np.sum(to_camera * to_center, axis=1)
    visible_mask = visibility > 0
    
    # Add random occlusion
    visible_mask &= (np.random.rand(len(points)) > occlusion_threshold)
    
    return points[visible_mask]


# ==============================================================
# POINT CLOUD NORMALIZATION
# ==============================================================

def normalize_point_cloud(points, target_count=2048):
    """
    Normalize point cloud to fixed number of points.
    Center at origin and scale to unit sphere.
    
    Args:
        points: (N, 3) array
        target_count: Desired number of points
        
    Returns:
        normalized_points: (target_count, 3) array
    """
    # Resample to target count
    n_current = len(points)
    
    if n_current > target_count:
        # Randomly subsample
        indices = np.random.choice(n_current, target_count, replace=False)
        points = points[indices]
    elif n_current < target_count:
        # Oversample by repeating random points
        indices = np.random.choice(n_current, target_count - n_current, replace=True)
        points = np.vstack([points, points[indices]])
    
    # Center at origin
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 0:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered
    
    return points_normalized


# ==============================================================
# HIGH-LEVEL OBJECT GENERATION
# ==============================================================

def generate_object(object_type, size_variation=True):
    """
    Generate object point cloud with optional size variation.
    
    Args:
        object_type: "cube", "cylinder", or "complex"
        size_variation: If True, randomize size within ±20%
        
    Returns:
        points: (N, 3) array of points on object surface
    """
    if object_type == "cube":
        size = 0.15  # 15cm nominal
        if size_variation:
            size *= np.random.uniform(0.8, 1.2)
        return sample_cube_surface(size=size, num_points=4000)
    
    elif object_type == "cylinder":
        radius = 0.075  # 7.5cm nominal
        height = 0.20   # 20cm nominal
        if size_variation:
            radius *= np.random.uniform(0.8, 1.2)
            height *= np.random.uniform(0.8, 1.2)
        return sample_cylinder_surface(radius=radius, height=height, num_points=4000)
    
    elif object_type == "complex":
        # Placeholder - load from CAD file
        raise NotImplementedError("Complex object loading not yet implemented. Use STL loader.")
    
    else:
        raise ValueError(f"Unknown object type: {object_type}")