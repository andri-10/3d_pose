import numpy as np
from transforms3d.euler import euler2mat

def generate_cube_points(size=1.0, num_points=2000):
    """Generate a random point cloud representing a cube."""
    pts = np.random.uniform(-size/2, size/2, (num_points, 3))
    return pts

def random_pose():
    """Generate a random rotation and translation."""
    angles = np.random.uniform(-np.pi, np.pi, 3)
    R = euler2mat(angles[0], angles[1], angles[2])
    t = np.random.uniform(-0.1, 0.1, 3)
    return R, t

def apply_pose(points, R, t):
    """Apply R and t to point cloud."""
    return (R @ points.T).T + t

def add_noise(pc, sigma=0.01):
    noise = np.random.normal(0, sigma, pc.shape)
    return pc + noise

def random_dropout(pc, drop_rate=0.2):
    mask = np.random.rand(pc.shape[0]) > drop_rate
    return pc[mask]

def apply_visibility_mask(pc, normal=np.array([0, 0, 1]), threshold=0.0):
    dots = pc @ normal
    mask = dots > threshold
    return pc[mask]

def laser_slice(pc, plane_normal, plane_offset, thickness=0.005):
    dots = pc @ plane_normal
    mask = np.abs(dots - plane_offset) < thickness
    return pc[mask]

def mems_sweep(pc, plane_normal, offsets, thickness=0.03):
    all_slices = []
    for d in offsets:
        slice_pc = laser_slice(pc, plane_normal, d, thickness)
        if slice_pc.shape[0] > 0:
            all_slices.append(slice_pc)
    if all_slices:
        return np.vstack(all_slices)
    return np.empty((0, 3))
