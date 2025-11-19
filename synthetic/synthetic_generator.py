import numpy as np
from transforms3d.euler import euler2mat

# ==============================================================
# 1. Surface Sampling for CUBES & CYLINDERS
# ==============================================================

def sample_cube_surface(size=1.0, num_points=2000, randomize=True):
    if randomize:
        size = size * np.random.uniform(0.8, 1.2)

    pts = []
    s = size / 2

    for _ in range(num_points):
        face = np.random.randint(0, 6)
        x = np.random.uniform(-s, s)
        y = np.random.uniform(-s, s)

        if face == 0: pts.append([ s,  x,  y ])
        if face == 1: pts.append([-s,  x,  y ])
        if face == 2: pts.append([ x,  s,  y ])
        if face == 3: pts.append([ x, -s,  y ])
        if face == 4: pts.append([ x,  y,  s ])
        if face == 5: pts.append([ x,  y, -s ])

    return np.array(pts)


def sample_cylinder_surface(radius=0.5, height=1.0, num_points=2000, randomize=True):
    if randomize:
        radius = radius * np.random.uniform(0.8, 1.2)
        height = height * np.random.uniform(0.8, 1.2)

    pts = []
    h = height / 2

    for _ in range(num_points):
        choice = np.random.choice(["side", "top", "bottom"])

        if choice == "side":
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.uniform(-h, h)
            pts.append([radius*np.cos(theta), radius*np.sin(theta), z])

        elif choice == "top":
            r = np.sqrt(np.random.uniform(0, radius**2))
            theta = np.random.uniform(0, 2*np.pi)
            pts.append([r*np.cos(theta), r*np.sin(theta), h])

        elif choice == "bottom":
            r = np.sqrt(np.random.uniform(0, radius**2))
            theta = np.random.uniform(0, 2*np.pi)
            pts.append([r*np.cos(theta), r*np.sin(theta), -h])

    return np.array(pts)


# ==============================================================
# 2. Apply Pose (Rotation + Translation)
# ==============================================================

def limited_random_pose(max_deg=30, max_trans=0.1):
    ax, ay, az = np.radians(np.random.uniform(-max_deg, max_deg, 3))
    R = euler2mat(ax, ay, az)
    t = np.random.uniform(-max_trans, max_trans, 3)
    return R, t

def apply_pose(pc, R, t):
    return (R @ pc.T).T + t


# ==============================================================
# 3. Noise, Dropout, Visibility
# ==============================================================

def add_noise(pc, sigma=0.005):
    return pc + np.random.normal(0, sigma, pc.shape)

def random_dropout(pc, drop_rate=0.1):
    mask = np.random.rand(pc.shape[0]) > drop_rate
    return pc[mask]

def visibility_mask(pc, normal=np.array([0, 0, 1]), threshold=0.0):
    dots = pc @ normal
    return pc[dots > threshold]


# ==============================================================
# 4. Laser Slice & MEMS Sweep
# ==============================================================

def laser_slice(pc, plane_normal, plane_offset, thickness=0.01):
    d = pc @ plane_normal
    mask = np.abs(d - plane_offset) < thickness
    return pc[mask]

def mems_sweep(pc, plane_normal, thetas_x, thetas_y, thickness=0.01):
    slices = []
    idx = 0

    for tx in thetas_x:
        for ty in thetas_y:
            plane_offset = 0.5 * (np.sin(tx) + np.sin(ty))  # simple model, adjustable later
            sl = laser_slice(pc, plane_normal, plane_offset, thickness)
            if sl.shape[0] > 0:
                slices.append(sl)

    if len(slices) == 0:
        return np.empty((0, 3))

    return np.vstack(slices)


# ==============================================================
# 5. Normalize Point Count (fixed N)
# ==============================================================

def normalize_point_count(pc, target_n=1024):
    n = pc.shape[0]

    if n >= target_n:
        idx = np.random.choice(n, target_n, replace=False)
        return pc[idx]
    else:
        # pad by repeating random points
        idx = np.random.choice(n, target_n-n, replace=True)
        return np.vstack([pc, pc[idx]])


# ==============================================================
# 6. High-level Object Generator
# ==============================================================

def generate_object_pointcloud(object_type="cube"):
    if object_type == "cube":
        return sample_cube_surface()
    elif object_type == "cylinder":
        return sample_cylinder_surface()
    else:
        raise ValueError("Unsupported object type. Add CAD support later.")


# ==============================================================
# - Helper function
# ==============================================================

import numpy as np

def simple_two_face_scan(size=1.0, grid_n=32, noise=0.002, randomize=True):
    if randomize:
        size = size * np.random.uniform(0.8, 1.2)
    s = size / 2.0

    u = np.linspace(-s, s, grid_n)
    v = np.linspace(-s, s, grid_n)

    xs, ys = np.meshgrid(u, v)
    
    front = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, s)], axis=1)

    zs, ys2 = np.meshgrid(u, v)
    side = np.stack([np.full(zs.size, s), ys2.ravel(), zs.ravel()], axis=1)

    pts = np.vstack([front, side])

    pts += np.random.normal(0, noise, pts.shape)
    return pts
