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
# 4. Ray Intersection and MEMS angle scan
# ==============================================================

def ray_box_intersection(origin, direction, box_min, box_max):
    invD = 1.0 / direction
    t0s = (box_min - origin) * invD
    t1s = (box_max - origin) * invD

    tmin = np.maximum.reduce(np.minimum(t0s, t1s))
    tmax = np.minimum.reduce(np.maximum(t0s, t1s))

    if tmax < 0 or tmin > tmax:
        return None

    t_hit = tmin if tmin >= 0 else tmax
    return origin + t_hit * direction

def constant_angle_scan_cube(size=1.0,
                             cube_center=np.array([0.0, 0.0, 1.0]),
                             fov_x_deg=4.0,
                             fov_y_deg=4.0,
                             n_x=64,
                             n_y=64,
                             randomize=True):
    if randomize:
        size = size * np.random.uniform(0.8, 1.2)

    s = size / 2.0
    box_min = cube_center - s
    box_max = cube_center + s

    origin = np.array([0.0, 0.0, 0.0])

    thetas_x = np.linspace(-np.radians(fov_x_deg/2), np.radians(fov_x_deg/2), n_x)
    thetas_y = np.linspace(-np.radians(fov_y_deg/2), np.radians(fov_y_deg/2), n_y)

    pts = []

    for th in thetas_y:
        for ph in thetas_x:
            dx = np.tan(ph)
            dy = np.tan(th)
            dz = 1.0

            d = np.array([dx, dy, dz])
            d = d / np.linalg.norm(d)

            hit = ray_box_intersection(origin, d, box_min, box_max)
            if hit is not None:
                pts.append(hit)

    return np.array(pts)

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

