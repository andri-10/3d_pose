"""
Laser Triangulation Simulator
Simulates MEMS scanning and laser-camera triangulation
"""

import numpy as np
from scipy.spatial.transform import Rotation


class LaserTriangulationSimulator:
    """
    Simulates laser triangulation scanning system.
    """
    
    def __init__(self, config):
        """
        Initialize simulator with system configuration.
        
        Args:
            config: Dictionary with camera, laser, MEMS parameters
        """
        # Camera parameters
        self.camera_pos = np.array(config['geometry']['camera']['position'])
        self.fx = config['camera']['intrinsics']['fx']
        self.fy = config['camera']['intrinsics']['fy']
        self.cx = config['camera']['intrinsics']['cx']
        self.cy = config['camera']['intrinsics']['cy']
        self.img_width = config['camera']['resolution'][0]
        self.img_height = config['camera']['resolution'][1]
        
        # Laser/MEMS parameters
        self.laser_pos = np.array(config['geometry']['laser_mems']['effective_origin'])
        self.baseline = config['geometry']['baseline']
        
        # MEMS scan parameters
        self.angle_range = config['mems']['scan']['angle_range_deg']
        self.grid_size = config['mems']['scan']['grid_size']
        
        # Noise parameters
        self.pixel_noise_sigma = config['training_data']['noise_model']['pixel_noise_sigma']
        self.mems_noise_sigma = config['training_data']['noise_model']['mems_angle_noise_sigma']
        
        # Build camera intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)
    
    
    def get_mems_scan_angles(self):
        """
        Generate grid of MEMS angles for scanning.
        
        Returns:
            angles: (N, 2) array of (theta_x, theta_y) in degrees
        """
        theta_x = np.linspace(self.angle_range[0], self.angle_range[1], self.grid_size[0])
        theta_y = np.linspace(self.angle_range[0], self.angle_range[1], self.grid_size[1])
        
        # Create grid (raster scan pattern)
        angles = []
        for ty in theta_y:
            for tx in theta_x:
                angles.append([tx, ty])
        
        return np.array(angles)
    
    
    def laser_beam_direction(self, theta_x_deg, theta_y_deg):
        """
        Calculate laser beam direction after MEMS reflection.
        
        Note: Beam deflects 2× mirror angle (law of reflection)
        
        Args:
            theta_x_deg: MEMS mechanical angle X (degrees)
            theta_y_deg: MEMS mechanical angle Y (degrees)
            
        Returns:
            direction: (3,) normalized beam direction vector
        """
        # Optical deflection = 2 × mechanical angle
        deflection_x = 2 * np.radians(theta_x_deg)
        deflection_y = 2 * np.radians(theta_y_deg)
        
        # Initial beam direction (forward)
        direction = np.array([
            np.tan(deflection_x),
            np.tan(deflection_y),
            1.0
        ])
        
        # Normalize
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    
    def laser_plane_equation(self, theta_x_deg, theta_y_deg):
        """
        Calculate laser plane equation for given MEMS angles.
        
        Plane passes through laser origin with normal perpendicular to beam.
        
        Args:
            theta_x_deg: MEMS angle X (degrees)
            theta_y_deg: MEMS angle Y (degrees)
            
        Returns:
            plane: (4,) array [a, b, c, d] for plane equation ax + by + cz + d = 0
        """
        # Laser beam direction
        beam_dir = self.laser_beam_direction(theta_x_deg, theta_y_deg)
        
        # Plane normal (perpendicular to beam and baseline)
        baseline_vec = self.laser_pos - self.camera_pos
        normal = np.cross(beam_dir, baseline_vec)
        normal = normal / np.linalg.norm(normal)
        
        # Plane equation: n · (p - p0) = 0  →  n·p + d = 0
        d = -np.dot(normal, self.laser_pos)
        
        plane = np.array([normal[0], normal[1], normal[2], d])
        
        return plane
    
    
    def ray_plane_intersection(self, ray_origin, ray_direction, plane):
        """
        Calculate intersection of ray with plane.
        
        Args:
            ray_origin: (3,) ray starting point
            ray_direction: (3,) ray direction (normalized)
            plane: (4,) plane equation [a, b, c, d]
            
        Returns:
            point: (3,) intersection point, or None if no intersection
        """
        a, b, c, d = plane
        normal = np.array([a, b, c])
        
        # Check if ray is parallel to plane
        denom = np.dot(normal, ray_direction)
        if np.abs(denom) < 1e-6:
            return None
        
        # Calculate intersection distance t
        t = -(np.dot(normal, ray_origin) + d) / denom
        
        # Check if intersection is in front of camera
        if t < 0:
            return None
        
        # Calculate intersection point
        point = ray_origin + t * ray_direction
        
        return point
    
    
    def pixel_to_ray(self, u, v):
        """
        Convert pixel coordinates to 3D ray direction from camera.
        
        Args:
            u, v: Pixel coordinates
            
        Returns:
            ray_direction: (3,) normalized ray direction
        """
        # Homogeneous pixel coordinates
        pixel_h = np.array([u, v, 1.0])
        
        # Unproject to 3D direction
        direction = self.K_inv @ pixel_h
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    
    def project_point_to_pixel(self, point_3d):
        """
        Project 3D point to pixel coordinates.
        
        Args:
            point_3d: (3,) point in world coordinates
            
        Returns:
            u, v: Pixel coordinates, or None if behind camera
        """
        # Transform to camera frame (camera at origin)
        point_cam = point_3d - self.camera_pos
        
        # Check if in front of camera
        if point_cam[2] <= 0:
            return None, None
        
        # Project to pixel
        pixel_h = self.K @ point_cam
        u = pixel_h[0] / pixel_h[2]
        v = pixel_h[1] / pixel_h[2]
        
        # Check if in image bounds
        if u < 0 or u >= self.img_width or v < 0 or v >= self.img_height:
            return None, None
        
        return u, v
    
    
    def scan_object(self, object_points, add_noise=True):
        """
        Simulate scanning an object with MEMS laser triangulation.
        
        Args:
            object_points: (N, 3) array of points on object surface
            add_noise: If True, add realistic noise
            
        Returns:
            scanned_points: (M, 3) array of reconstructed 3D points
        """
        scan_angles = self.get_mems_scan_angles()
        scanned_points = []
        
        for theta_x, theta_y in scan_angles:
            # Add MEMS angle noise
            if add_noise:
                theta_x += np.random.normal(0, self.mems_noise_sigma)
                theta_y += np.random.normal(0, self.mems_noise_sigma)
            
            # Get laser plane for this MEMS angle
            laser_plane = self.laser_plane_equation(theta_x, theta_y)
            
            # Find which object points are hit by laser
            # (Simplified: check which points are close to laser plane)
            a, b, c, d = laser_plane
            normal = np.array([a, b, c])
            distances = np.abs(object_points @ normal + d)
            
            # Points within 1mm of laser plane are "illuminated"
            hit_mask = distances < 0.001
            hit_points = object_points[hit_mask]
            
            if len(hit_points) == 0:
                continue
            
            # For each hit point, check if visible by camera
            for point in hit_points:
                u, v = self.project_point_to_pixel(point)
                
                if u is None:
                    continue  # Point not in camera view
                
                # Add pixel noise
                if add_noise:
                    u += np.random.normal(0, self.pixel_noise_sigma)
                    v += np.random.normal(0, self.pixel_noise_sigma)
                
                # Triangulate: intersect camera ray with laser plane
                ray_dir = self.pixel_to_ray(u, v)
                reconstructed = self.ray_plane_intersection(self.camera_pos, ray_dir, laser_plane)
                
                if reconstructed is not None:
                    scanned_points.append(reconstructed)
        
        if len(scanned_points) == 0:
            return np.array([])
        
        return np.array(scanned_points)


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def simulate_laser_scan(object_points, pose_position, pose_quaternion, config, add_noise=True):
    """
    Complete simulation: place object at pose and scan it.
    
    Args:
        object_points: (N, 3) points in object frame
        pose_position: (3,) object position in world frame
        pose_quaternion: (4,) object orientation [qw, qx, qy, qz]
        config: System configuration dictionary
        add_noise: Whether to add realistic noise
        
    Returns:
        point_cloud: (M, 3) scanned point cloud
    """
    from src.data_generation.geometry import apply_pose
    
    # Transform object to world frame
    object_in_world = apply_pose(object_points, pose_position, pose_quaternion)
    
    # Simulate scanning
    simulator = LaserTriangulationSimulator(config)
    scanned_cloud = simulator.scan_object(object_in_world, add_noise=add_noise)
    
    return scanned_cloud