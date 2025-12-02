"""
Synthetic Dataset Generator
Generates training data for laser triangulation pose estimation
"""

import numpy as np
import h5py
from tqdm import tqdm
import yaml
from pathlib import Path

from src.data_generation.geometry import (
    generate_object,
    random_pose,
    apply_pose,
    add_noise,
    random_dropout,
    add_outliers,
    normalize_point_cloud
)


from src.data_generation.laser_simulator import simulate_laser_scan


class SyntheticDatasetGenerator:
    """
    Generate synthetic training dataset for pose estimation.
    """
    
    def __init__(self, config_path="config/system_config.yaml"):
        """
        Initialize generator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.object_types = self.config['training_data']['dataset_size']['classes']
        self.samples_per_class = self.config['training_data']['dataset_size']['samples_per_class']
        self.target_points = self.config['point_cloud']['target_points']
        
        # Pose variation ranges
        pose_var = self.config['training_data']['pose_variation']
        self.position_range = {
            'x': pose_var['position']['x_range'],
            'y': pose_var['position']['y_range'],
            'z': pose_var['position']['z_range']
        }
        self.rotation_range = {
            'roll': pose_var['rotation']['roll_range_deg'],
            'pitch': pose_var['rotation']['pitch_range_deg'],
            'yaw': pose_var['rotation']['yaw_range_deg']
        }
        
        # Noise parameters
        noise_cfg = self.config['training_data']['noise_model']
        self.noise_sigma = noise_cfg['point_noise_sigma']
        self.dropout_rate = noise_cfg['dropout_rate']
        self.outlier_rate = noise_cfg['outlier_rate']
    
    
    def generate_single_sample(self, object_type, add_noise=True, add_augmentation=True):
        """
        Generate a single training sample.
        
        Args:
            object_type: "cube", "cylinder", or "complex"
            add_noise: Whether to add realistic noise
            add_augmentation: Whether to apply data augmentation
            
        Returns:
            point_cloud: (target_points, 3) normalized point cloud
            pose: (7,) pose vector [x, y, z, qw, qx, qy, qz]
            object_class: Integer class label
        """
        # Generate object surface points
        object_points = generate_object(object_type, size_variation=True)
        
        # Generate random pose
        position, quaternion = random_pose(self.position_range, self.rotation_range)
        
        # Simulate laser scanning
        scanned_cloud = simulate_laser_scan(
            object_points,
            position,
            quaternion,
            self.config,
            add_noise=add_noise
        )
        
        # Handle empty scans (rare but possible)
        if len(scanned_cloud) < 100:
            # Fallback: use ideal object points with pose
            scanned_cloud = apply_pose(object_points, position, quaternion)
        
        # Apply noise and augmentation
        if add_augmentation:
            scanned_cloud = add_noise(scanned_cloud, sigma=self.noise_sigma)
            scanned_cloud = random_dropout(scanned_cloud, dropout_rate=self.dropout_rate)
            scanned_cloud = add_outliers(scanned_cloud, outlier_rate=self.outlier_rate)
        
        # Normalize to fixed point count
        if len(scanned_cloud) > 0:
            point_cloud = normalize_point_cloud(scanned_cloud, target_count=self.target_points)
        else:
            # Emergency fallback
            point_cloud = np.zeros((self.target_points, 3))
        
        # Encode pose as 7D vector
        pose = np.concatenate([position, quaternion])
        
        # Encode object class
        class_mapping = {"cube": 0, "cylinder": 1, "complex": 2}
        object_class = class_mapping[object_type]
        
        return point_cloud.astype(np.float32), pose.astype(np.float32), object_class
    
    
    def generate_dataset(self, output_path="data/synthetic/dataset.h5"):
        """
        Generate complete training dataset.
        
        Args:
            output_path: Path to save HDF5 file
        """
        # Calculate total samples
        total_samples = self.samples_per_class * len(self.object_types)
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            point_clouds = f.create_dataset(
                'point_clouds',
                shape=(total_samples, self.target_points, 3),
                dtype=np.float32,
                compression='gzip'
            )
            
            poses = f.create_dataset(
                'poses',
                shape=(total_samples, 7),
                dtype=np.float32
            )
            
            labels = f.create_dataset(
                'labels',
                shape=(total_samples,),
                dtype=np.int32
            )
            
            # Generate samples
            idx = 0
            for object_type in self.object_types:
                print(f"\nGenerating {self.samples_per_class} samples for {object_type}...")
                
                for _ in tqdm(range(self.samples_per_class)):
                    pc, pose, label = self.generate_single_sample(object_type)
                    
                    point_clouds[idx] = pc
                    poses[idx] = pose
                    labels[idx] = label
                    
                    idx += 1
            
            # Save metadata
            f.attrs['total_samples'] = total_samples
            f.attrs['target_points'] = self.target_points
            f.attrs['object_types'] = self.object_types
            f.attrs['pose_format'] = '[x, y, z, qw, qx, qy, qz]'
            
        print(f"\nDataset saved to {output_path}")
        print(f"Total samples: {total_samples}")
        print(f"File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")
    
    
    def generate_splits(self, dataset_path="data/synthetic/dataset.h5"):
        """
        Generate train/val/test splits from dataset.
        
        Args:
            dataset_path: Path to HDF5 dataset
        """
        split_config = self.config['training_data']['split']
        
        with h5py.File(dataset_path, 'r') as f:
            total = len(f['point_clouds'])
            indices = np.random.permutation(total)
            
            n_train = int(total * split_config['train'])
            n_val = int(total * split_config['val'])
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            # Save splits
            split_path = Path(dataset_path).parent / "splits.npz"
            np.savez(
                split_path,
                train=train_idx,
                val=val_idx,
                test=test_idx
            )
            
            print(f"\nSplits saved to {split_path}")
            print(f"Train: {len(train_idx)} samples")
            print(f"Val: {len(val_idx)} samples")
            print(f"Test: {len(test_idx)} samples")


# ==============================================================
# MAIN EXECUTION
# ==============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("--config", default="config/system_config.yaml", help="Config file path")
    parser.add_argument("--output", default="data/synthetic/dataset.h5", help="Output HDF5 file")
    parser.add_argument("--splits", action="store_true", help="Generate train/val/test splits")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = SyntheticDatasetGenerator(args.config)
    generator.generate_dataset(args.output)
    
    if args.splits:
        generator.generate_splits(args.output)
    
    print("\n✅ Dataset generation complete!")