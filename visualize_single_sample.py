# visualize_single_sample.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data_generation.dataset_generator import SyntheticDatasetGenerator

# ------------------------------------------------------------
# Generate ONE sample and plot its point cloud in 3D
# ------------------------------------------------------------

def visualize_point_cloud(points, title="Point Cloud"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    plt.show()


if __name__ == "__main__":
    # Load generator
    gen = SyntheticDatasetGenerator("config/system_config.yaml")

    # Choose object type: "cube", "cylinder", "complex"
    object_type = "cube"

    # Generate one sample WITHOUT saving dataset
    pc, pose, label = gen.generate_single_sample(
        object_type,
        add_noise=False,
        add_augmentation=False
    )

    print("Number of points:", pc.shape[0])
    print(pc[:10])  # print first 10

    print("Generated point cloud shape:", pc.shape)
    print("Pose:", pose)
    print("Class label:", label)

    # Visualize
    visualize_point_cloud(pc, title=f"Sample {object_type} point cloud")
