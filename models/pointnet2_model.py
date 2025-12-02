"""
PointNet++ Architecture for 6D Pose Estimation
Predicts [x, y, z, qw, qx, qy, qz] from point cloud
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between points.
    
    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling.
    
    Args:
        xyz: (B, N, 3)
        npoint: Number of points to sample
    Returns:
        centroids: (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Query ball point grouping.
    
    Args:
        radius: Local region radius
        nsample: Max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction Layer"""
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C)
        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, mlp[-1])
        """
        B, N, C = xyz.shape
        S = self.npoint
        
        # Sample points
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).repeat(1, 1, 3))  # (B, npoint, 3)
        
        # Query ball
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
        
        # Group points
        grouped_xyz = torch.gather(xyz, 1, idx.view(B, S * self.nsample, 1).repeat(1, 1, 3)).view(B, S, self.nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # Relative coordinates
        
        if points is not None:
            grouped_points = torch.gather(points, 1, idx.view(B, S * self.nsample, 1).repeat(1, 1, points.shape[-1])).view(B, S, self.nsample, -1)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        
        # MLP
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        
        # Max pooling
        new_points = torch.max(grouped_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        
        return new_xyz, new_points


class PointNet2PoseEstimation(nn.Module):
    """
    PointNet++ for 6D Pose Estimation.
    Outputs: [x, y, z, qw, qx, qy, qz]
    """
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, input_channels, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        
        # Global feature
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        
        # Position head
        self.fc_pos = nn.Linear(128, 3)
        
        # Orientation head (quaternion)
        self.fc_rot = nn.Linear(128, 4)
    
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) point cloud
        Returns:
            pose: (B, 7) [x, y, z, qw, qx, qy, qz]
        """
        B, N, _ = xyz.shape
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # Global max pooling
        global_feat = torch.max(l2_points, 1)[0]  # (B, 256)
        
        # Shared layers
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        # Position prediction
        position = self.fc_pos(x)  # (B, 3)
        
        # Orientation prediction (quaternion)
        quaternion = self.fc_rot(x)  # (B, 4)
        quaternion = F.normalize(quaternion, p=2, dim=1)  # Normalize to unit quaternion
        
        # Concatenate pose
        pose = torch.cat([position, quaternion], dim=1)  # (B, 7)
        
        return pose


# ==============================================================
# LOSS FUNCTIONS
# ==============================================================

class PoseLoss(nn.Module):
    """
    Combined loss for position and orientation.
    """
    
    def __init__(self, pos_weight=1.0, rot_weight=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
    
    def forward(self, pred_pose, gt_pose):
        """
        Args:
            pred_pose: (B, 7) [x, y, z, qw, qx, qy, qz]
            gt_pose: (B, 7) [x, y, z, qw, qx, qy, qz]
        Returns:
            loss: Scalar
        """
        # Position loss (Smooth L1)
        pred_pos = pred_pose[:, :3]
        gt_pos = gt_pose[:, :3]
        pos_loss = F.smooth_l1_loss(pred_pos, gt_pos)
        
        # Orientation loss (Geodesic distance on SO(3))
        pred_quat = pred_pose[:, 3:]
        gt_quat = gt_pose[:, 3:]
        
        # Quaternion distance: d = 1 - |q1·q2|
        dot_product = torch.abs(torch.sum(pred_quat * gt_quat, dim=1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        rot_loss = (1.0 - dot_product).mean()
        
        # Total loss
        total_loss = self.pos_weight * pos_loss + self.rot_weight * rot_loss
        
        return total_loss, pos_loss, rot_loss


# ==============================================================
# MODEL FACTORY
# ==============================================================

def create_model(config=None):
    """
    Create PointNet++ model for pose estimation.
    
    Args:
        config: Optional configuration dictionary
    Returns:
        model: PointNet2PoseEstimation instance
    """
    model = PointNet2PoseEstimation(input_channels=3)
    return model