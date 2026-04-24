"""
PointNet++ encoder.
We implement a simplified but complete version rather than depending
on an external PointNet++ library, so it's fully differentiable and
portable to any cloud compute environment.

Architecture:
  Set Abstraction (SA) layers progressively downsample the point cloud
  while aggregating local geometry into higher-dimensional features.
  Final global feature is a max-pooled latent code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from canonical_ar.models.utils import MLP


def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Farthest point sampling.
    Args:
        xyz: (B, N, 3)
        n_samples: int
    Returns:
        indices: (B, n_samples)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)
    # Start from a random point
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        indices[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)              # (B, N)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)

    return indices


def ball_query(
    xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    radius: float,
    n_samples: int,
) -> torch.Tensor:
    """
    Ball query: for each query point, find n_samples neighbors within radius.
    Args:
        xyz: (B, N, 3) all points
        query_xyz: (B, M, 3) query centers
        radius: float
        n_samples: int
    Returns:
        indices: (B, M, n_samples)
    """
    B, N, _ = xyz.shape
    M = query_xyz.shape[1]
    device = xyz.device

    # Pairwise distances: (B, M, N)
    diff = query_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # (B, M, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)                 # (B, M, N)

    # Mask points outside radius
    mask = dist_sq > radius ** 2
    dist_sq[mask] = float("inf")

    # Take n_samples closest (within radius)
    _, idx = dist_sq.topk(n_samples, dim=-1, largest=False)  # (B, M, n_samples)

    # For points with fewer than n_samples neighbors, repeat the first neighbor
    first = idx[:, :, 0:1].expand_as(idx)
    invalid = mask.gather(2, idx)
    idx[invalid] = first[invalid]

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.
    Args:
        points: (B, N, C)
        idx: (B, M) or (B, M, K)
    Returns:
        gathered: (B, M, C) or (B, M, K, C)
    """
    B = points.shape[0]
    if idx.dim() == 2:
        M = idx.shape[1]
        idx_expanded = idx.unsqueeze(-1).expand(B, M, points.shape[-1])
        return points.gather(1, idx_expanded)
    else:
        B, M, K = idx.shape
        idx_flat = idx.reshape(B, -1)
        gathered = index_points(points, idx_flat)  # (B, M*K, C)
        return gathered.reshape(B, M, K, -1)


class SetAbstraction(nn.Module):
    """
    One PointNet++ Set Abstraction layer.
    Downsamples via FPS, aggregates local neighborhoods via mini-PointNet.
    """
    def __init__(
        self,
        n_points: int,
        radius: float,
        n_samples: int,
        in_dim: int,
        mlp_dims: list[int],
    ):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_samples = n_samples

        # Mini PointNet MLP applied to each local point
        dims = [in_dim + 3] + mlp_dims  # +3 for relative xyz
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = mlp_dims[-1]

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            features: (B, N, C)
        Returns:
            new_xyz: (B, n_points, 3)
            new_features: (B, n_points, out_dim)
        """
        B, N, _ = xyz.shape

        # Farthest point sampling
        fps_idx = farthest_point_sample(xyz, self.n_points)  # (B, n_points)
        new_xyz = index_points(xyz, fps_idx)                  # (B, n_points, 3)

        # Ball query
        idx = ball_query(xyz, new_xyz, self.radius, self.n_samples)
        # (B, n_points, n_samples)

        # Gather neighbor points and features
        grouped_xyz = index_points(xyz, idx)        # (B, n_points, n_samples, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)          # relative coordinates
        grouped_feat = index_points(features, idx)  # (B, n_points, n_samples, C)

        grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        # (B, n_points, n_samples, C+3)

        # Apply MLP then max pool over neighborhood
        B, M, K, D = grouped.shape
        grouped_flat = grouped.reshape(B * M * K, D)
        out_flat = self.mlp(grouped_flat)            # needs BN, so flatten
        out = out_flat.reshape(B, M, K, -1)
        new_features = out.max(dim=2).values         # (B, n_points, out_dim)

        return new_xyz, new_features


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ encoder.
    Three SA layers progressively compress the point cloud.
    Output: global latent code of shape (B, output_dim).
    """
    def __init__(self, input_dim: int = 6, output_dim: int = 256):
        super().__init__()
        # input_dim = 3 (xyz) + 3 (normals) = 6 by default
        feat_dim = input_dim - 3  # features beyond xyz

        self.sa1 = SetAbstraction(
            n_points=512, radius=0.2, n_samples=32,
            in_dim=feat_dim, mlp_dims=[64, 64, 128],
        )
        self.sa2 = SetAbstraction(
            n_points=128, radius=0.4, n_samples=64,
            in_dim=128, mlp_dims=[128, 128, 256],
        )
        self.sa3 = SetAbstraction(
            n_points=32, radius=0.8, n_samples=128,
            in_dim=256, mlp_dims=[256, 256, 512],
        )

        # Global feature projection
        self.global_proj = MLP(
            input_dim=512,
            hidden_dims=[512],
            output_dim=output_dim,
            use_residual=True,
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) — point positions
            features: (B, N, C) — per-point features (normals, color, etc.)
        Returns:
            code: (B, output_dim)
        """
        # SA layers
        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)

        # Global max pool then project
        global_feat = feat3.max(dim=1).values   # (B, 512)
        code = self.global_proj(global_feat)    # (B, output_dim)
        return code
