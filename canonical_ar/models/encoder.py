"""
PointNet++ encoder.
Modified to expose intermediate per-point features at SA2 resolution
for use in cross-attention. The forward() method now has two modes:

  encode_full()   — original behaviour, returns global code (B, output_dim)
  encode_local()  — returns (xyz_sa2, feat_sa2) at 128-point resolution
                    for cross-attention, plus the SA3 layer for later pooling

Global code path is preserved for backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from canonical_ar.models.utils import MLP


def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    device = xyz.device
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    for i in range(n_samples):
        indices[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=-1)
    return indices


def ball_query(
    xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    radius: float,
    n_samples: int,
) -> torch.Tensor:
    B, N, _ = xyz.shape
    M = query_xyz.shape[1]
    device = xyz.device
    diff = query_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)
    mask = dist_sq > radius ** 2
    dist_sq[mask] = float("inf")
    _, idx = dist_sq.topk(n_samples, dim=-1, largest=False)
    first = idx[:, :, 0:1].expand_as(idx)
    invalid = mask.gather(2, idx)
    idx[invalid] = first[invalid]
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    if idx.dim() == 2:
        M = idx.shape[1]
        idx_expanded = idx.unsqueeze(-1).expand(B, M, points.shape[-1])
        return points.gather(1, idx_expanded)
    else:
        B, M, K = idx.shape
        idx_flat = idx.reshape(B, -1)
        gathered = index_points(points, idx_flat)
        return gathered.reshape(B, M, K, -1)


class SetAbstraction(nn.Module):
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
        dims = [in_dim + 3] + mlp_dims
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
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.n_points)
        new_xyz = index_points(xyz, fps_idx)
        idx = ball_query(xyz, new_xyz, self.radius, self.n_samples)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        grouped_feat = index_points(features, idx)
        grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        B, M, K, D = grouped.shape
        grouped_flat = grouped.reshape(B * M * K, D)
        out_flat = self.mlp(grouped_flat)
        out = out_flat.reshape(B, M, K, -1)
        new_features = out.max(dim=2).values
        return new_xyz, new_features


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ encoder with two forward modes:

    forward(xyz, feat)              -> global code (B, output_dim)   [original]
    forward_local(xyz, feat)        -> (xyz_128, feat_128, sa3_module)
                                       for cross-attention pipeline
    """
    def __init__(self, input_dim: int = 6, output_dim: int = 256):
        super().__init__()
        feat_dim = input_dim - 3

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
        self.global_proj = MLP(
            input_dim=512,
            hidden_dims=[512],
            output_dim=output_dim,
            use_residual=True,
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Original path — returns global code (B, output_dim)."""
        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)
        global_feat = feat3.max(dim=1).values
        return self.global_proj(global_feat)

    def forward_local(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns intermediate per-point features at SA2 resolution.
        Used by the cross-attention pipeline.

        Returns:
            xyz_sa2:  (B, 128, 3)   positions after SA1+SA2 downsampling
            feat_sa2: (B, 128, 256) per-point features after SA1+SA2
        """
        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        return xyz2, feat2

    def pool_from_local(
        self, xyz_sa2: torch.Tensor, feat_sa2: torch.Tensor
    ) -> torch.Tensor:
        """
        Continue from SA2 features to global code.
        Used after cross-attention enriches feat_sa2.

        Args:
            xyz_sa2:  (B, 128, 3)
            feat_sa2: (B, 128, 256)  — may be cross-attention enriched
        Returns:
            code: (B, output_dim)
        """
        _, feat3 = self.sa3(xyz_sa2, feat_sa2)
        global_feat = feat3.max(dim=1).values
        return self.global_proj(global_feat)
