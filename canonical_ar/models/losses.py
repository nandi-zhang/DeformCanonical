"""
Loss functions for the deformation field model.

Two geometric components only:
1. Chamfer loss: deformed canonical points should match the observed point cloud
2. Smoothness loss: nearby canonical points should have similar displacements
3. Magnitude loss: prevent degenerate collapse

Attachment is NOT a training loss — it's an evaluation metric computed
in evaluate.py. If the deformation field is correct, virtual content
attachment is correct by definition: content registered at canonical
coordinates gets mapped to the right observation-space position by the
same field. Supervising attachment separately is redundant with Chamfer
and adds noise from the sparse query point sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


def chamfer_distance(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between two point sets.
    Args:
        pts_a: (B, N, 3)
        pts_b: (B, M, 3)
    Returns:
        loss: scalar
    """
    diff = pts_a.unsqueeze(2) - pts_b.unsqueeze(1)  # (B, N, M, 3)
    dist_sq = (diff ** 2).sum(dim=-1)                # (B, N, M)
    min_a_to_b = dist_sq.min(dim=2).values.mean()
    min_b_to_a = dist_sq.min(dim=1).values.mean()
    return min_a_to_b + min_b_to_a


def smoothness_loss(
    canonical_pts: torch.Tensor,
    displacements: torch.Tensor,
    k_neighbors: int = 8,
) -> torch.Tensor:
    """
    Laplacian smoothness: each point's displacement should be close to
    the mean displacement of its k nearest canonical neighbors.
    Args:
        canonical_pts: (B, N, 3)
        displacements: (B, N, 3)
    Returns:
        loss: scalar
    """
    B, N, _ = canonical_pts.shape
    diff = canonical_pts.unsqueeze(2) - canonical_pts.unsqueeze(1)  # (B, N, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)                               # (B, N, N)
    eye = torch.eye(N, device=canonical_pts.device).unsqueeze(0)
    dist_sq = dist_sq + eye * 1e9
    _, idx = dist_sq.topk(k_neighbors, dim=-1, largest=False)       # (B, N, k)
    idx_exp = idx.unsqueeze(-1).expand(B, N, k_neighbors, 3)
    disp_exp = displacements.unsqueeze(2).expand(B, N, N, 3)
    neighbor_disp = disp_exp.gather(2, idx_exp)                     # (B, N, k, 3)
    mean_neighbor_disp = neighbor_disp.mean(dim=2)                  # (B, N, 3)
    lap = displacements - mean_neighbor_disp
    return (lap ** 2).sum(dim=-1).mean()


def magnitude_loss(displacements: torch.Tensor) -> torch.Tensor:
    """Penalize large deformations — prevents collapse."""
    return (displacements ** 2).sum(dim=-1).mean()


class DeformationLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        loss_cfg = cfg.train.loss
        self.w_chamfer = loss_cfg.chamfer_weight
        self.w_smooth = loss_cfg.smoothness_weight
        self.w_magnitude = loss_cfg.magnitude_weight

    def forward(
        self,
        deformed_canonical: torch.Tensor,  # (B, N_c, 3)
        displacements: torch.Tensor,        # (B, N_c, 3)
        obs_xyz: torch.Tensor,             # (B, N_o, 3)
        canonical_pts: torch.Tensor,        # (B, N_c, 3)
    ) -> dict[str, torch.Tensor]:
        l_chamfer = chamfer_distance(deformed_canonical, obs_xyz)
        l_smooth = smoothness_loss(canonical_pts, displacements)
        l_magnitude = magnitude_loss(displacements)

        total = (
            self.w_chamfer * l_chamfer
            + self.w_smooth * l_smooth
            + self.w_magnitude * l_magnitude
        )

        return {
            "loss": total,
            "loss_chamfer": l_chamfer.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_magnitude": l_magnitude.detach(),
        }
