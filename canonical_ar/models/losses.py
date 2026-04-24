"""
Loss functions for the deformation field model.

Three components:
1. Chamfer loss: deformed canonical points should match the observed point cloud
2. Smoothness loss: nearby canonical points should have similar displacements
3. Attachment loss: virtual content points should land at their ground-truth positions

The attachment loss is the actual task objective — it's what makes this
useful for AR rather than just a generic deformation estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Pairwise distances (B, N, M)
    diff = pts_a.unsqueeze(2) - pts_b.unsqueeze(1)  # (B, N, M, 3)
    dist_sq = (diff ** 2).sum(dim=-1)                # (B, N, M)

    # For each point in a, find nearest in b
    min_a_to_b = dist_sq.min(dim=2).values.mean()   # scalar
    # For each point in b, find nearest in a
    min_b_to_a = dist_sq.min(dim=1).values.mean()   # scalar

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

    # Pairwise distances in canonical space
    diff = canonical_pts.unsqueeze(2) - canonical_pts.unsqueeze(1)  # (B, N, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, N)

    # Zero out self-distances
    eye = torch.eye(N, device=canonical_pts.device).unsqueeze(0)
    dist_sq = dist_sq + eye * 1e9

    # K nearest neighbors
    _, idx = dist_sq.topk(k_neighbors, dim=-1, largest=False)  # (B, N, k)

    # Gather neighbor displacements
    idx_exp = idx.unsqueeze(-1).expand(B, N, k_neighbors, 3)
    disp_exp = displacements.unsqueeze(2).expand(B, N, N, 3)
    neighbor_disp = disp_exp.gather(2, idx_exp)  # (B, N, k, 3)

    # Laplacian: displacement - mean(neighbor displacements)
    mean_neighbor_disp = neighbor_disp.mean(dim=2)  # (B, N, 3)
    lap = displacements - mean_neighbor_disp         # (B, N, 3)

    return (lap ** 2).sum(dim=-1).mean()


def magnitude_loss(displacements: torch.Tensor) -> torch.Tensor:
    """
    Penalize large deformations — prevents degenerate solutions where
    the field collapses everything to a single point.
    """
    return (displacements ** 2).sum(dim=-1).mean()


class DeformationLoss(nn.Module):
    """
    Combined loss for training the deformation field model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        loss_cfg = cfg.train.loss
        self.w_chamfer = loss_cfg.chamfer_weight
        self.w_smooth = loss_cfg.smoothness_weight
        self.w_magnitude = loss_cfg.magnitude_weight
        self.w_attachment = loss_cfg.attachment_weight

    def forward(
        self,
        # Model outputs
        deformed_canonical: torch.Tensor,    # (B, N_c, 3) deformed canonical pts
        displacements: torch.Tensor,          # (B, Q, 3)  virtual content displacements
        deformed_query: torch.Tensor,         # (B, Q, 3)  virtual content deformed
        # Targets
        obs_xyz: torch.Tensor,               # (B, N_o, 3) observed point cloud
        canonical_pts: torch.Tensor,          # (B, N_c, 3) for smoothness loss
        gt_deformed_query: torch.Tensor,      # (B, Q, 3)  ground truth virtual content positions
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict of individual losses and total loss.
        """
        l_chamfer = chamfer_distance(deformed_canonical, obs_xyz)
        l_smooth = smoothness_loss(canonical_pts, displacements)
        l_magnitude = magnitude_loss(displacements)
        l_attach = F.mse_loss(deformed_query, gt_deformed_query)

        total = (
            self.w_chamfer * l_chamfer
            + self.w_smooth * l_smooth
            + self.w_magnitude * l_magnitude
            + self.w_attachment * l_attach
        )

        return {
            "loss": total,
            "loss_chamfer": l_chamfer.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_magnitude": l_magnitude.detach(),
            "loss_attachment": l_attach.detach(),
        }
