"""
Loss functions for the deformation field model.

Primary loss: dense point-to-point correspondence
  For each canonical point, we know exactly where it should be in
  observation space (from synthetic deformation GT). Direct L2 supervision
  on all 2048 points — no Chamfer approximation needed.

Visibility-weighted: visible points (whose GT falls near the observed cloud)
  get full weight. Occluded points get reduced weight — the model should
  still predict plausible displacements for them (via smoothness), but
  we don't penalize it as harshly since real occluded regions are ambiguous.

Secondary loss: InfoNCE contrastive on cross-attention features
  Teaches the encoder to produce discriminative local features that
  transfer to real objects. For each canonical SA2 point, its enriched
  feature should match the feature of the nearest observed SA2 point
  (positive pair) and be dissimilar to all others (negatives).

Regularization:
  Smoothness: Laplacian on displacement field
  Magnitude: tiny weight, prevents collapse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def correspondence_loss(
    deformed_canonical: torch.Tensor,     # (B, N, 3) predicted
    gt_deformed_canonical: torch.Tensor,  # (B, N, 3) ground truth
    visible_mask: torch.Tensor,           # (B, N) float, 1=visible 0=occluded
    occluded_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Visibility-weighted dense correspondence loss.
    Visible points: full L2 supervision.
    Occluded points: reduced weight — model should extrapolate smoothly.
    """
    per_point = (deformed_canonical - gt_deformed_canonical).norm(dim=-1)  # (B, N)

    # Separate visible and occluded
    vis_loss = (per_point * visible_mask).sum() / (visible_mask.sum() + 1e-6)
    occ_mask = 1.0 - visible_mask
    occ_loss = (per_point * occ_mask).sum() / (occ_mask.sum() + 1e-6)

    total = vis_loss + occluded_weight * occ_loss
    return {
        "loss_correspondence": total,
        "loss_corr_visible": vis_loss.detach(),
        "loss_corr_occluded": occ_loss.detach(),
    }


def infonce_loss(
    can_feat: torch.Tensor,   # (B, N, D) enriched canonical SA2 features
    obs_feat: torch.Tensor,   # (B, M, D) observed SA2 features
    can_xyz: torch.Tensor,    # (B, N, 3) canonical SA2 positions
    gt_deformed_can: torch.Tensor,  # (B, N, 3) GT deformed positions of SA2 pts
    obs_xyz: torch.Tensor,    # (B, M, 3) observed SA2 positions
    temperature: float = 0.1,
    validity_threshold: float = 0.15,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss on cross-attention features.

    For each canonical SA2 point i:
      - Positive: observed SA2 point j* nearest to GT deformed position of i
      - Negatives: all other observed SA2 points
      - Valid only if dist(gt_deformed_i, nearest_obs) < validity_threshold

    This teaches: "canonical region i's features should match the
    observation features at the region it deformed to."
    """
    B, N, D = can_feat.shape
    M = obs_feat.shape[1]

    # Normalize features for cosine similarity
    can_norm = F.normalize(can_feat, dim=-1)   # (B, N, D)
    obs_norm = F.normalize(obs_feat, dim=-1)   # (B, M, D)

    # Similarity matrix: (B, N, M)
    sim = torch.bmm(can_norm, obs_norm.transpose(1, 2)) / temperature

    # Find positive pair: nearest observed SA2 point to GT deformed position
    # gt_deformed_can: (B, N, 3), obs_xyz: (B, M, 3)
    diff = gt_deformed_can.unsqueeze(2) - obs_xyz.unsqueeze(1)  # (B, N, M, 3)
    dist = diff.norm(dim=-1)                                      # (B, N, M)
    min_dist, pos_idx = dist.min(dim=-1)                         # (B, N)

    # Validity mask: only supervise canonical points whose GT is near observed
    valid = (min_dist < validity_threshold)  # (B, N) bool

    if valid.sum() == 0:
        return torch.tensor(0.0, device=can_feat.device)

    # InfoNCE: log(exp(sim[i,j*]) / sum_k exp(sim[i,k]))
    # = sim[i,j*] - log(sum_k exp(sim[i,k]))
    log_denom = torch.logsumexp(sim, dim=-1)  # (B, N)

    # Gather positive similarity
    pos_idx_exp = pos_idx.unsqueeze(-1)       # (B, N, 1)
    pos_sim = sim.gather(2, pos_idx_exp).squeeze(-1)  # (B, N)

    per_point_loss = -(pos_sim - log_denom)   # (B, N)

    # Average only over valid points
    loss = (per_point_loss * valid.float()).sum() / (valid.float().sum() + 1e-6)
    return loss


def smoothness_loss(
    canonical_pts: torch.Tensor,
    displacements: torch.Tensor,
    k_neighbors: int = 8,
) -> torch.Tensor:
    """Laplacian smoothness regularization."""
    B, N, _ = canonical_pts.shape
    diff = canonical_pts.unsqueeze(2) - canonical_pts.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)
    eye = torch.eye(N, device=canonical_pts.device).unsqueeze(0)
    dist_sq = dist_sq + eye * 1e9
    _, idx = dist_sq.topk(k_neighbors, dim=-1, largest=False)
    idx_exp = idx.unsqueeze(-1).expand(B, N, k_neighbors, 3)
    disp_exp = displacements.unsqueeze(2).expand(B, N, N, 3)
    neighbor_disp = disp_exp.gather(2, idx_exp)
    lap = displacements - neighbor_disp.mean(dim=2)
    return (lap ** 2).sum(dim=-1).mean()


def magnitude_loss(displacements: torch.Tensor) -> torch.Tensor:
    return (displacements ** 2).sum(dim=-1).mean()


class DeformationLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        lc = cfg.train.loss
        self.w_correspondence = lc.correspondence_weight
        self.w_contrastive = lc.contrastive_weight
        self.w_smooth = lc.smoothness_weight
        self.w_magnitude = lc.magnitude_weight
        self.occluded_weight = lc.get("occluded_weight", 0.1)
        self.temperature = lc.get("temperature", 0.1)
        self.validity_threshold = lc.get("validity_threshold", 0.15)

    def forward(
        self,
        # Decoder outputs
        deformed_canonical: torch.Tensor,     # (B, N_c, 3)
        displacements: torch.Tensor,           # (B, N_c, 3)
        canonical_pts: torch.Tensor,           # (B, N_c, 3)
        # GT
        gt_deformed_canonical: torch.Tensor,  # (B, N_c, 3)
        visible_mask: torch.Tensor,            # (B, N_c) float
        # Cross-attention features for contrastive
        enriched_can_feat: torch.Tensor,      # (B, 128, D)
        obs_feat_sa2: torch.Tensor,           # (B, 128, D)
        can_xyz_sa2: torch.Tensor,            # (B, 128, 3)
        obs_xyz_sa2: torch.Tensor,            # (B, 128, 3)
        gt_deformed_can_sa2: torch.Tensor,    # (B, 128, 3) GT for SA2 pts
    ) -> dict[str, torch.Tensor]:

        # Primary: dense correspondence
        corr = correspondence_loss(
            deformed_canonical, gt_deformed_canonical,
            visible_mask, self.occluded_weight,
        )

        # Secondary: InfoNCE contrastive
        l_contrastive = infonce_loss(
            enriched_can_feat, obs_feat_sa2,
            can_xyz_sa2, gt_deformed_can_sa2, obs_xyz_sa2,
            temperature=self.temperature,
            validity_threshold=self.validity_threshold,
        )

        # Regularization
        l_smooth = smoothness_loss(canonical_pts, displacements)
        l_magnitude = magnitude_loss(displacements)

        total = (
            self.w_correspondence * corr["loss_correspondence"]
            + self.w_contrastive * l_contrastive
            + self.w_smooth * l_smooth
            + self.w_magnitude * l_magnitude
        )

        return {
            "loss": total,
            "loss_correspondence":  corr["loss_correspondence"].detach(),
            "loss_corr_visible":    corr["loss_corr_visible"],
            "loss_corr_occluded":   corr["loss_corr_occluded"],
            "loss_contrastive":     l_contrastive.detach(),
            "loss_smooth":          l_smooth.detach(),
            "loss_magnitude":       l_magnitude.detach(),
        }
