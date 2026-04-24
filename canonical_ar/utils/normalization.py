"""
Point cloud normalization.

All point clouds must be normalized before entering the encoder.
Without this, objects at different scales and positions produce
wildly different activations, making training unstable and
generalization across objects impossible.

Convention:
  - Canonical space: centered at origin, scaled to fit unit sphere
  - We track (centroid, scale) per point cloud so we can invert
    the normalization when placing virtual content back in world space

Important: canonical and observed point clouds are normalized
INDEPENDENTLY in terms of centering, but share the same scale factor
(derived from the canonical cloud). This ensures the deformation field
lives in a consistent scale regardless of how the observed cloud drifts.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class NormalizationParams:
    """Stores the transform needed to invert normalization."""
    centroid: torch.Tensor   # (3,) or (B, 3)
    scale: torch.Tensor      # scalar or (B,)

    def to(self, device):
        return NormalizationParams(
            centroid=self.centroid.to(device),
            scale=self.scale.to(device),
        )


def normalize_point_cloud(
    pts: torch.Tensor,
    ref_pts: torch.Tensor = None,
) -> tuple[torch.Tensor, NormalizationParams]:
    """
    Center and scale a point cloud to fit within the unit sphere.

    Args:
        pts:     (B, N, 3) or (N, 3) — points to normalize
        ref_pts: (B, N, 3) or (N, 3) — if given, derive scale from this
                 cloud instead of pts. Use this to normalize observed
                 point clouds with canonical scale.

    Returns:
        normalized_pts: same shape as pts
        params: NormalizationParams for inversion
    """
    squeeze = pts.dim() == 2
    if squeeze:
        pts = pts.unsqueeze(0)
        if ref_pts is not None:
            ref_pts = ref_pts.unsqueeze(0)

    # Centroid from pts itself (we always center on the observed/canonical cloud)
    centroid = pts.mean(dim=1, keepdim=True)   # (B, 1, 3)
    pts_centered = pts - centroid

    # Scale from ref_pts if provided, else from pts
    scale_src = ref_pts - ref_pts.mean(dim=1, keepdim=True) if ref_pts is not None else pts_centered
    # Furthest point distance from origin
    scale = scale_src.norm(dim=-1).max(dim=-1).values  # (B,)
    scale = scale.clamp(min=1e-6)

    normalized = pts_centered / scale.unsqueeze(-1).unsqueeze(-1)

    params = NormalizationParams(
        centroid=centroid.squeeze(1),   # (B, 3)
        scale=scale,                    # (B,)
    )

    if squeeze:
        normalized = normalized.squeeze(0)
        params = NormalizationParams(
            centroid=params.centroid.squeeze(0),
            scale=params.scale.squeeze(0),
        )

    return normalized, params


def denormalize_points(
    pts: torch.Tensor,
    params: NormalizationParams,
) -> torch.Tensor:
    """
    Invert normalization: bring points back to original world scale/position.

    Args:
        pts:    (B, Q, 3) normalized points
        params: NormalizationParams from normalize_point_cloud

    Returns:
        world_pts: (B, Q, 3)
    """
    scale = params.scale.unsqueeze(-1).unsqueeze(-1)    # (B, 1, 1)
    centroid = params.centroid.unsqueeze(1)              # (B, 1, 3)
    return pts * scale + centroid


def normalize_batch(batch: dict[str, torch.Tensor]) -> tuple[dict, NormalizationParams]:
    """
    Normalize a full training batch in-place.
    Canonical cloud sets the reference scale.
    Observed cloud is centered independently but uses canonical scale.

    Returns normalized batch and params (needed at inference to recover world coords).
    """
    # Normalize canonical cloud — this sets the reference scale
    can_norm, params = normalize_point_cloud(batch["canonical_xyz"])

    # Normalize observed cloud: center on itself, but scale with canonical scale
    obs_centroid = batch["obs_xyz"].mean(dim=1, keepdim=True)
    obs_centered = batch["obs_xyz"] - obs_centroid
    obs_norm = obs_centered / params.scale.unsqueeze(-1).unsqueeze(-1)

    # Query points live in canonical space — normalize with canonical params
    query_norm = (batch["query_pts"] - params.centroid.unsqueeze(1)) / params.scale.unsqueeze(-1).unsqueeze(-1)
    gt_query_norm = (batch["gt_deformed_query"] - params.centroid.unsqueeze(1)) / params.scale.unsqueeze(-1).unsqueeze(-1)

    return {
        **batch,
        "canonical_xyz": can_norm,
        "obs_xyz": obs_norm,
        "query_pts": query_norm,
        "gt_deformed_query": gt_query_norm,
        # Store obs centroid separately for potential inverse use
        "_obs_centroid": obs_centroid.squeeze(1),
    }, params


# ── Numpy versions for use in data generation ─────────────────────────────────

def normalize_np(pts: np.ndarray, ref_pts: np.ndarray = None):
    """
    Numpy version for use inside the Dataset __getitem__.
    Args:
        pts:     (N, 3)
        ref_pts: (N, 3) optional reference for scale
    Returns:
        normalized: (N, 3)
        centroid: (3,)
        scale: float
    """
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    src = (ref_pts - ref_pts.mean(axis=0)) if ref_pts is not None else pts_centered
    scale = float(np.linalg.norm(src, axis=1).max())
    scale = max(scale, 1e-6)
    return (pts_centered / scale).astype(np.float32), centroid.astype(np.float32), scale
