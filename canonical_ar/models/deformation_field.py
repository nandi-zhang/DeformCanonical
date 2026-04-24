"""
DeformationFieldNet: the top-level model.

Takes:
  - canonical point cloud (from pre-computed Gaussian splat / scan)
  - observed point cloud (from RGB-D at runtime)
  - query points in canonical space (virtual content positions)

Produces:
  - deformed positions of query points in observation space
  - deformation field displacements

The key design principle: rigid objects are a degenerate case.
A bottle produces near-zero displacements. A pillow produces
spatially varying displacements. Same model, same weights.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from canonical_ar.models.encoder import PointNetPlusPlus
from canonical_ar.models.decoder import DeformationDecoder


class DeformationFieldNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        enc_cfg = cfg.model.encoder
        dec_cfg = cfg.model.decoder
        fusion_cfg = cfg.model.fusion

        latent_dim = enc_cfg.output_dim

        # Two encoders: one for the canonical shape, one for the observation.
        # Shared architecture, separate weights — canonical and observed
        # point clouds have different distributions (canonical is clean,
        # observed is partial/noisy).
        self.canonical_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=latent_dim,
        )
        self.obs_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=latent_dim,
        )

        # Fusion: concat both codes and project to a single latent
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, fusion_cfg.output_dim),
            nn.LayerNorm(fusion_cfg.output_dim),
            nn.GELU(),
            nn.Linear(fusion_cfg.output_dim, fusion_cfg.output_dim),
        )

        # Deformation field decoder
        self.decoder = DeformationDecoder(
            latent_dim=fusion_cfg.output_dim,
            hidden_dim=dec_cfg.hidden_dims[0],
            n_layers=len(dec_cfg.hidden_dims),
            pe_freqs=dec_cfg.positional_encoding_freqs,
        )

    def encode(
        self,
        canonical_xyz: torch.Tensor,
        canonical_feat: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode both point clouds into a single fused latent code.
        Args:
            canonical_xyz: (B, N_c, 3)
            canonical_feat: (B, N_c, C)
            obs_xyz: (B, N_o, 3)
            obs_feat: (B, N_o, C)
        Returns:
            z: (B, fusion_dim)
        """
        z_canonical = self.canonical_encoder(canonical_xyz, canonical_feat)
        z_obs = self.obs_encoder(obs_xyz, obs_feat)
        z = self.fusion(torch.cat([z_canonical, z_obs], dim=-1))
        return z

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        canonical_feat: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_feat: torch.Tensor,
        query_pts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            canonical_xyz:  (B, N_c, 3) canonical point positions
            canonical_feat: (B, N_c, C) canonical point features (normals etc.)
            obs_xyz:        (B, N_o, 3) observed point positions
            obs_feat:       (B, N_o, C) observed point features
            query_pts:      (B, Q, 3)   virtual content points in canonical space

        Returns dict:
            z:              (B, fusion_dim)  fused latent code
            displacements:  (B, Q, 3)        per-query displacement
            deformed_pts:   (B, Q, 3)        query_pts + displacements
                                             = virtual content in observation space
        """
        z = self.encode(canonical_xyz, canonical_feat, obs_xyz, obs_feat)
        decoder_out = self.decoder(query_pts, z)

        return {
            "z": z,
            "displacements": decoder_out["displacements"],
            "deformed_pts": decoder_out["deformed_pts"],
        }

    @torch.no_grad()
    def infer(
        self,
        canonical_xyz: torch.Tensor,
        canonical_feat: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_feat: torch.Tensor,
        query_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for runtime inference.
        Returns only deformed_pts (B, Q, 3) — positions of virtual content
        in current observation space.
        """
        self.eval()
        out = self.forward(
            canonical_xyz, canonical_feat,
            obs_xyz, obs_feat,
            query_pts,
        )
        return out["deformed_pts"]
