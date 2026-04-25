"""
DeformationFieldNet v2 — with cross-attention fusion.

Architecture change from v1:
  OLD: canonical → global code
       observed  → global code
       concat → MLP → z

  NEW: canonical → SA1 → SA2 → per-point features (128, 256)
       observed  → SA1 → SA2 → per-point features (128, 256)
                                        ↓
               cross-attention with relative positional bias
               (canonical queries attend to observed keys/values)
                                        ↓
               enriched canonical features (128, 256)
                                        ↓
               SA3 → global pool → z (256)

The decoder is unchanged — same FiLM MLP, same interface.
Rigid objects are handled by the cross-attention finding the
corresponding observed region for each canonical point, making
the rigid transform directly readable from the attention pattern
rather than having to be inferred from opaque global codes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from canonical_ar.models.encoder import PointNetPlusPlus
from canonical_ar.models.decoder import DeformationDecoder
from canonical_ar.models.cross_attention import CrossAttentionLayer


class DeformationFieldNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        enc_cfg = cfg.model.encoder
        dec_cfg = cfg.model.decoder
        fusion_cfg = cfg.model.fusion

        feat_dim = 256   # SA2 output dim — hardcoded to match encoder SA2

        # Two encoders: separate weights, shared architecture
        self.canonical_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=fusion_cfg.output_dim,
        )
        self.obs_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=fusion_cfg.output_dim,
        )

        # Cross-attention: canonical attends to observed at SA2 resolution
        self.cross_attn = CrossAttentionLayer(
            feat_dim=feat_dim,
            num_heads=4,
            dropout=0.1,
            pos_bias_hidden=64,
        )

        # After cross-attention, pool enriched canonical features to z
        # using the canonical encoder's SA3 + global_proj
        # (obs encoder's SA3 is not used in the new pipeline)

        # Deformation field decoder — unchanged
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
        Cross-attention encoding pipeline.

        1. Both point clouds go through SA1+SA2 to get local features
        2. Canonical features attend to observed features with pos bias
        3. Enriched canonical features are pooled to global z via SA3
        """
        # Step 1: extract local features at 128-point resolution
        can_xyz_sa2, can_feat_sa2 = self.canonical_encoder.forward_local(
            canonical_xyz, canonical_feat
        )                                         # (B, 128, 3), (B, 128, 256)

        obs_xyz_sa2, obs_feat_sa2 = self.obs_encoder.forward_local(
            obs_xyz, obs_feat
        )                                         # (B, 128, 3), (B, 128, 256)
        # Note: obs has fewer input points (1024 vs 2048) so SA layers
        # naturally produce a coarser but still 128-point representation

        # Step 2: cross-attention — canonical queries observed
        enriched_can_feat = self.cross_attn(
            can_feat=can_feat_sa2,
            obs_feat=obs_feat_sa2,
            can_xyz=can_xyz_sa2,
            obs_xyz=obs_xyz_sa2,
        )                                         # (B, 128, 256)

        # Step 3: pool enriched features to global z
        z = self.canonical_encoder.pool_from_local(can_xyz_sa2, enriched_can_feat)
        # (B, output_dim)

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
        Full forward pass. Interface identical to v1.

        Args:
            canonical_xyz:  (B, N_c, 3)
            canonical_feat: (B, N_c, C)
            obs_xyz:        (B, N_o, 3)
            obs_feat:       (B, N_o, C)
            query_pts:      (B, Q, 3)

        Returns dict:
            z:              (B, fusion_dim)
            displacements:  (B, Q, 3)
            deformed_pts:   (B, Q, 3)
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
        """Runtime inference — returns deformed_pts (B, Q, 3)."""
        self.eval()
        out = self.forward(
            canonical_xyz, canonical_feat,
            obs_xyz, obs_feat,
            query_pts,
        )
        return out["deformed_pts"]
