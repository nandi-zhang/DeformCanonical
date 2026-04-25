"""
DeformationFieldNet v3 — local correspondence-aware decoder.

Key change from v2:
  The decoder now receives per-point local features from the cross-attention,
  not just the global z. For each canonical point p_i, the decoder gets:
    - p_i: canonical position
    - z: global deformation context (from pooled enriched features)
    - local_feat_i: cross-attention feature of the nearest SA2 point to p_i
                    This encodes "p_i's region corresponds to this part
                    of the observation"

This makes the model truly local — it can predict spatially varying
deformation fields where different regions deform differently, because
each point has its own local correspondence feature guiding the prediction.

The encode() method now returns intermediate features needed for:
  1. Local feature lookup for the decoder
  2. InfoNCE contrastive loss computation in train.py
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

        self.sa2_feat_dim = 256  # SA2 output dim

        self.canonical_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=fusion_cfg.output_dim,
        )
        self.obs_encoder = PointNetPlusPlus(
            input_dim=enc_cfg.input_dim,
            output_dim=fusion_cfg.output_dim,
        )

        self.cross_attn = CrossAttentionLayer(
            feat_dim=self.sa2_feat_dim,
            num_heads=4,
            dropout=0.1,
            pos_bias_hidden=64,
        )

        # Decoder now takes local_feat concatenated with positional encoding
        # local_feat is projected to match the decoder's hidden dim
        self.local_proj = nn.Linear(self.sa2_feat_dim, dec_cfg.hidden_dims[0])

        self.decoder = DeformationDecoder(
            latent_dim=fusion_cfg.output_dim,
            hidden_dim=dec_cfg.hidden_dims[0],
            n_layers=len(dec_cfg.hidden_dims),
            pe_freqs=dec_cfg.positional_encoding_freqs,
            local_feat_dim=dec_cfg.hidden_dims[0],  # projected local feat dim
        )

    def encode(
        self,
        canonical_xyz: torch.Tensor,   # (B, N_c, 3)
        canonical_feat: torch.Tensor,  # (B, N_c, C)
        obs_xyz: torch.Tensor,         # (B, N_o, 3)
        obs_feat: torch.Tensor,        # (B, N_o, C)
    ) -> dict[str, torch.Tensor]:
        """
        Full encoding pipeline.
        Returns all intermediate features needed for decoder and loss.
        """
        # SA1 + SA2 for both clouds
        can_xyz_sa2, can_feat_sa2 = self.canonical_encoder.forward_local(
            canonical_xyz, canonical_feat
        )  # (B, 128, 3), (B, 128, 256)

        obs_xyz_sa2, obs_feat_sa2 = self.obs_encoder.forward_local(
            obs_xyz, obs_feat
        )  # (B, 128, 3), (B, 128, 256)

        # Cross-attention: canonical attends to observed
        enriched_can_feat = self.cross_attn(
            can_feat=can_feat_sa2,
            obs_feat=obs_feat_sa2,
            can_xyz=can_xyz_sa2,
            obs_xyz=obs_xyz_sa2,
        )  # (B, 128, 256)

        # Pool enriched features to global z
        z = self.canonical_encoder.pool_from_local(can_xyz_sa2, enriched_can_feat)
        # (B, output_dim)

        return {
            "z": z,
            "can_xyz_sa2": can_xyz_sa2,          # (B, 128, 3)
            "enriched_can_feat": enriched_can_feat,  # (B, 128, 256)
            "obs_xyz_sa2": obs_xyz_sa2,            # (B, 128, 3)
            "obs_feat_sa2": obs_feat_sa2,          # (B, 128, 256)
        }

    def get_local_features(
        self,
        query_pts: torch.Tensor,       # (B, Q, 3) canonical coords
        can_xyz_sa2: torch.Tensor,     # (B, 128, 3) SA2 canonical positions
        enriched_can_feat: torch.Tensor,  # (B, 128, 256) enriched features
    ) -> torch.Tensor:
        """
        For each query point, find its nearest SA2 canonical point
        and return that point's enriched feature.

        This gives each query point local correspondence information:
        "my SA2 neighborhood corresponds to this part of the observation."

        Returns: (B, Q, hidden_dim) projected local features
        """
        # Pairwise distances: (B, Q, 128)
        diff = query_pts.unsqueeze(2) - can_xyz_sa2.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1)
        # Nearest SA2 point for each query point
        nn_idx = dist.argmin(dim=-1)  # (B, Q)

        # Gather enriched features
        B, Q = nn_idx.shape
        D = enriched_can_feat.shape[-1]
        idx_exp = nn_idx.unsqueeze(-1).expand(B, Q, D)
        local_feat = enriched_can_feat.gather(1, idx_exp)  # (B, Q, 256)

        # Project to decoder hidden dim
        return self.local_proj(local_feat)  # (B, Q, hidden_dim)

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        canonical_feat: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_feat: torch.Tensor,
        query_pts: torch.Tensor,       # (B, Q, 3) virtual content positions
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            deformed_pts:       (B, Q, 3) virtual content in observation space
            displacements:      (B, Q, 3) displacement vectors
            z:                  (B, D) global latent
            + all intermediate features from encode()
        """
        enc = self.encode(canonical_xyz, canonical_feat, obs_xyz, obs_feat)
        z = enc["z"]

        # Get local features for query points
        local_feat = self.get_local_features(
            query_pts, enc["can_xyz_sa2"], enc["enriched_can_feat"]
        )  # (B, Q, hidden_dim)

        decoder_out = self.decoder(query_pts, z, local_feat)

        return {
            "deformed_pts":      decoder_out["deformed_pts"],
            "displacements":     decoder_out["displacements"],
            "z":                 z,
            "can_xyz_sa2":       enc["can_xyz_sa2"],
            "enriched_can_feat": enc["enriched_can_feat"],
            "obs_xyz_sa2":       enc["obs_xyz_sa2"],
            "obs_feat_sa2":      enc["obs_feat_sa2"],
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
            canonical_xyz, canonical_feat, obs_xyz, obs_feat, query_pts
        )
        return out["deformed_pts"]
