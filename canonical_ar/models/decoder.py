"""
Deformation field decoder — v3.

Now accepts local correspondence features alongside the global z.
For each query point p_i, the decoder receives:
  - pe(p_i): positional encoding of canonical position
  - z: global deformation context (FiLM conditioning)
  - local_feat_i: cross-attention feature of nearest SA2 point
                  encodes "where this region went in the observation"

local_feat is added to the input before the FiLM layers,
giving the decoder point-specific context while z handles global conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from canonical_ar.models.utils import PositionalEncoding


class FiLMLayer(nn.Module):
    """FiLM conditioning: y = gamma(z) * x + beta(z)"""
    def __init__(self, feature_dim: int, code_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(code_dim, feature_dim)
        self.beta_proj = nn.Linear(code_dim, feature_dim)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(z)
        beta = self.beta_proj(z)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class DeformationDecoder(nn.Module):
    """
    FiLM-conditioned MLP decoder with local feature input.

    Architecture:
        concat(pe(query_xyz), local_feat) → input_proj → [FiLM layer] x N → output_head → displacement
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 4,
        pe_freqs: int = 6,
        local_feat_dim: int = 256,
    ):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=pe_freqs, include_input=True)
        pe_dim = self.pe.output_dim  # 3 + 3*2*pe_freqs

        # Input: positional encoding + local feature
        input_dim = pe_dim + local_feat_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.films = nn.ModuleList([
            FiLMLayer(hidden_dim, latent_dim) for _ in range(n_layers)
        ])
        self.acts = nn.ModuleList([nn.GELU() for _ in range(n_layers)])

        self.output_head = nn.Linear(hidden_dim, 3)
        nn.init.normal_(self.output_head.weight, std=1e-4)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        canonical_pts: torch.Tensor,   # (B, Q, 3)
        z: torch.Tensor,               # (B, latent_dim)
        local_feat: torch.Tensor,      # (B, Q, local_feat_dim)
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            canonical_pts: query points in canonical space
            z: global latent code
            local_feat: per-point local correspondence features

        Returns dict with displacements and deformed_pts.
        """
        # Positional encode query points
        pe = self.pe(canonical_pts)           # (B, Q, pe_dim)

        # Concatenate with local feature
        x = torch.cat([pe, local_feat], dim=-1)  # (B, Q, pe_dim + local_feat_dim)
        x = self.input_proj(x)                    # (B, Q, hidden_dim)

        # FiLM-conditioned layers with residuals
        for linear, norm, film, act in zip(
            self.layers, self.norms, self.films, self.acts
        ):
            residual = x
            x = linear(x)
            x = norm(x)
            x = film(x, z)
            x = act(x)
            x = x + residual

        displacements = self.output_head(x)         # (B, Q, 3)
        deformed_pts = canonical_pts + displacements

        return {
            "displacements": displacements,
            "deformed_pts":  deformed_pts,
        }
