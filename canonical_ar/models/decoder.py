"""
Deformation field decoder.

Given:
  - A fused latent code z (B, latent_dim) encoding the current observation
    relative to the canonical shape
  - Query points p in canonical space (B, Q, 3)

Outputs:
  - Displacement vectors delta (B, Q, 3)
  - Deformed positions p + delta (B, Q, 3)

The decoder is conditioned on z via FiLM (Feature-wise Linear Modulation),
which scales and shifts each hidden layer's activations. This is more
expressive than simple concatenation and is standard in conditional NeRF work.

For rigid objects, the network should learn to output a globally consistent
displacement (equivalent to a rigid transform). For deformable objects it
outputs a spatially varying field. The same network handles both — rigidity
is a special case with near-zero or globally consistent displacements.
"""

import torch
import torch.nn as nn
from canonical_ar.models.utils import PositionalEncoding


class FiLMLayer(nn.Module):
    """
    FiLM conditioning: y = gamma(z) * x + beta(z)
    where z is the conditioning code and x is the feature vector.
    """
    def __init__(self, feature_dim: int, code_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(code_dim, feature_dim)
        self.beta_proj = nn.Linear(code_dim, feature_dim)
        # Init gamma to 1, beta to 0 (identity at start)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Q, feature_dim) or (B, feature_dim)
            z: (B, code_dim)
        """
        gamma = self.gamma_proj(z)  # (B, feature_dim)
        beta = self.beta_proj(z)    # (B, feature_dim)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, feature_dim) -> broadcast over Q
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class DeformationDecoder(nn.Module):
    """
    MLP decoder with FiLM conditioning at every layer.

    Architecture:
        pos_enc(query_xyz) -> [Linear -> LayerNorm -> FiLM -> GELU] x N -> Linear -> displacement
    """
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 4,
        pe_freqs: int = 6,
    ):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=pe_freqs, include_input=True)
        input_dim = self.pe.output_dim  # 3 + 3*2*pe_freqs

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # FiLM-conditioned hidden layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.films = nn.ModuleList([
            FiLMLayer(hidden_dim, latent_dim) for _ in range(n_layers)
        ])
        self.acts = nn.ModuleList([
            nn.GELU() for _ in range(n_layers)
        ])

        # Output head: displacement prediction
        # Init near-zero so training starts close to identity (no deformation)
        self.output_head = nn.Linear(hidden_dim, 3)
        nn.init.normal_(self.output_head.weight, std=1e-4)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        canonical_pts: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            canonical_pts: (B, Q, 3) — query points in canonical space
            z: (B, latent_dim) — fused latent code

        Returns dict with:
            displacements: (B, Q, 3)
            deformed_pts: (B, Q, 3) — canonical_pts + displacements
        """
        # Positional encode query points
        x = self.pe(canonical_pts)          # (B, Q, pe_dim)
        x = self.input_proj(x)              # (B, Q, hidden_dim)

        # FiLM-conditioned layers
        for linear, norm, film, act in zip(
            self.layers, self.norms, self.films, self.acts
        ):
            residual = x
            x = linear(x)
            x = norm(x)
            x = film(x, z)
            x = act(x)
            x = x + residual  # residual connection

        displacements = self.output_head(x)         # (B, Q, 3)
        deformed_pts = canonical_pts + displacements

        return {
            "displacements": displacements,
            "deformed_pts": deformed_pts,
        }
