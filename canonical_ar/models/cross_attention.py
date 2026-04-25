"""
Cross-Attention with Relative Positional Bias

Allows canonical point features to attend to observed point features
while being aware of their relative 3D spatial positions.

Key idea: attention score between canonical point i and observed point j is:

    score(i, j) = dot(q_i, k_j) / sqrt(d) + relative_pos_bias(xyz_can_i - xyz_obs_j)

The relative positional bias is a small MLP that maps the 3D offset vector
to a scalar. This encodes:
  - Nearby points should attend to each other more strongly
  - The direction of offset matters (a point on top attends differently
    to points below vs above)
  - The network learns this spatial prior from data

After cross-attention, each canonical point's feature is enriched with
information about which observed region it most likely corresponds to.
We then do one more SA layer to globally pool this into z.

Why this fixes the rigid case:
  Without cross-attention, the encoder produces two independent global
  codes and hopes the fusion MLP can figure out the transform. With
  cross-attention, each canonical point explicitly finds its corresponding
  observed region before global pooling — so the rigid transform is
  directly encoded in the attention pattern rather than having to be
  inferred from opaque global summaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionBias(nn.Module):
    """
    Maps a 3D relative offset (xyz_canonical - xyz_observed) to a scalar
    attention bias. Learned MLP so the network decides how much spatial
    proximity matters and in what directions.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        xyz_can: torch.Tensor,   # (B, N, 3)
        xyz_obs: torch.Tensor,   # (B, M, 3)
    ) -> torch.Tensor:
        """
        Returns bias: (B, N, M) — one scalar per (canonical, observed) pair.
        """
        # Relative offsets: (B, N, M, 3)
        rel = xyz_can.unsqueeze(2) - xyz_obs.unsqueeze(1)
        B, N, M, _ = rel.shape
        # MLP over last dim
        bias = self.mlp(rel.reshape(B * N * M, 3))   # (B*N*M, 1)
        return bias.reshape(B, N, M)                  # (B, N, M)


class CrossAttentionLayer(nn.Module):
    """
    Multi-head cross-attention: canonical features (queries) attend to
    observed features (keys/values), with relative positional bias.

    Args:
        feat_dim:   dimension of input features (must match SA2 output)
        num_heads:  number of attention heads
        dropout:    attention dropout for regularization
    """
    def __init__(
        self,
        feat_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        pos_bias_hidden: int = 64,
    ):
        super().__init__()
        assert feat_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Query, key, value projections
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)

        # Relative positional bias (shared across heads)
        self.pos_bias = RelativePositionBias(hidden_dim=pos_bias_hidden)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_k = nn.LayerNorm(feat_dim)

        # Feed-forward after attention (standard transformer block)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.norm_out = nn.LayerNorm(feat_dim)
        self.norm_ffn = nn.LayerNorm(feat_dim)

    def forward(
        self,
        can_feat: torch.Tensor,    # (B, N, feat_dim) canonical features
        obs_feat: torch.Tensor,    # (B, M, feat_dim) observed features
        can_xyz: torch.Tensor,     # (B, N, 3) canonical positions
        obs_xyz: torch.Tensor,     # (B, M, 3) observed positions
    ) -> torch.Tensor:
        """
        Returns attended canonical features: (B, N, feat_dim)
        """
        B, N, D = can_feat.shape
        M = obs_feat.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        # Normalize inputs
        q_in = self.norm_q(can_feat)
        k_in = self.norm_k(obs_feat)

        # Project to Q, K, V
        Q = self.q_proj(q_in).reshape(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.k_proj(k_in).reshape(B, M, H, Dh).transpose(1, 2)  # (B, H, M, Dh)
        V = self.v_proj(k_in).reshape(B, M, H, Dh).transpose(1, 2)  # (B, H, M, Dh)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale     # (B, H, N, M)

        # Add relative positional bias (same across all heads)
        pos_bias = self.pos_bias(can_xyz, obs_xyz)                    # (B, N, M)
        attn = attn + pos_bias.unsqueeze(1)                           # broadcast over heads

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)                                   # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, D)                    # (B, N, D)
        out = self.out_proj(out)

        # Residual + norm
        can_feat = self.norm_out(can_feat + out)

        # Feed-forward with residual
        can_feat = self.norm_ffn(can_feat + self.ffn(can_feat))

        return can_feat                                                # (B, N, feat_dim)
