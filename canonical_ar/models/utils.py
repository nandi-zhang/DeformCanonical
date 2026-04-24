import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 3D points, following NeRF convention.
    Maps R^3 -> R^(3 * 2 * num_freqs + 3)
    """
    def __init__(self, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 3)
        Returns:
            encoded: (..., output_dim)
        """
        # x: (..., 3), freqs: (F,)
        # outer product: (..., 3, F)
        x_freq = x.unsqueeze(-1) * self.freqs  # (..., 3, F)
        x_freq = x_freq.reshape(*x.shape[:-1], -1)  # (..., 3*F)
        encoded = torch.cat([
            torch.sin(x_freq),
            torch.cos(x_freq),
        ], dim=-1)  # (..., 3*2*F)
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        return encoded


class MLP(nn.Module):
    """Generic MLP with LayerNorm and residual connections."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], output_dim)

        # Residual projection if input/output dims differ
        self.residual_proj = (
            nn.Linear(input_dim, dims[-1])
            if use_residual and input_dim != dims[-1]
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        if self.use_residual:
            h = h + self.residual_proj(x)
        return self.output(h)
