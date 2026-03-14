"""Learned Jacobian network for MD5.

A small transformer that predicts MD5's local sensitivity: how each input byte
affects each hash byte at a given operating point.
"""

import torch
import torch.nn as nn


class JacobianTransformerBlock(nn.Module):
    """Standard pre-norm transformer block (no conditioning signal)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h

        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class JacobianNet(nn.Module):
    """Predicts local Jacobian of MD5: how each input byte affects each hash byte."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Byte embedding for the 64 input bytes (values 0-255)
        self.byte_embed = nn.Embedding(256, d_model)
        # Position embedding for 64 positions
        self.pos_embed = nn.Embedding(64, d_model)

        # Transformer layers
        self.blocks = nn.ModuleList([
            JacobianTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: per position, predict sensitivity to each of 16 hash bytes
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 64) long tensor of byte values [0, 255]
        Returns:
            (B, 64, 16) Jacobian matrix — sensitivity of each hash byte
                         to each input byte at this operating point
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device)

        h = self.byte_embed(x) + self.pos_embed(positions)

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)
        return self.out_proj(h)
