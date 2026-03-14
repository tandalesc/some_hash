"""Learned Jacobian network for MD5.

A small transformer that predicts MD5's local sensitivity: how each input byte
affects each hash byte at a given operating point.

Supports optional auxiliary features:
- STE Jacobian: approximate Jacobian from differentiable soft MD5 (64x16 matrix)
- Intermediate states: MD5 internal state snapshots at various rounds (Nx16 bytes)

These give the network a starting point — it learns the residual correction
rather than predicting from scratch.
"""

import torch
import torch.nn as nn


class JacobianTransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

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
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class JacobianNet(nn.Module):
    """Predicts local Jacobian of MD5: how each input byte affects each hash byte.

    Args:
        d_model: transformer hidden dimension
        n_heads: attention heads
        n_layers: transformer layers
        dropout: dropout rate
        use_ste_features: if True, accepts STE Jacobian as additional input
        use_intermediate_features: if True, accepts soft MD5 intermediate states
        n_snapshots: number of intermediate state snapshots (when using intermediates)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.0,
                 use_ste_features: bool = False,
                 use_intermediate_features: bool = False,
                 n_snapshots: int = 7):
        super().__init__()
        self.d_model = d_model
        self.use_ste_features = use_ste_features
        self.use_intermediate_features = use_intermediate_features

        # Byte embedding for the 64 input bytes (values 0-255)
        self.byte_embed = nn.Embedding(256, d_model)
        # Position embedding for 64 positions
        self.pos_embed = nn.Embedding(64, d_model)

        # Optional: project STE Jacobian row (16 dims) into d_model
        if use_ste_features:
            self.ste_proj = nn.Linear(16, d_model)

        # Optional: project intermediate state features into d_model
        # Each snapshot is 16 bytes of state; we project all snapshots together
        if use_intermediate_features:
            self.intermediate_proj = nn.Linear(n_snapshots * 16, d_model)

        # Transformer layers
        self.blocks = nn.ModuleList([
            JacobianTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: per position, predict sensitivity to each of 16 hash bytes
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 16)

    def forward(
        self,
        x: torch.Tensor,
        ste_jacobian: torch.Tensor | None = None,
        intermediates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 64) long tensor of byte values [0, 255]
            ste_jacobian: (B, 64, 16) approximate Jacobian from STE (optional)
            intermediates: (B, N, 16) intermediate state snapshots (optional)
        Returns:
            (B, 64, 16) Jacobian matrix
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device)

        h = self.byte_embed(x) + self.pos_embed(positions)

        # Add STE Jacobian features if provided
        if self.use_ste_features and ste_jacobian is not None:
            h = h + self.ste_proj(ste_jacobian)

        # Add intermediate state features if provided
        # Broadcast: intermediates are global (not per-position), so we
        # flatten and project to d_model, then add to all positions
        if self.use_intermediate_features and intermediates is not None:
            # intermediates: (B, N, 16) -> (B, N*16)
            flat = intermediates.reshape(B, -1).float()
            global_feat = self.intermediate_proj(flat)  # (B, d_model)
            h = h + global_feat.unsqueeze(1)  # broadcast to all positions

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)
        J = self.out_proj(h)  # (B, 64, 16)

        # If STE features provided, predict residual on top of STE
        if self.use_ste_features and ste_jacobian is not None:
            J = ste_jacobian + J

        return J
