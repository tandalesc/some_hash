import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional / timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero: projects conditioning to (gamma, beta, gate)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (normed_and_scaled, gate). Caller applies gate after sublayer."""
        gamma, beta, gate = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(x) * (1 + gamma) + beta, gate.sigmoid()


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.adaln_attn = AdaLNZero(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.adaln_ffn = AdaLNZero(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Self-attention with adaLN-Zero
        h, gate = self.adaln_attn(x, cond)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate * h

        # FFN with adaLN-Zero
        h, gate = self.adaln_ffn(x, cond)
        h = self.ffn(h)
        x = x + gate * h
        return x


class HashConditioner(nn.Module):
    """Projects MD5 hash bytes + noise level into a conditioning vector."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.hash_embed = nn.Linear(cfg.hash_bytes, cfg.d_model)
        self.time_embed = SinusoidalEmbedding(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, hash_bytes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hash_bytes: (B, 16) long tensor of hash byte values [0, 255]
            t: (B,) float tensor of noise levels in [0, 1]
        Returns:
            (B, d_model) conditioning vector
        """
        h = self.hash_embed(hash_bytes.float() / 255.0)
        h = h + self.time_embed(t * 1000.0)  # scale t for richer frequencies
        return self.mlp(h)


class Denoiser(nn.Module):
    """Transformer denoiser for discrete byte-level diffusion."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.conditioner = HashConditioner(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, 256)  # predict over 256 byte values (no MASK)

    def forward(
        self,
        x: torch.Tensor,
        hash_bytes: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 64) long tensor — noisy/masked message bytes
            hash_bytes: (B, 16) long tensor — target MD5 hash
            t: (B,) float — mask ratio / noise level
        Returns:
            (B, 64, 256) logits over byte values per position
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device)
        h = self.token_embed(x) + self.pos_embed(positions)
        cond = self.conditioner(hash_bytes, t)

        for block in self.blocks:
            h = block(h, cond)

        h = self.out_norm(h)
        return self.out_proj(h)
