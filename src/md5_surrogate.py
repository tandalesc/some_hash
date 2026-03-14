"""Differentiable MD5 surrogate — learns the forward pass round by round.

Two representation modes:
- Byte-level: state = 16 floats (4 words × 4 bytes). Compact but the network
  must learn bit decomposition internally.
- Bit-level: state = 128 floats (4 words × 32 bits). Larger but boolean ops
  (AND, XOR, OR) are trivially learnable on individual bits.

Training modes:
- Teacher forcing: each round gets the real intermediate state (fast convergence)
- Autoregressive: chain rounds, backprop through full chain (fixes error accumulation)
- Scheduled rollout: gradually transition from teacher forcing to autoregressive
"""

import torch
import torch.nn as nn

from .md5_gpu import _SHIFT


class BitRoundLayer(nn.Module):
    """Learns one MD5 round at bit level.

    Input: 128 state bits + 32 message word bits + 2 round info = 162
    Output: 128 state bits
    """

    def __init__(self, d_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(162, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 128),
        )

    def forward(self, state: torch.Tensor, msg_word: torch.Tensor,
                round_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 128) float — 4 words as 128 bits
            msg_word: (B, 32) float — 1 word as 32 bits
            round_info: (B, 2) float — [round_index/64, shift/25]
        Returns:
            (B, 128) float — predicted next state bits
        """
        x = torch.cat([state, msg_word, round_info], dim=-1)
        return self.net(x)


class ByteRoundLayer(nn.Module):
    """Learns one MD5 round at byte level.

    Input: 16 state bytes + 4 message word bytes + 2 round info = 22
    Output: 16 state bytes
    """

    def __init__(self, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(22, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 16),
        )

    def forward(self, state: torch.Tensor, msg_word: torch.Tensor,
                round_info: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, msg_word, round_info], dim=-1)
        return self.net(x)


# Alias for backward compat
RoundLayer = ByteRoundLayer


class MD5Surrogate(nn.Module):
    """Differentiable MD5 surrogate with per-round learned layers.

    Args:
        num_rounds: MD5 rounds to model (1-64)
        d_hidden: hidden dim for each round's MLP
        shared_weights: if True, all rounds share the same MLP
        bit_level: if True, use 128-bit state representation instead of 16-byte
    """

    def __init__(self, num_rounds: int = 64, d_hidden: int = 256,
                 shared_weights: bool = False, bit_level: bool = False):
        super().__init__()
        self.num_rounds = num_rounds
        self.bit_level = bit_level
        self.state_dim = 128 if bit_level else 16
        self.word_dim = 32 if bit_level else 4

        LayerClass = BitRoundLayer if bit_level else ByteRoundLayer

        if shared_weights:
            layer = LayerClass(d_hidden)
            self.rounds = nn.ModuleList([layer] * num_rounds)
        else:
            self.rounds = nn.ModuleList([
                LayerClass(d_hidden) for _ in range(num_rounds)
            ])

        # MD5 message schedule
        self._schedule = []
        for i in range(64):
            if i < 16:
                self._schedule.append(i)
            elif i < 32:
                self._schedule.append((5 * i + 1) % 16)
            elif i < 48:
                self._schedule.append((3 * i + 5) % 16)
            else:
                self._schedule.append((7 * i) % 16)

    def _get_initial_state(self, B: int, device: torch.device) -> torch.Tensor:
        """MD5 initial state as normalized float tensor."""
        if self.bit_level:
            return _words_to_bits_batch(
                torch.tensor([[0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]],
                             dtype=torch.int64, device=device).expand(B, -1)
            )
        else:
            init_bytes = [
                0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
            ]
            return torch.tensor(
                init_bytes, dtype=torch.float32, device=device
            ).unsqueeze(0).expand(B, -1) / 255.0

    def _prepare_message(self, message: torch.Tensor) -> torch.Tensor:
        """Convert message to word chunks in the right representation.

        Args:
            message: (B, 64) float — message bytes normalized to [0,1]
        Returns:
            (B, 16, word_dim) — 16 message words
        """
        B = message.shape[0]
        if self.bit_level:
            # Convert each byte to 8 bits, group into 32-bit words
            # message: (B, 64) bytes → (B, 64, 8) bits → (B, 512) → (B, 16, 32)
            msg_bytes_int = (message * 255.0).round().long()
            bits = _bytes_to_bits_flat(msg_bytes_int)  # (B, 512)
            return bits.reshape(B, 16, 32)
        else:
            return message.reshape(B, 16, 4)

    def forward(self, message_bytes: torch.Tensor,
                initial_state: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive forward pass (no teacher forcing)."""
        B = message_bytes.shape[0]
        device = message_bytes.device

        state = initial_state if initial_state is not None else self._get_initial_state(B, device)
        msg_words = self._prepare_message(message_bytes)

        for i in range(self.num_rounds):
            g = self._schedule[i]
            word = msg_words[:, g, :]
            round_info = torch.tensor(
                [i / 64.0, _SHIFT[i] / 25.0],
                device=device, dtype=torch.float32
            ).unsqueeze(0).expand(B, -1)
            state = self.rounds[i](state, word, round_info)

        return state


# --- Conversion utilities for bit-level representation ---

def _words_to_bits_batch(words: torch.Tensor) -> torch.Tensor:
    """Convert (B, 4) int64 word state to (B, 128) float bits."""
    bits = []
    for w_idx in range(4):
        w = words[:, w_idx]
        for bit in range(32):
            bits.append(((w >> bit) & 1).float())
    return torch.stack(bits, dim=1)


def _bytes_to_bits_flat(byte_tensor: torch.Tensor) -> torch.Tensor:
    """Convert (B, N) int64 byte tensor to (B, N*8) float bit tensor."""
    B, N = byte_tensor.shape
    bits = []
    for bit in range(8):
        bits.append(((byte_tensor >> bit) & 1).float())
    # bits is 8 tensors of (B, N), interleave to (B, N, 8) → (B, N*8)
    return torch.stack(bits, dim=-1).reshape(B, N * 8)


def words_to_state_bytes(states: torch.Tensor) -> torch.Tensor:
    """Convert (B, 4) int64 word states to (B, 16) float byte states normalized to [0,1]."""
    result = []
    for i in range(4):
        w = states[:, i]
        result.append((w & 0xFF).float())
        result.append(((w >> 8) & 0xFF).float())
        result.append(((w >> 16) & 0xFF).float())
        result.append(((w >> 24) & 0xFF).float())
    return torch.stack(result, dim=1) / 255.0


def words_to_state_bits(states: torch.Tensor) -> torch.Tensor:
    """Convert (B, 4) int64 word states to (B, 128) float bit states."""
    return _words_to_bits_batch(states)
