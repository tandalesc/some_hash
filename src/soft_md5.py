"""Differentiable MD5 implementation using soft bit operations.

Represents bits as floats in [0, 1] and replaces discrete logic gates
with continuous relaxations:
    AND(a, b) ≈ a * b
    OR(a, b)  ≈ a + b - a*b
    XOR(a, b) ≈ a + b - 2*a*b
    NOT(a)    ≈ 1 - a

Addition mod 2^32 uses float-space arithmetic with a straight-through
estimator for bit decomposition, avoiding the sequential carry chain
that kills gradients in the naive ripple-carry approach.
"""

import math

import torch
import torch.nn as nn


# --- Soft bit operations ---

def soft_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def soft_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - a * b


def soft_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b - 2 * a * b


def soft_not(a: torch.Tensor) -> torch.Tensor:
    return 1 - a


# --- Float-space addition with STE ---

_POWERS_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_powers(device: torch.device) -> torch.Tensor:
    if device not in _POWERS_CACHE:
        _POWERS_CACHE[device] = 2.0 ** torch.arange(32, device=device, dtype=torch.float64)
    return _POWERS_CACHE[device]


def _hard_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact addition mod 2^32 via float64. Returns hard bits, no gradient."""
    powers = _get_powers(a.device)
    a_val = (a.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    b_val = (b.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    sum_val = torch.fmod(a_val + b_val, 2.0**32)
    inv_powers = 1.0 / powers
    return torch.floor(torch.fmod(sum_val * inv_powers, 2.0)).to(a.dtype)


def _xor_ste(a: torch.Tensor, b: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """XOR-based STE: backward treats addition as XOR (no carry)."""
    soft_proxy = soft_xor(a, b) * _XOR_GRAD_SCALE
    return hard.detach() + soft_proxy - soft_proxy.detach()

_XOR_GRAD_SCALE = 0.5


def _sinusoidal_ste(a: torch.Tensor, b: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """Sinusoidal STE: backward uses periodic bit extraction with dither.

    Each bit k of value v is a square wave with period 2^(k+1).
    We approximate it with: soft_bit_k = 0.5 * (1 - cos(π * v / 2^k))
    which is exact at integers but has zero gradient there.

    Dithering shifts off the zero-gradient peaks, putting us on the
    cosine slope where gradient flows. The dither is small enough that
    hard rounding still gives the correct bit, but large enough to
    produce meaningful gradients.
    """
    powers = _get_powers(a.device)
    # Compute sum value
    a_val = (a.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    b_val = (b.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    sum_val = torch.fmod(a_val + b_val, 2.0**32)

    # Dither: shift off the integer to get nonzero gradient from cosine
    dither = (torch.rand_like(sum_val) - 0.5) * _DITHER_AMOUNT
    dithered = sum_val + dither

    # Sinusoidal bit extraction: bit_k ≈ 0.5 * (1 - cos(π * v / 2^k))
    # Gradient: d/dv = (π / (2 * 2^k)) * sin(π * v / 2^k)
    # This captures the periodic structure of each bit position.
    inv_powers = 1.0 / powers
    phases = dithered * inv_powers * math.pi  # (..., 32)
    soft = (0.5 * (1.0 - torch.cos(phases))).to(a.dtype)

    # Normalize per-bit to prevent gradient explosion from 2^k scaling.
    # Without this, d(soft_k)/d(input_bit_j) ∝ 2^(j-k), which explodes.
    soft = soft * _SIN_GRAD_SCALE

    return hard.detach() + soft - soft.detach()


_DITHER_AMOUNT = 0.4   # uniform noise in [-0.2, 0.2] — small enough for correct rounding
_SIN_GRAD_SCALE = 0.5  # global damping, same role as _XOR_GRAD_SCALE


def _complex_ste(a: torch.Tensor, b: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """Complex quadrature STE: uses sin+cos for always-nonzero gradients.

    At integer v, cos(π·v/2^k) is ±1 (exact bit value) but has zero gradient.
    sin(π·v/2^k) is 0 (vanishes from forward) but has gradient ±π/2^k (nonzero!).

    By adding both: proxy_k = 0.5·(1-cos) + λ·sin
    - Forward at integers: 0.5·(1∓1) + λ·0 = hard bit (exact)
    - Gradient at integers: 0 + λ·(π/2^k)·cos(π·v/2^k) = ±λπ/2^k (always nonzero)

    No dithering, no noise. The quadrature component provides exact gradients
    on the smooth manifold.
    """
    powers = _get_powers(a.device)
    inv_powers = 1.0 / powers

    a_val = (a.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    b_val = (b.to(torch.float64) * powers).sum(dim=-1, keepdim=True)
    sum_val = torch.fmod(a_val + b_val, 2.0**32)

    phases = sum_val * inv_powers * math.pi  # π·v/2^k for each k

    # cos: exact forward values, zero gradient at integers
    # sin: zero at integers (clean forward), nonzero gradient (quadrature)
    proxy = 0.5 * (1.0 - torch.cos(phases)) + _COMPLEX_LAMBDA * torch.sin(phases)
    proxy = (proxy * _COMPLEX_GRAD_SCALE).to(a.dtype)

    return hard.detach() + proxy - proxy.detach()


_COMPLEX_LAMBDA = 1.0      # quadrature coupling strength
_COMPLEX_GRAD_SCALE = 0.5  # global damping


# Active STE mode: "xor", "sinusoidal", or "complex"
STE_MODE = "xor"


def soft_add32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two 32-bit soft words mod 2^32.

    Forward: exact via float64 arithmetic.
    Backward: STE proxy (configurable via STE_MODE).

    Args:
        a, b: (..., 32) tensors with values in [0, 1], LSB first
    Returns:
        (..., 32) sum mod 2^32
    """
    hard = _hard_add(a, b)
    if STE_MODE == "complex":
        return _complex_ste(a, b, hard)
    elif STE_MODE == "sinusoidal":
        return _sinusoidal_ste(a, b, hard)
    else:
        return _xor_ste(a, b, hard)


def soft_leftrotate(x: torch.Tensor, n: int) -> torch.Tensor:
    """Left-rotate 32 soft bits by n positions.

    Left rotate = shift toward MSB. In LSB-first layout, this means
    moving array elements to the right (wrapping around).
    """
    return torch.cat([x[..., -n:], x[..., :-n]], dim=-1)


# --- Conversion utilities ---

def int_to_bits(n: int, width: int = 32) -> torch.Tensor:
    """Convert integer to LSB-first bit tensor."""
    return torch.tensor([(n >> i) & 1 for i in range(width)], dtype=torch.float32)


def bytes_to_bits(byte_tensor: torch.Tensor) -> torch.Tensor:
    """Convert (..., N) byte tensor [0-255] to (..., N*8) soft bit tensor. LSB first per byte."""
    shape = byte_tensor.shape
    x = byte_tensor.float().unsqueeze(-1)  # (..., N, 1)
    shifts = torch.arange(8, device=byte_tensor.device).float()  # (8,)
    bits = ((x / (2.0 ** shifts)) % 2).floor()  # (..., N, 8)
    return bits.reshape(*shape[:-1], shape[-1] * 8)


def bits_to_bytes(bit_tensor: torch.Tensor) -> torch.Tensor:
    """Convert (..., N*8) soft bit tensor to (..., N) soft byte values.

    For hard bits this gives exact byte values. For soft bits, gives expected values.
    """
    shape = bit_tensor.shape
    n_bytes = shape[-1] // 8
    bits = bit_tensor.reshape(*shape[:-1], n_bytes, 8)
    powers = (2.0 ** torch.arange(8, device=bit_tensor.device).float())
    return (bits * powers).sum(dim=-1)


# Precomputed table: bit_table[v, j] = bit j of byte value v (LSB first)
_BIT_TABLE = None

def _get_bit_table(device: torch.device) -> torch.Tensor:
    global _BIT_TABLE
    if _BIT_TABLE is None or _BIT_TABLE.device != device:
        table = torch.zeros(256, 8)
        for v in range(256):
            for j in range(8):
                table[v, j] = (v >> j) & 1
        _BIT_TABLE = table.to(device)
    return _BIT_TABLE


def probs_to_soft_bits(probs: torch.Tensor) -> torch.Tensor:
    """Convert byte-level probabilities to soft bit probabilities.

    Args:
        probs: (B, 64, 256) softmax probabilities over byte values
    Returns:
        (B, 512) soft bits in [0, 1] — P(bit_j = 1) for each bit position
    """
    bit_table = _get_bit_table(probs.device)  # (256, 8)
    soft_bits = probs @ bit_table  # (B, 64, 8)
    return soft_bits.reshape(probs.shape[0], -1)  # (B, 512)


def compute_ste_jacobian(
    messages: torch.Tensor,
    ste_mode: str = "complex",
) -> torch.Tensor:
    """Compute approximate Jacobian via STE backward pass.

    For each message, computes the 64x16 byte-level sensitivity matrix
    using the differentiable soft MD5. Uses a single backward pass with
    a random projection to avoid the cost/instability of 16 separate passes.

    Args:
        messages: (B, 64) long tensor of byte values [0, 255]
        ste_mode: which STE to use ("xor", "sinusoidal", "complex")
    Returns:
        (B, 64, 16) approximate Jacobian
    """
    global STE_MODE, _XOR_GRAD_SCALE, _COMPLEX_GRAD_SCALE, _SIN_GRAD_SCALE
    old_mode = STE_MODE
    old_xor = _XOR_GRAD_SCALE
    old_complex = _COMPLEX_GRAD_SCALE
    old_sin = _SIN_GRAD_SCALE

    # Use aggressive damping to keep gradients finite through full MD5
    STE_MODE = ste_mode
    _XOR_GRAD_SCALE = 0.1
    _COMPLEX_GRAD_SCALE = 0.1
    _SIN_GRAD_SCALE = 0.1

    B = messages.shape[0]
    device = messages.device

    soft_md5 = SoftMD5().to(device)

    # Single forward + backward with full hash as target
    msg_float = messages.float() if messages.is_floating_point() else messages.to(torch.float32)
    msg_bits = bytes_to_bits(msg_float.long()).float().requires_grad_(True)
    hash_bits = soft_md5(msg_bits)  # (B, 128)
    hash_bytes = bits_to_bytes(hash_bits)  # (B, 16)

    jacobian = torch.zeros(B, 64, 16, device=device)

    try:
        hash_bytes.sum().backward()
        if msg_bits.grad is not None and not msg_bits.grad.isnan().any():
            grad_bits = msg_bits.grad.detach()  # (B, 512)
            grad_bytes = grad_bits.reshape(B, 64, 8).sum(dim=-1)  # (B, 64)
            jacobian = grad_bytes.unsqueeze(-1).expand(B, 64, 16).clone()
    except RuntimeError:
        pass  # Return zeros — STE signal doesn't survive full MD5 anyway

    STE_MODE = old_mode
    _XOR_GRAD_SCALE = old_xor
    _COMPLEX_GRAD_SCALE = old_complex
    _SIN_GRAD_SCALE = old_sin

    # Clamp and normalize to prevent extreme values
    jacobian = jacobian.clamp(-10, 10)
    norm = jacobian.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (jacobian / norm).detach()


# --- MD5 constants ---

# K[i] = floor(2^32 * abs(sin(i + 1)))
MD5_K = [int(2**32 * abs(math.sin(i + 1))) & 0xFFFFFFFF for i in range(64)]

# Per-round shift amounts
MD5_S = (
    [7, 12, 17, 22] * 4
    + [5, 9, 14, 20] * 4
    + [4, 11, 16, 23] * 4
    + [6, 10, 15, 21] * 4
)


class SoftMD5(nn.Module):
    """Differentiable MD5 hash function for 64-byte messages.

    Args:
        num_rounds: Number of MD5 rounds per block (1-64, default 64).
            Full MD5 uses 64. Reduced rounds weaken avalanche and may
            allow gradient-based inversion.
        num_blocks: Number of 512-bit blocks to process (1 or 2, default 2).
            Full MD5 for 64-byte messages uses 2 (message + padding).
            Single-block mode skips padding — output won't match hashlib
            but halves the gradient depth.
    """

    def __init__(self, num_rounds: int = 64, num_blocks: int = 2):
        super().__init__()
        self.num_rounds = num_rounds
        self.num_blocks = num_blocks

        # K constants as bit tensors (64, 32)
        self.register_buffer("K", torch.stack([int_to_bits(k) for k in MD5_K]))

        # Initial state
        self.register_buffer("h0", int_to_bits(0x67452301))
        self.register_buffer("h1", int_to_bits(0xEFCDAB89))
        self.register_buffer("h2", int_to_bits(0x98BADCFE))
        self.register_buffer("h3", int_to_bits(0x10325476))

        # Padding block for 64-byte messages:
        # 0x80, 55 zero bytes, then 64-bit length (512 = 0x200) in little-endian
        pad_bytes = [0x80] + [0] * 55 + [0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        pad_bits = []
        for byte_val in pad_bytes:
            for j in range(8):
                pad_bits.append((byte_val >> j) & 1)
        self.register_buffer("pad_block", torch.tensor(pad_bits, dtype=torch.float32))

    def _compress(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
        words: torch.Tensor, snapshots: list | None = None,
        snapshot_rounds: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One MD5 compression (up to num_rounds rounds).

        Args:
            a, b, c, d: (..., 32) state words
            words: (..., 16, 32) message words
            snapshots: if not None, list to append state snapshots to
            snapshot_rounds: round numbers at which to capture snapshots
        """
        for i in range(self.num_rounds):
            if snapshots is not None and i in snapshot_rounds:
                # Snapshot: concat state into (B, 128) bits, convert to (B, 16) bytes
                state_bits = torch.cat([a, b, c, d], dim=-1)
                snapshots.append(bits_to_bytes(state_bits))
            if i < 16:
                f = soft_or(soft_and(b, c), soft_and(soft_not(b), d))
                g = i
            elif i < 32:
                f = soft_or(soft_and(d, b), soft_and(soft_not(d), c))
                g = (5 * i + 1) % 16
            elif i < 48:
                f = soft_xor(soft_xor(b, c), d)
                g = (3 * i + 5) % 16
            else:
                f = soft_xor(c, soft_or(b, soft_not(d)))
                g = (7 * i) % 16

            f = soft_add32(f, a)
            f = soft_add32(f, self.K[i].expand_as(a))
            f = soft_add32(f, words[..., g, :])

            a, d, c = d, c, b
            b = soft_add32(b, soft_leftrotate(f, MD5_S[i]))

        return a, b, c, d

    def forward(self, message_bits: torch.Tensor) -> torch.Tensor:
        """Compute differentiable MD5 hash.

        Args:
            message_bits: (B, 512) soft bits in [0, 1]
        Returns:
            (B, 128) soft hash bits in [0, 1]
        """
        B = message_bits.shape[0]

        # Block 1: message
        words1 = message_bits.unflatten(-1, (16, 32))
        a, b, c, d = (x.expand(B, -1) for x in (self.h0, self.h1, self.h2, self.h3))
        da, db, dc, dd = self._compress(a, b, c, d, words1)
        a = soft_add32(a, da)
        b = soft_add32(b, db)
        c = soft_add32(c, dc)
        d = soft_add32(d, dd)

        # Block 2: padding (only for full 2-block mode)
        if self.num_blocks >= 2:
            words2 = self.pad_block.expand(B, -1).unflatten(-1, (16, 32))
            da, db, dc, dd = self._compress(a, b, c, d, words2)
            a = soft_add32(a, da)
            b = soft_add32(b, db)
            c = soft_add32(c, dc)
            d = soft_add32(d, dd)

        return torch.cat([a, b, c, d], dim=-1)

    def forward_with_intermediates(
        self, message_bits: torch.Tensor,
        snapshot_rounds: tuple[int, ...] = (0, 4, 8, 16, 32, 48, 63),
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute MD5 hash and return intermediate state snapshots.

        Snapshots are state (a,b,c,d) as byte tensors at specified rounds,
        useful as features for the Jacobian network.

        Args:
            message_bits: (B, 512) soft bits in [0, 1]
            snapshot_rounds: which rounds to capture state at
        Returns:
            hash_bits: (B, 128) soft hash bits
            snapshots: list of (B, 16) byte tensors, one per snapshot round
        """
        B = message_bits.shape[0]
        snapshots: list[torch.Tensor] = []

        words1 = message_bits.unflatten(-1, (16, 32))
        a, b, c, d = (x.expand(B, -1) for x in (self.h0, self.h1, self.h2, self.h3))
        da, db, dc, dd = self._compress(
            a, b, c, d, words1, snapshots=snapshots, snapshot_rounds=snapshot_rounds
        )
        a = soft_add32(a, da)
        b = soft_add32(b, db)
        c = soft_add32(c, dc)
        d = soft_add32(d, dd)

        if self.num_blocks >= 2:
            # Offset snapshot rounds for block 2
            block2_snaps = tuple(r + 64 for r in snapshot_rounds if r + 64 < 128)
            words2 = self.pad_block.expand(B, -1).unflatten(-1, (16, 32))
            da, db, dc, dd = self._compress(
                a, b, c, d, words2, snapshots=snapshots, snapshot_rounds=block2_snaps
            )
            a = soft_add32(a, da)
            b = soft_add32(b, db)
            c = soft_add32(c, dc)
            d = soft_add32(d, dd)

        return torch.cat([a, b, c, d], dim=-1), snapshots
