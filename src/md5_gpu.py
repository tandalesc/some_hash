"""Pure PyTorch MD5 implementation using int64 tensors.

Batched, runs on GPU. No autograd needed — this is for fast forward evaluation.
Uses int64 tensors to avoid overflow on 32-bit additions, masking with
& 0xFFFFFFFF after each addition for mod 2^32.
"""

import math

import torch


# MD5 per-round shift amounts
_SHIFT = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
]

# MD5 constants: K[i] = floor(2^32 * abs(sin(i+1)))
_K = [int(math.floor((2**32) * abs(math.sin(i + 1)))) for i in range(64)]

# Initial state
_A0 = 0x67452301
_B0 = 0xEFCDAB89
_C0 = 0x98BADCFE
_D0 = 0x10325476

MASK32 = 0xFFFFFFFF


def _left_rotate(x: torch.Tensor, n: int) -> torch.Tensor:
    """Left rotate 32-bit value stored in int64 tensor."""
    return ((x << n) | (x >> (32 - n))) & MASK32


def _bytes_to_words(block: torch.Tensor) -> torch.Tensor:
    """Convert (B, 64) byte tensor to (B, 16) little-endian 32-bit words."""
    # block: (B, 64) -> reshape to (B, 16, 4)
    b = block.reshape(block.shape[0], 16, 4)
    # Little-endian: word = b0 + b1<<8 + b2<<16 + b3<<24
    words = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16) | (b[:, :, 3] << 24)
    return words


def _words_to_bytes(state: torch.Tensor) -> torch.Tensor:
    """Convert (B, 4) state words to (B, 16) little-endian bytes."""
    result = []
    for i in range(4):
        w = state[:, i]
        result.append(w & 0xFF)
        result.append((w >> 8) & 0xFF)
        result.append((w >> 16) & 0xFF)
        result.append((w >> 24) & 0xFF)
    return torch.stack(result, dim=1)


def _process_block(M: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                   c: torch.Tensor, d: torch.Tensor,
                   K_tensor: torch.Tensor,
                   num_rounds: int = 64) -> tuple:
    """Process a single 512-bit block through MD5 compression.

    Args:
        M: (B, 16) message words
        a, b, c, d: (B,) current state
        K_tensor: (64,) precomputed constants on device
        num_rounds: number of MD5 rounds to run (1-64, default 64)
    Returns:
        Updated (a, b, c, d)
    """
    a0, b0, c0, d0 = a, b, c, d

    for i in range(num_rounds):
        if i < 16:
            # F(b,c,d) = (b & c) | (~b & d)
            f = (b & c) | ((b ^ MASK32) & d)
            g = i
        elif i < 32:
            # G(b,c,d) = (d & b) | (~d & c)
            f = (d & b) | ((d ^ MASK32) & c)
            g = (5 * i + 1) % 16
        elif i < 48:
            # H(b,c,d) = b ^ c ^ d
            f = b ^ c ^ d
            g = (3 * i + 5) % 16
        else:
            # I(b,c,d) = c ^ (b | ~d)
            f = c ^ (b | (d ^ MASK32))
            g = (7 * i) % 16

        f = (f + a + K_tensor[i] + M[:, g]) & MASK32
        a = d
        d = c
        c = b
        b = (b + _left_rotate(f, _SHIFT[i])) & MASK32

    a = (a + a0) & MASK32
    b = (b + b0) & MASK32
    c = (c + c0) & MASK32
    d = (d + d0) & MASK32

    return a, b, c, d


def md5(messages: torch.Tensor, num_rounds: int = 64,
        num_blocks: int = 2) -> torch.Tensor:
    """Compute MD5 hash of 64-byte messages using pure PyTorch.

    Args:
        messages: (B, 64) int64 tensor of byte values [0, 255]
        num_rounds: MD5 rounds per block (1-64, default 64 = full MD5)
        num_blocks: 1 = message block only, 2 = message + padding (default)
    Returns:
        (B, 16) int64 tensor of hash bytes [0, 255]
    """
    B = messages.shape[0]
    device = messages.device

    K_tensor = torch.tensor(_K, dtype=torch.int64, device=device)
    M1 = _bytes_to_words(messages)

    a = torch.full((B,), _A0, dtype=torch.int64, device=device)
    b = torch.full((B,), _B0, dtype=torch.int64, device=device)
    c = torch.full((B,), _C0, dtype=torch.int64, device=device)
    d = torch.full((B,), _D0, dtype=torch.int64, device=device)

    # Block 1: message
    a, b, c, d = _process_block(M1, a, b, c, d, K_tensor, num_rounds)

    # Block 2: padding
    if num_blocks >= 2:
        pad_block = torch.zeros(B, 64, dtype=torch.int64, device=device)
        pad_block[:, 0] = 0x80
        pad_block[:, 56] = 0x00
        pad_block[:, 57] = 0x02
        M2 = _bytes_to_words(pad_block)
        a, b, c, d = _process_block(M2, a, b, c, d, K_tensor, num_rounds)

    state = torch.stack([a, b, c, d], dim=1)
    return _words_to_bytes(state)


def md5_intermediates(messages: torch.Tensor, num_rounds: int = 64,
                      num_blocks: int = 1) -> list[torch.Tensor]:
    """Compute MD5 and return state (a,b,c,d) after every round.

    Args:
        messages: (B, 64) int64 byte tensor
        num_rounds: rounds per block
        num_blocks: 1 or 2
    Returns:
        List of (B, 4) int64 tensors — state after each round.
        Length = num_rounds * num_blocks + 1 (includes initial state).
    """
    B = messages.shape[0]
    device = messages.device
    K_tensor = torch.tensor(_K, dtype=torch.int64, device=device)
    M1 = _bytes_to_words(messages)

    a = torch.full((B,), _A0, dtype=torch.int64, device=device)
    b = torch.full((B,), _B0, dtype=torch.int64, device=device)
    c = torch.full((B,), _C0, dtype=torch.int64, device=device)
    d = torch.full((B,), _D0, dtype=torch.int64, device=device)

    states = [torch.stack([a, b, c, d], dim=1)]  # initial state

    def _run_block(M, a, b, c, d, nr):
        a0, b0, c0, d0 = a, b, c, d
        for i in range(nr):
            if i < 16:
                f = (b & c) | ((b ^ MASK32) & d)
                g = i
            elif i < 32:
                f = (d & b) | ((d ^ MASK32) & c)
                g = (5 * i + 1) % 16
            elif i < 48:
                f = b ^ c ^ d
                g = (3 * i + 5) % 16
            else:
                f = c ^ (b | (d ^ MASK32))
                g = (7 * i) % 16
            f = (f + a + K_tensor[i] + M[:, g]) & MASK32
            a = d
            d = c
            c = b
            b = (b + _left_rotate(f, _SHIFT[i])) & MASK32
            states.append(torch.stack([a, b, c, d], dim=1))
        a = (a + a0) & MASK32
        b = (b + b0) & MASK32
        c = (c + c0) & MASK32
        d = (d + d0) & MASK32
        return a, b, c, d

    a, b, c, d = _run_block(M1, a, b, c, d, num_rounds)

    if num_blocks >= 2:
        pad_block = torch.zeros(B, 64, dtype=torch.int64, device=device)
        pad_block[:, 0] = 0x80
        pad_block[:, 56] = 0x00
        pad_block[:, 57] = 0x02
        M2 = _bytes_to_words(pad_block)
        a, b, c, d = _run_block(M2, a, b, c, d, num_rounds)

    return states


def verify_against_hashlib(num_tests: int = 1000, device: str = "cpu") -> bool:
    """Verify GPU MD5 matches hashlib for random inputs.

    Args:
        num_tests: Number of random messages to test.
        device: Device to run on.
    Returns:
        True if all tests pass.
    """
    import hashlib

    messages_raw = [list(bytes(torch.randint(0, 256, (64,)).tolist())) for _ in range(num_tests)]
    messages_tensor = torch.tensor(messages_raw, dtype=torch.int64, device=device)

    gpu_hashes = md5(messages_tensor).cpu().tolist()

    for i in range(num_tests):
        expected = list(hashlib.md5(bytes(messages_raw[i])).digest())
        if gpu_hashes[i] != expected:
            print(f"MISMATCH at index {i}")
            print(f"  input:    {messages_raw[i][:8]}...")
            print(f"  expected: {expected}")
            print(f"  got:      {gpu_hashes[i]}")
            return False

    print(f"All {num_tests} tests passed.")
    return True


if __name__ == "__main__":
    verify_against_hashlib()
