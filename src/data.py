import hashlib
import os

import torch


def generate_batch(batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (message_bytes, hash_bytes) pairs on the fly.

    Returns:
        messages: (batch_size, 64) uint8 tensor — 512-bit message as 64 bytes
        hashes:   (batch_size, 16) uint8 tensor — 128-bit MD5 hash as 16 bytes
    """
    raw = [os.urandom(64) for _ in range(batch_size)]
    digests = [hashlib.md5(m).digest() for m in raw]

    messages = torch.tensor([list(m) for m in raw], dtype=torch.long, device=device)
    hashes = torch.tensor([list(d) for d in digests], dtype=torch.long, device=device)
    return messages, hashes


def make_val_set(n: int = 10_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed validation set for consistent evaluation."""
    return generate_batch(n, device="cpu")
