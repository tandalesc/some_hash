import hashlib

import torch


def compute_md5_batch(messages: torch.Tensor) -> torch.Tensor:
    """Compute MD5 hashes for a batch of messages on CPU.

    Args:
        messages: (B, 64) long tensor of byte values
    Returns:
        (B, 16) long tensor of hash byte values
    """
    results = []
    for msg in messages.cpu().tolist():
        digest = hashlib.md5(bytes(msg)).digest()
        results.append(list(digest))
    return torch.tensor(results, dtype=torch.long)


def exact_match_rate(pred_messages: torch.Tensor, target_hashes: torch.Tensor) -> float:
    """Fraction of generated messages whose MD5 exactly matches the target."""
    pred_hashes = compute_md5_batch(pred_messages)
    target_hashes = target_hashes.cpu()
    matches = (pred_hashes == target_hashes).all(dim=-1)
    return matches.float().mean().item()


def bit_agreement(pred_messages: torch.Tensor, target_hashes: torch.Tensor) -> float:
    """Average fraction of hash bits matching between prediction and target.

    Random chance is 50%. Anything above that means the model learned something.
    """
    pred_hashes = compute_md5_batch(pred_messages)
    target_hashes = target_hashes.cpu()

    # Expand bytes to bits
    pred_bits = _bytes_to_bits(pred_hashes)
    target_bits = _bytes_to_bits(target_hashes)

    agreement = (pred_bits == target_bits).float().mean().item()
    return agreement


def byte_agreement(pred_messages: torch.Tensor, target_hashes: torch.Tensor) -> float:
    """Average fraction of hash bytes matching between prediction and target."""
    pred_hashes = compute_md5_batch(pred_messages)
    target_hashes = target_hashes.cpu()
    return (pred_hashes == target_hashes).float().mean().item()


def _bytes_to_bits(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, N) byte tensor to (B, N*8) bit tensor."""
    bits = []
    for i in range(8):
        bits.append((x >> (7 - i)) & 1)
    return torch.stack(bits, dim=-1).reshape(x.shape[0], -1)


def metrics_report(pred_messages: torch.Tensor, target_hashes: torch.Tensor) -> dict:
    """Full metrics report."""
    return {
        "exact_match_rate": exact_match_rate(pred_messages, target_hashes),
        "bit_agreement": bit_agreement(pred_messages, target_hashes),
        "byte_agreement": byte_agreement(pred_messages, target_hashes),
    }
