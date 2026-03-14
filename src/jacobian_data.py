"""Training data generator for the Jacobian network.

Generates perturbation data entirely on GPU using torch random ops
and the md5_gpu module for batched MD5 computation.
"""

import torch

from .md5_gpu import md5


def generate_jacobian_batch(
    batch_size: int,
    perturbations_per_msg: int = 8,
    max_delta: int = 3,
    device: str = "cpu",
) -> dict:
    """Generate training data for the Jacobian network.

    For each message:
    1. Sample random 64-byte message
    2. Compute base MD5 hash
    3. For each perturbation:
       - Pick random byte position
       - Apply random small perturbation (+-1 to +-max_delta)
       - Compute perturbed MD5 hash
       - Record (position, delta, hash_change)

    Args:
        batch_size: Number of base messages.
        perturbations_per_msg: Number of perturbations per message.
        max_delta: Maximum absolute perturbation value.
        device: Device for tensors.

    Returns:
        dict with:
            messages: (B, 64) base messages
            positions: (B, K) perturbed byte positions
            deltas: (B, K) perturbation magnitudes (signed)
            hash_changes: (B, K, 16) actual change in hash bytes (signed, [-128, 127])
    """
    B = batch_size
    K = perturbations_per_msg

    # Sample random 64-byte messages (all on GPU via torch)
    messages = torch.randint(0, 256, (B, 64), dtype=torch.int64, device=device)

    # Compute base hashes
    base_hashes = md5(messages)  # (B, 16)

    # Sample random positions and deltas for perturbations
    positions = torch.randint(0, 64, (B, K), dtype=torch.int64, device=device)

    # Sample non-zero deltas in [-max_delta, max_delta]
    # First sample from [1, max_delta], then randomly negate
    abs_deltas = torch.randint(1, max_delta + 1, (B, K), dtype=torch.int64, device=device)
    signs = torch.randint(0, 2, (B, K), dtype=torch.int64, device=device) * 2 - 1  # -1 or +1
    deltas = abs_deltas * signs

    # Apply perturbations and compute perturbed hashes
    # Expand messages for all perturbations: (B, K, 64)
    expanded = messages.unsqueeze(1).expand(B, K, 64).clone()

    # Scatter the perturbations into the right positions
    # positions: (B, K) -> (B, K, 1) for gather/scatter
    pos_idx = positions.unsqueeze(-1)  # (B, K, 1)
    original_vals = torch.gather(expanded, 2, pos_idx).squeeze(-1)  # (B, K)
    perturbed_vals = (original_vals + deltas) % 256  # wrap around byte range
    expanded.scatter_(2, pos_idx, perturbed_vals.unsqueeze(-1))

    # Reshape for batched MD5: (B*K, 64)
    perturbed_flat = expanded.reshape(B * K, 64)
    perturbed_hashes = md5(perturbed_flat).reshape(B, K, 16)

    # Compute signed hash changes: wrap to [-128, 127]
    base_expanded = base_hashes.unsqueeze(1).expand(B, K, 16)
    hash_changes = (perturbed_hashes - base_expanded + 128) % 256 - 128

    return {
        "messages": messages,
        "positions": positions,
        "deltas": deltas,
        "hash_changes": hash_changes,
    }


def compute_jacobian_targets(
    messages: torch.Tensor,
    positions: torch.Tensor,
    deltas: torch.Tensor,
    hash_changes: torch.Tensor,
) -> torch.Tensor:
    """Convert raw perturbation data to per-row Jacobian targets.

    For position i with perturbation delta, the Jacobian row target is:
    J_target[i] = hash_change / delta

    Args:
        messages: (B, 64) base messages (unused, kept for API consistency)
        positions: (B, K) perturbed byte positions
        deltas: (B, K) perturbation magnitudes (signed)
        hash_changes: (B, K, 16) actual change in hash bytes (signed)

    Returns:
        (B, K, 16) normalized sensitivity vectors
    """
    # Normalize by delta to get per-unit sensitivity
    # deltas: (B, K) -> (B, K, 1) for broadcasting
    targets = hash_changes.float() / deltas.float().unsqueeze(-1)
    return targets
