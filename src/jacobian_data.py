"""Training data generator for the Jacobian network.

Generates perturbation data entirely on GPU using torch random ops
and the md5_gpu module for batched MD5 computation.

Optionally computes STE Jacobian approximations and soft MD5 intermediate
states as auxiliary features.
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
    abs_deltas = torch.randint(1, max_delta + 1, (B, K), dtype=torch.int64, device=device)
    signs = torch.randint(0, 2, (B, K), dtype=torch.int64, device=device) * 2 - 1
    deltas = abs_deltas * signs

    # Apply perturbations and compute perturbed hashes
    expanded = messages.unsqueeze(1).expand(B, K, 64).clone()
    pos_idx = positions.unsqueeze(-1)
    original_vals = torch.gather(expanded, 2, pos_idx).squeeze(-1)
    perturbed_vals = (original_vals + deltas) % 256
    expanded.scatter_(2, pos_idx, perturbed_vals.unsqueeze(-1))

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

    Returns:
        (B, K, 16) normalized sensitivity vectors
    """
    targets = hash_changes.float() / deltas.float().unsqueeze(-1)
    return targets


def compute_ste_features(
    messages: torch.Tensor,
    ste_mode: str = "complex",
) -> torch.Tensor:
    """Compute STE Jacobian approximation as input features.

    Args:
        messages: (B, 64) int64 byte tensor
        ste_mode: which STE backward pass to use

    Returns:
        (B, 64, 16) approximate Jacobian from soft MD5
    """
    from .soft_md5 import compute_ste_jacobian
    return compute_ste_jacobian(messages, ste_mode=ste_mode)


def compute_intermediate_features(
    messages: torch.Tensor,
    snapshot_rounds: tuple[int, ...] = (0, 4, 8, 16, 32, 48, 63),
) -> torch.Tensor:
    """Compute soft MD5 intermediate state snapshots as features.

    Args:
        messages: (B, 64) int64 byte tensor
        snapshot_rounds: which rounds to capture state at

    Returns:
        (B, N, 16) intermediate state bytes, N = len(snapshot_rounds)
    """
    from .soft_md5 import SoftMD5, bytes_to_bits

    device = messages.device
    soft_md5 = SoftMD5().to(device)

    # Convert to soft bits (detached — no gradient needed for features)
    msg_bits = bytes_to_bits(messages).float()

    with torch.no_grad():
        _, snapshots = soft_md5.forward_with_intermediates(
            msg_bits, snapshot_rounds=snapshot_rounds
        )

    # Stack snapshots: list of (B, 16) -> (B, N, 16)
    return torch.stack(snapshots, dim=1)
