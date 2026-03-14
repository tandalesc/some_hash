#!/usr/bin/env python3
"""Verify soft MD5 correctness and measure gradient quality."""
import hashlib
import sys

import torch

sys.path.insert(0, ".")

from src.soft_md5 import SoftMD5, bytes_to_bits, bits_to_bytes


def check_correctness(n: int = 100):
    """Verify soft MD5 matches hashlib on hard (binary) inputs."""
    md5 = SoftMD5()
    matches = 0

    for _ in range(n):
        msg = torch.randint(0, 256, (64,))
        msg_bytes = bytes(msg.tolist())
        expected = hashlib.md5(msg_bytes).digest()
        expected_bits = bytes_to_bits(torch.tensor(list(expected)).unsqueeze(0))

        msg_bits = bytes_to_bits(msg.unsqueeze(0))
        got_bits = md5(msg_bits)

        # Round to hard bits and compare
        got_hard = (got_bits > 0.5).float()
        if torch.allclose(got_hard, expected_bits):
            matches += 1

    print(f"Correctness: {matches}/{n} exact matches with hashlib")
    return matches == n


def measure_gradient_quality(n_points: int = 20, n_bits: int = 512):
    """Measure cosine similarity between soft gradients and finite-difference gradients."""
    md5 = SoftMD5()
    cosine_sims = []

    for _ in range(n_points):
        # Random hard message
        msg_hard = torch.randint(0, 2, (1, 512)).float()
        # Random target hash
        target = torch.randint(0, 2, (1, 128)).float()

        # Soft gradient
        msg_soft = msg_hard.clone().requires_grad_(True)
        soft_hash = md5(msg_soft)
        loss = ((soft_hash - target) ** 2).sum()
        loss.backward()
        soft_grad = msg_soft.grad.detach().squeeze()

        # Finite-difference "gradient" through real MD5
        # Flip each input bit, measure change in loss
        fd_grad = torch.zeros(n_bits)
        base_hash_bits = (md5(msg_hard).detach() > 0.5).float()
        base_loss = ((base_hash_bits - target) ** 2).sum().item()

        for i in range(n_bits):
            flipped = msg_hard.clone()
            flipped[0, i] = 1.0 - flipped[0, i]
            flipped_hash = (md5(flipped).detach() > 0.5).float()
            flipped_loss = ((flipped_hash - target) ** 2).sum().item()
            fd_grad[i] = flipped_loss - base_loss

        # Cosine similarity
        if soft_grad.norm() > 0 and fd_grad.norm() > 0:
            cos = (soft_grad @ fd_grad) / (soft_grad.norm() * fd_grad.norm())
            cosine_sims.append(cos.item())

    avg_cos = sum(cosine_sims) / len(cosine_sims)
    print(f"Gradient quality ({n_points} random points):")
    print(f"  Mean cosine similarity: {avg_cos:.4f}")
    print(f"  Min: {min(cosine_sims):.4f}, Max: {max(cosine_sims):.4f}")
    print(f"  Positive (useful signal): {sum(1 for c in cosine_sims if c > 0)}/{len(cosine_sims)}")
    return avg_cos


def measure_gradient_norms():
    """Check if gradients survive through 128 rounds of mixing."""
    md5 = SoftMD5()

    # Test at hard bits
    msg = torch.randint(0, 2, (1, 512)).float().requires_grad_(True)
    target = torch.randint(0, 2, (1, 128)).float()
    soft_hash = md5(msg)
    loss = ((soft_hash - target) ** 2).sum()
    loss.backward()
    grad = msg.grad.detach()
    print(f"\nGradient norms (hard input bits):")
    print(f"  L2 norm: {grad.norm():.4f}")
    print(f"  Mean abs: {grad.abs().mean():.6f}")
    print(f"  Max abs: {grad.abs().max():.6f}")
    print(f"  Nonzero (>1e-8): {(grad.abs() > 1e-8).sum().item()}/512")

    # Test at soft bits (0.5 = maximum gradient flow)
    msg2 = torch.full((1, 512), 0.5, requires_grad=True)
    soft_hash2 = md5(msg2)
    loss2 = ((soft_hash2 - target) ** 2).sum()
    loss2.backward()
    grad2 = msg2.grad.detach()
    print(f"\nGradient norms (soft input bits = 0.5):")
    print(f"  L2 norm: {grad2.norm():.4f}")
    print(f"  Mean abs: {grad2.abs().mean():.6f}")
    print(f"  Max abs: {grad2.abs().max():.6f}")
    print(f"  Nonzero (>1e-8): {(grad2.abs() > 1e-8).sum().item()}/512")


if __name__ == "__main__":
    print("=== Soft MD5 Gradient Check ===\n")
    correct = check_correctness(100)
    if not correct:
        print("WARNING: Soft MD5 does not match hashlib! Fix before proceeding.")
        sys.exit(1)
    print()
    measure_gradient_norms()
    print()
    measure_gradient_quality(n_points=20)
