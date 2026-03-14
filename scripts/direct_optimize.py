#!/usr/bin/env python3
"""Direct gradient descent to find MD5 preimages.

No model, no training data. Just backprop through differentiable MD5
and optimize the input message directly.
"""
import argparse
import hashlib
import os
import sys
import time

import torch

sys.path.insert(0, ".")

from src.soft_md5 import SoftMD5, bytes_to_bits, bits_to_bytes


def optimize(
    target_hash: bytes,
    steps: int = 5000,
    lr: float = 0.1,
    n_restarts: int = 8,
    device: str = "cpu",
):
    """Try to find a preimage of target_hash via gradient descent.

    Runs multiple restarts in parallel as a batch.
    """
    md5 = SoftMD5().to(device)
    target_bits = bytes_to_bits(
        torch.tensor(list(target_hash), device=device).unsqueeze(0)
    ).expand(n_restarts, -1)

    # Learnable parameters — unconstrained, mapped to [0,1] via sigmoid
    params = torch.randn(n_restarts, 512, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=lr)

    best_bit_agree = 0.0
    best_msg = None

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        message_bits = params.sigmoid()
        soft_hash = md5(message_bits)

        # MSE loss on soft hash bits vs target
        loss = ((soft_hash - target_bits) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                hard_msgs = (message_bits > 0.5).float()
                hard_hash = md5(hard_msgs)
                hard_hash_rounded = (hard_hash > 0.5).float()
                bit_agree = (hard_hash_rounded == target_bits).float().mean(dim=-1)
                best_idx = bit_agree.argmax()
                best_ba = bit_agree[best_idx].item()

                if best_ba > best_bit_agree:
                    best_bit_agree = best_ba
                    best_msg = hard_msgs[best_idx]

                # Check for exact match
                for i in range(n_restarts):
                    msg_bytes = bits_to_bytes(hard_msgs[i:i+1]).squeeze().byte().tolist()
                    real_hash = hashlib.md5(bytes(msg_bytes)).digest()
                    if real_hash == target_hash:
                        elapsed = time.perf_counter() - t0
                        print(f"\n  EXACT MATCH at step {step}! ({elapsed:.1f}s)")
                        print(f"  Message: {bytes(msg_bytes).hex()}")
                        return True, bytes(msg_bytes)

            elapsed = time.perf_counter() - t0
            grad_norm = params.grad.norm().item() if params.grad is not None else 0
            print(
                f"  step {step:>5d} | loss {loss.item():.6f} | "
                f"bit_agree {best_ba:.3f} ({best_ba*128:.0f}/128) | "
                f"grad_norm {grad_norm:.4f} | {elapsed:.1f}s"
            )

    return False, best_msg


def main():
    parser = argparse.ArgumentParser(description="Direct gradient descent MD5 preimage search")
    parser.add_argument("--hash", type=str, default=None, help="Target MD5 hex hash")
    parser.add_argument("--random", action="store_true", help="Use random target hash")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=8, help="Parallel restarts (batch)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.hash:
        target = bytes.fromhex(args.hash)
    elif args.random:
        msg = os.urandom(64)
        target = hashlib.md5(msg).digest()
        print(f"Random target: {target.hex()}")
        print(f"(known preimage: {msg.hex()})")
    else:
        parser.error("Provide --hash or --random")

    print(f"\nOptimizing {args.restarts} restarts × {args.steps} steps (lr={args.lr})")
    print(f"Device: {args.device}\n")

    found, best_msg = optimize(
        target, steps=args.steps, lr=args.lr,
        n_restarts=args.restarts, device=args.device,
    )

    if not found:
        print(f"\nNo exact match found (expected — this is MD5).")
        print(f"Best bit agreement: {(best_msg is not None) and 'see above' or 'N/A'}")
        print(f"This measures whether gradients flow through the soft MD5 at all.")


if __name__ == "__main__":
    main()
