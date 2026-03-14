#!/usr/bin/env python3
"""Generate preimages for given MD5 hashes."""
import argparse
import hashlib
import sys

import torch

sys.path.insert(0, ".")

from src.config import Config
from src.diffusion import sample
from src.model import Denoiser


def main():
    parser = argparse.ArgumentParser(description="Sample MD5 preimages")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--hash", type=str, default=None, help="Target MD5 hex hash")
    parser.add_argument("--random", action="store_true", help="Sample for a random hash")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    if args.steps is not None:
        cfg.sampling_steps = args.steps

    model = Denoiser(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    if args.hash:
        hash_bytes = bytes.fromhex(args.hash)
        assert len(hash_bytes) == 16, "MD5 hash must be 32 hex chars (16 bytes)"
    elif args.random:
        import os
        msg = os.urandom(64)
        hash_bytes = hashlib.md5(msg).digest()
        print(f"Random target hash: {hash_bytes.hex()}")
        print(f"(original message: {msg.hex()})")
    else:
        parser.error("Provide --hash or --random")

    target = torch.tensor([list(hash_bytes)], dtype=torch.long, device=args.device)
    target = target.expand(args.n_samples, -1)

    with torch.no_grad():
        samples = sample(model, target, cfg)

    print(f"\nGenerated {args.n_samples} candidate preimages:")
    for i, s in enumerate(samples):
        msg_bytes = bytes(s.cpu().tolist())
        actual_hash = hashlib.md5(msg_bytes).digest()
        match = actual_hash == hash_bytes
        total_bits = 128
        agreeing = sum(
            8 - bin(a ^ b).count("1") for a, b in zip(actual_hash, hash_bytes)
        )
        print(
            f"  [{i}] hash={actual_hash.hex()} | "
            f"bit_agree={agreeing}/{total_bits} ({agreeing/total_bits:.1%}) | "
            f"{'MATCH!' if match else 'no match'}"
        )


if __name__ == "__main__":
    main()
