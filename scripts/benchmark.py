#!/usr/bin/env python3
"""Benchmarking suite with statistical tests."""
import argparse
import math
import sys

import torch

sys.path.insert(0, ".")

from src.config import Config
from src.data import make_val_set
from src.diffusion import sample
from src.eval import metrics_report
from src.model import Denoiser


def main():
    parser = argparse.ArgumentParser(description="Benchmark MD5 preimage model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-hashes", type=int, default=1000, help="Number of hashes to test")
    parser.add_argument("--samples-per-hash", type=int, default=1, help="Samples per hash")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    if args.steps is not None:
        cfg.sampling_steps = args.steps

    model = Denoiser(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    # Generate test hashes
    _, hashes = make_val_set(args.n_hashes)
    hashes = hashes.to(args.device)

    print(f"Benchmarking on {args.n_hashes} hashes, {args.samples_per_hash} sample(s) each")
    print(f"Sampling steps: {cfg.sampling_steps}")
    print()

    all_reports = []
    batch_size = min(256, args.n_hashes)

    for start in range(0, args.n_hashes, batch_size):
        end = min(start + batch_size, args.n_hashes)
        batch_hashes = hashes[start:end]

        if args.samples_per_hash > 1:
            batch_hashes = batch_hashes.repeat_interleave(args.samples_per_hash, dim=0)

        with torch.no_grad():
            samples = sample(model, batch_hashes, cfg)

        report = metrics_report(samples, batch_hashes)
        all_reports.append(report)
        print(f"  batch [{start}:{end}] | bit_agree={report['bit_agreement']:.4f}")

    # Aggregate
    avg_exact = sum(r["exact_match_rate"] for r in all_reports) / len(all_reports)
    avg_bit = sum(r["bit_agreement"] for r in all_reports) / len(all_reports)
    avg_byte = sum(r["byte_agreement"] for r in all_reports) / len(all_reports)

    print()
    print("=" * 60)
    print(f"  Exact match rate:  {avg_exact:.6f}  (random: ~0)")
    print(f"  Bit agreement:     {avg_bit:.4f}    (random: 0.5000)")
    print(f"  Byte agreement:    {avg_byte:.4f}    (random: 0.0039)")
    print("=" * 60)

    # Statistical significance test for bit agreement
    if avg_bit > 0.5:
        n = args.n_hashes * 128  # total bits compared
        z = (avg_bit - 0.5) / math.sqrt(0.25 / n)
        print(f"\n  Bit agreement z-score vs random: {z:.2f}")
        if z > 3.0:
            print("  Result: statistically significant (p < 0.001)")
        elif z > 2.0:
            print("  Result: statistically significant (p < 0.05)")
        else:
            print("  Result: not statistically significant")
    else:
        print("\n  Bit agreement at or below random chance.")


if __name__ == "__main__":
    main()
