#!/usr/bin/env python3
"""Sweep over reduced-round MD5 to find where gradient signal dies.

For each round count and STE mode, measures:
  1. Gradient cosine similarity (soft grad vs finite-difference)
  2. Direct optimization bit agreement after N steps

This maps the boundary between "invertible by gradient" and "cryptographically hard."

Usage:
    python scripts/round_sweep.py --device cuda:1
    python scripts/round_sweep.py --device cpu --opt-steps 500        # quick test
    python scripts/round_sweep.py --device cuda:1 --ste sinusoidal    # sinusoidal only
    python scripts/round_sweep.py --device cuda:1 --ste both          # compare both
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

import src.soft_md5 as smd5
from src.soft_md5 import SoftMD5


def measure_cosine_sim(num_rounds: int, num_blocks: int, n_points: int, device: str) -> dict:
    """Measure gradient quality for a given round count."""
    md5 = SoftMD5(num_rounds=num_rounds, num_blocks=num_blocks).to(device)
    cosines = []

    for _ in range(n_points):
        msg_hard = torch.randint(0, 2, (1, 512), device=device).float()
        target = torch.randint(0, 2, (1, 128), device=device).float()

        # Soft gradient
        msg = msg_hard.clone().requires_grad_(True)
        soft_hash = md5(msg)
        loss = ((soft_hash - target) ** 2).sum()
        loss.backward()
        soft_grad = msg.grad.detach().squeeze()

        # Finite-difference gradient
        with torch.no_grad():
            base_hash = (md5(msg_hard) > 0.5).float()
            base_loss = ((base_hash - target) ** 2).sum()
            fd_grad = torch.zeros(512, device=device)

            for i in range(512):
                flipped = msg_hard.clone()
                flipped[0, i] = 1.0 - flipped[0, i]
                flipped_hash = (md5(flipped) > 0.5).float()
                fd_grad[i] = ((flipped_hash - target) ** 2).sum() - base_loss

        if soft_grad.norm() > 0 and fd_grad.norm() > 0:
            cos = (soft_grad @ fd_grad) / (soft_grad.norm() * fd_grad.norm())
            cosines.append(cos.item())

    return {
        "mean_cos": sum(cosines) / len(cosines) if cosines else 0,
        "max_cos": max(cosines) if cosines else 0,
        "pct_positive": sum(1 for c in cosines if c > 0) / len(cosines) if cosines else 0,
    }


def measure_direct_opt(
    num_rounds: int, num_blocks: int, steps: int, n_restarts: int, device: str,
) -> dict:
    """Run direct optimization and measure best bit agreement."""
    md5 = SoftMD5(num_rounds=num_rounds, num_blocks=num_blocks).to(device)

    # Generate a target hash using THIS reduced-round MD5
    with torch.no_grad():
        true_msg_bits = torch.randint(0, 2, (1, 512), device=device).float()
        target_bits = (md5(true_msg_bits) > 0.5).float().expand(n_restarts, -1)

    params = torch.randn(n_restarts, 512, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=0.1)

    best_bit_agree = 0.5

    for step in range(1, steps + 1):
        message_bits = params.sigmoid()
        soft_hash = md5(message_bits)
        loss = ((soft_hash - target_bits) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % max(1, steps // 4) == 0 or step == steps:
            with torch.no_grad():
                hard = (message_bits > 0.5).float()
                hard_hash = (md5(hard) > 0.5).float()
                ba = (hard_hash == target_bits).float().mean(dim=-1).max().item()
                if ba > best_bit_agree:
                    best_bit_agree = ba

    return {"best_bit_agree": best_bit_agree, "final_loss": loss.item()}


def run_sweep(ste_mode: str, round_counts: list[int], args) -> list[dict]:
    """Run the full sweep for one STE mode."""
    smd5.STE_MODE = ste_mode
    results = []

    for nr in round_counts:
        t0 = time.perf_counter()
        print(f"  [{ste_mode:>10s}] rounds={nr:>2d} ...", end=" ", flush=True)

        cos = measure_cosine_sim(nr, args.blocks, args.cos_points, args.device)
        opt = measure_direct_opt(nr, args.blocks, args.opt_steps, args.opt_restarts, args.device)

        elapsed = time.perf_counter() - t0
        r = {"ste": ste_mode, "rounds": nr, **cos, **opt, "time": elapsed}
        results.append(r)
        print(f"cos={cos['mean_cos']:+.4f}  bit_agree={opt['best_bit_agree']:.3f}  ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Reduced-round MD5 gradient sweep")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cos-points", type=int, default=20, help="Points for cosine measurement")
    parser.add_argument("--opt-steps", type=int, default=2000, help="Steps for direct optimization")
    parser.add_argument("--opt-restarts", type=int, default=16, help="Parallel restarts")
    parser.add_argument("--rounds", type=str, default="1,2,3,4,6,8,16,32,64",
                        help="Comma-separated round counts to test")
    parser.add_argument("--blocks", type=int, default=1,
                        help="Number of blocks (1=single block, 2=full MD5 with padding)")
    parser.add_argument("--ste", type=str, default="all",
                        choices=["xor", "sinusoidal", "complex", "all"],
                        help="STE mode to test")
    args = parser.parse_args()

    round_counts = [int(r) for r in args.rounds.split(",")]
    modes = ["xor", "sinusoidal", "complex"] if args.ste == "all" else [args.ste]

    print(f"Reduced-round MD5 sweep: rounds={round_counts}")
    print(f"STE modes: {modes}, Blocks: {args.blocks}, Device: {args.device}")
    print(f"Cosine: {args.cos_points} points, Opt: {args.opt_steps} steps × {args.opt_restarts} restarts")
    print()

    all_results = []
    for mode in modes:
        results = run_sweep(mode, round_counts, args)
        all_results.extend(results)
        print()

    # Summary table
    if len(modes) > 1:
        # Side-by-side comparison
        by_mode = {m: {r["rounds"]: r for r in all_results if r["ste"] == m} for m in modes}

        # Header
        header_parts = [f"{'Rounds':>6s}"]
        for m in modes:
            label = m[:7].upper()
            header_parts.append(f"{'Cos':>7s} {'BA':>6s}")
        print("        " + "   ".join(f"--- {m} ---" for m in modes))
        print(f"{'Rounds':>8s}" + "   ".join(f"{'Cos':>8s} {'BitAgr':>8s}" for _ in modes))
        print("-" * (8 + len(modes) * 19))

        for nr in round_counts:
            parts = [f"{nr:>8d}"]
            for m in modes:
                r = by_mode[m].get(nr, {})
                parts.append(f"{r.get('mean_cos', 0):>+8.4f} {r.get('best_bit_agree', 0):>8.3f}")
            print("   ".join(parts))

        # Winner per round count
        print()
        for nr in round_counts:
            scores = {m: by_mode[m].get(nr, {}).get("best_bit_agree", 0) for m in modes}
            best_mode = max(scores, key=scores.get)
            best_score = scores[best_mode]
            others = [v for k, v in scores.items() if k != best_mode]
            if others and best_score > max(others) + 0.01:
                print(f"  Rounds={nr}: {best_mode} wins ({best_score:.3f} vs {max(others):.3f})")
    else:
        print(f"{'Rounds':>6s} {'STE':>12s} {'Cos(mean)':>10s} {'Cos(max)':>10s} {'%Pos':>6s} {'BitAgree':>10s} {'Loss':>8s}")
        print("-" * 65)
        for r in all_results:
            print(
                f"{r['rounds']:>6d} {r['ste']:>12s} {r['mean_cos']:>+10.4f} {r['max_cos']:>+10.4f} "
                f"{r['pct_positive']:>5.0%} {r['best_bit_agree']:>10.3f} {r['final_loss']:>8.4f}"
            )

    # Interpretation
    print()
    for mode in modes:
        mode_results = [r for r in all_results if r["ste"] == mode]
        invertible = [r for r in mode_results if r["best_bit_agree"] > 0.6]
        if invertible:
            threshold = max(invertible, key=lambda r: r["rounds"])
            print(f"  {mode}: invertible up to ~{threshold['rounds']} rounds (bit_agree > 60%)")
        else:
            print(f"  {mode}: no round count achieved >60% bit agreement")


if __name__ == "__main__":
    main()
