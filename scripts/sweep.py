#!/usr/bin/env python3
"""Quick model sizing sweep — runs short training for each config and compares.

Usage:
    python scripts/sweep.py                      # CPU, fast
    python scripts/sweep.py --device mps         # Mac GPU
    python scripts/sweep.py --device cuda        # 3090
    python scripts/sweep.py configs/sweep/*.toml # specific configs
"""
import argparse
import glob
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from src.config import Config
from src.data import generate_batch
from src.diffusion import compute_loss, mask_message, sample, sample_mask_ratio
from src.eval import metrics_report
from src.model import Denoiser


def run_one(cfg: Config, steps: int, device: str) -> dict:
    """Train for `steps` steps, return final loss + metrics + throughput."""
    torch.manual_seed(cfg.seed)
    device_type = device.split(":")[0]
    model = Denoiser(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map.get(cfg.dtype, torch.float32)
    use_amp = amp_dtype != torch.float32

    # Warmup (compile + kernels)
    for _ in range(3):
        msgs, hashes = generate_batch(cfg.batch_size, device=device)
        t = sample_mask_ratio(cfg.batch_size, device=device)
        x_noisy, mask = mask_message(msgs, t, cfg.mask_token)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x_noisy, hashes, t)
            loss = compute_loss(logits, msgs, mask)
        loss.backward()
        optimizer.zero_grad()

    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()

    losses = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        msgs, hashes = generate_batch(cfg.batch_size, device=device)
        t = sample_mask_ratio(cfg.batch_size, device=device)
        x_noisy, mask = mask_message(msgs, t, cfg.mask_token)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x_noisy, hashes, t)
            loss = compute_loss(logits, msgs, mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    steps_per_sec = steps / elapsed
    samples_per_sec = steps * cfg.batch_size / elapsed

    # Quick sampling-based check
    model_for_sampling = model
    model_for_sampling.eval()
    sample_hashes = generate_batch(256, device=device)[1]
    with torch.no_grad():
        cfg_sampling = Config(
            seq_len=cfg.seq_len, vocab_size=cfg.vocab_size,
            mask_token=cfg.mask_token, sampling_steps=min(32, cfg.sampling_steps),
        )
        generated = sample(model_for_sampling, sample_hashes, cfg_sampling)
    report = metrics_report(generated, sample_hashes)

    return {
        "params": param_count,
        "final_loss": sum(losses[-10:]) / 10,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
        "bit_agreement": report["bit_agreement"],
        "byte_agreement": report["byte_agreement"],
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Model sizing sweep")
    parser.add_argument("configs", nargs="*", help="TOML config files to sweep")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args()

    if args.configs:
        config_paths = []
        for p in args.configs:
            config_paths.extend(glob.glob(p))
    else:
        config_paths = sorted(glob.glob("configs/sweep/*.toml"))

    if not config_paths:
        print("No configs found. Put .toml files in configs/sweep/ or pass paths as arguments.")
        return

    print(f"Sweeping {len(config_paths)} configs, {args.steps} steps each, device={args.device}")
    print()

    results = []
    for path in config_paths:
        cfg = Config.from_toml(path)
        if args.batch_size is not None:
            cfg = cfg.override(batch_size=args.batch_size)
        if args.dtype is not None:
            cfg = cfg.override(dtype=args.dtype)
        cfg = cfg.override(device=args.device)

        name = path.split("/")[-1].replace(".toml", "")
        tag = f"L{cfg.n_layers}_d{cfg.d_model}_h{cfg.n_heads}"
        print(f"  Running {name} ({tag}, {sum(p.numel() for p in Denoiser(cfg).parameters()):,} params)...")

        r = run_one(cfg, args.steps, args.device)
        r["name"] = name
        r["tag"] = tag
        results.append(r)

    # Summary table
    print()
    header = f"{'Config':<20s} {'Params':>10s} {'Loss':>8s} {'Bit%':>7s} {'Byte%':>7s} {'step/s':>8s} {'samp/s':>9s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<20s} {r['params']:>10,d} {r['final_loss']:>8.4f} "
            f"{r['bit_agreement']:>6.3f} {r['byte_agreement']:>6.4f} "
            f"{r['steps_per_sec']:>8.1f} {r['samples_per_sec']:>9.0f}"
        )

    # Efficiency ranking
    print()
    print("Efficiency ranking (bit_agreement per second of training):")
    ranked = sorted(results, key=lambda r: r["bit_agreement"] / r["elapsed"], reverse=True)
    for i, r in enumerate(ranked):
        eff = r["bit_agreement"] / r["elapsed"]
        print(f"  {i+1}. {r['name']} — {eff:.4f} bit_agree/sec")


if __name__ == "__main__":
    main()
