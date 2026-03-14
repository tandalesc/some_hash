#!/usr/bin/env python3
"""Train the learned Jacobian network (Phase 1).

Trains a small transformer to predict MD5's local sensitivity:
how each input byte affects each hash byte at a given operating point.

Supports three modes:
  --mode scratch:   learn from raw message bytes only (baseline)
  --mode ste:       learn residual on top of STE Jacobian approximation
  --mode full:      STE Jacobian + soft MD5 intermediate states
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import JacobianConfig
from src.jacobian_data import (
    compute_intermediate_features,
    compute_jacobian_targets,
    compute_ste_features,
    generate_jacobian_batch,
)
from src.jacobian_net import JacobianNet


def _save_checkpoint(model, optimizer, cfg, step, path):
    state = model._orig_mod.state_dict() if cfg.compile else model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step}, path)


def _compute_features(messages, cfg, device):
    """Compute optional STE and intermediate features."""
    ste_jac = None
    intermediates = None

    if cfg.use_ste_features:
        ste_jac = compute_ste_features(messages, ste_mode=cfg.ste_mode)

    if cfg.use_intermediate_features:
        intermediates = compute_intermediate_features(messages)

    return ste_jac, intermediates


def run_validation(model, cfg, device, device_type, amp_dtype, use_amp, num_batches=4):
    """Run validation on held-out perturbation data."""
    total_cos = 0.0
    total_positive = 0
    total_count = 0

    for _ in range(num_batches):
        data = generate_jacobian_batch(
            batch_size=cfg.batch_size,
            perturbations_per_msg=cfg.perturbations_per_msg,
            max_delta=cfg.max_delta,
            device=device,
            num_rounds=cfg.num_rounds,
            num_blocks=cfg.num_blocks,
        )

        messages = data["messages"]
        positions = data["positions"]
        deltas = data["deltas"]
        hash_changes = data["hash_changes"]
        targets = compute_jacobian_targets(messages, positions, deltas, hash_changes)

        ste_jac, intermediates = _compute_features(messages, cfg, device)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            J = model(messages, ste_jacobian=ste_jac, intermediates=intermediates)

        B, K = positions.shape
        pos_idx = positions.unsqueeze(-1).expand(B, K, 16)
        J_pred = torch.gather(J, 1, pos_idx)

        predicted_change = J_pred * deltas.float().unsqueeze(-1)
        actual_change = hash_changes.float()

        cos_sim = F.cosine_similarity(
            predicted_change.reshape(-1, 16),
            actual_change.reshape(-1, 16),
            dim=-1,
        )

        total_cos += cos_sim.sum().item()
        total_positive += (cos_sim > 0).sum().item()
        total_count += cos_sim.numel()

    mean_cos = total_cos / total_count
    pct_positive = total_positive / total_count * 100.0

    return {
        "mean_cosine": mean_cos,
        "pct_positive": pct_positive,
    }


def train(cfg: JacobianConfig):
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")
    device_type = device.split(":")[0]

    torch.manual_seed(cfg.seed)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map.get(cfg.dtype, torch.float32)
    use_amp = amp_dtype != torch.float32

    model = JacobianNet(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        use_ste_features=cfg.use_ste_features,
        use_intermediate_features=cfg.use_intermediate_features,
        n_snapshots=cfg.n_snapshots,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    features = []
    if cfg.use_ste_features:
        features.append(f"ste({cfg.ste_mode})")
    if cfg.use_intermediate_features:
        features.append(f"intermediates({cfg.n_snapshots})")
    mode = " + ".join(features) if features else "scratch"

    print(f"JacobianNet parameters: {param_count:,}")
    print(f"Mode: {mode}")
    print(f"MD5: {cfg.num_rounds} rounds, {cfg.num_blocks} blocks")
    print(f"Loss: {cfg.loss_fn}")
    print(f"Config: bs={cfg.batch_size}, lr={cfg.lr}, layers={cfg.n_layers}, "
          f"d_model={cfg.d_model}, dtype={cfg.dtype}, device={device}")
    print(f"Perturbations: {cfg.perturbations_per_msg}/msg, max_delta={cfg.max_delta}")

    # Log target statistics from one batch
    with torch.no_grad():
        sample_data = generate_jacobian_batch(
            min(cfg.batch_size, 64), cfg.perturbations_per_msg,
            cfg.max_delta, device, cfg.num_rounds, cfg.num_blocks,
        )
        sample_targets = compute_jacobian_targets(
            sample_data["messages"], sample_data["positions"],
            sample_data["deltas"], sample_data["hash_changes"],
        )
        t_mean = sample_targets.abs().mean().item()
        t_std = sample_targets.std().item()
        t_max = sample_targets.abs().max().item()
        nonzero_frac = (sample_data["hash_changes"] != 0).float().mean().item()
        print(f"Target stats: mean_abs={t_mean:.2f}, std={t_std:.2f}, "
              f"max_abs={t_max:.1f}, nonzero_frac={nonzero_frac:.3f}")

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed, continuing without: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_schedule(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"Training for {cfg.max_steps} steps, batch_size={cfg.batch_size}")
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        model.train()

        data = generate_jacobian_batch(
            batch_size=cfg.batch_size,
            perturbations_per_msg=cfg.perturbations_per_msg,
            max_delta=cfg.max_delta,
            device=device,
            num_rounds=cfg.num_rounds,
            num_blocks=cfg.num_blocks,
        )

        messages = data["messages"]
        positions = data["positions"]
        deltas = data["deltas"]
        hash_changes = data["hash_changes"]

        ste_jac, intermediates = _compute_features(messages, cfg, device)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            J = model(messages, ste_jacobian=ste_jac, intermediates=intermediates)

            B, K = positions.shape
            pos_idx = positions.unsqueeze(-1).expand(B, K, 16)
            J_pred = torch.gather(J, 1, pos_idx)

            predicted_change = J_pred * deltas.float().unsqueeze(-1)
            actual_change = hash_changes.float()

            if cfg.loss_fn == "cosine":
                # Negative cosine similarity (maximize alignment, not magnitude)
                cos = F.cosine_similarity(
                    predicted_change.reshape(-1, 16),
                    actual_change.reshape(-1, 16),
                    dim=-1,
                )
                loss = 1.0 - cos.mean()
            else:
                loss = F.mse_loss(predicted_change, actual_change)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            lr = optimizer.param_groups[0]["lr"]

            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    predicted_change.detach().reshape(-1, 16),
                    actual_change.reshape(-1, 16),
                    dim=-1,
                )
                mean_cos = cos_sim.mean().item()

            print(
                f"step {step:>6d} | loss {loss.item():.4f} | "
                f"cos {mean_cos:.4f} | lr {lr:.2e} | {steps_per_sec:.1f} steps/s"
            )

        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                report = run_validation(model, cfg, device, device_type, amp_dtype, use_amp)
            print(
                f"  METRICS | mean_cosine={report['mean_cosine']:.4f} | "
                f"pct_positive={report['pct_positive']:.1f}%"
            )

        if cfg.save_every > 0 and step % cfg.save_every == 0:
            path = ckpt_dir / f"step_{step}.pt"
            _save_checkpoint(model, optimizer, cfg, step, path)
            print(f"  Saved checkpoint to {path}")

    path = ckpt_dir / "final.pt"
    _save_checkpoint(model, optimizer, cfg, step, path)
    print(f"Saved final checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train learned Jacobian network")
    parser.add_argument("config", nargs="?", default=None, help="Path to TOML config file")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["scratch", "ste", "full"],
                        help="Feature mode: scratch, ste (STE Jacobian), full (STE + intermediates)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-rounds", type=int, default=None,
                        help="MD5 rounds per block (1-64, default 64)")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["mse", "cosine"], help="Loss function")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.config:
        cfg = JacobianConfig.from_toml(args.config)
    else:
        cfg = JacobianConfig()

    # Apply mode shortcut
    if args.mode == "ste":
        cfg = cfg.override(use_ste_features=True, use_intermediate_features=False)
    elif args.mode == "full":
        cfg = cfg.override(use_ste_features=True, use_intermediate_features=True)
    elif args.mode == "scratch":
        cfg = cfg.override(use_ste_features=False, use_intermediate_features=False)

    cfg = cfg.override(
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        device=args.device,
        num_rounds=args.num_rounds,
        loss_fn=args.loss,
        compile=False if args.no_compile else None,
    )

    train(cfg)


if __name__ == "__main__":
    main()
