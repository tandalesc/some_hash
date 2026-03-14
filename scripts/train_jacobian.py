#!/usr/bin/env python3
"""Train the learned Jacobian network (Phase 1).

Trains a small transformer to predict MD5's local sensitivity:
how each input byte affects each hash byte at a given operating point.
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
from src.jacobian_data import compute_jacobian_targets, generate_jacobian_batch
from src.jacobian_net import JacobianNet


def _save_checkpoint(model, optimizer, cfg, step, path):
    state = model._orig_mod.state_dict() if cfg.compile else model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step}, path)


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
        )

        messages = data["messages"]
        positions = data["positions"]
        deltas = data["deltas"]
        hash_changes = data["hash_changes"]
        targets = compute_jacobian_targets(messages, positions, deltas, hash_changes)

        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            J = model(messages)  # (B, 64, 16)

        B, K = positions.shape

        # Gather predicted Jacobian rows for perturbed positions
        pos_idx = positions.unsqueeze(-1).expand(B, K, 16)  # (B, K, 16)
        J_pred = torch.gather(J, 1, pos_idx)  # (B, K, 16)

        # Predicted change = J_pred * delta
        predicted_change = J_pred * deltas.float().unsqueeze(-1)
        actual_change = hash_changes.float()

        # Cosine similarity per perturbation
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
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"JacobianNet parameters: {param_count:,}")
    print(f"Config: bs={cfg.batch_size}, lr={cfg.lr}, layers={cfg.n_layers}, "
          f"d_model={cfg.d_model}, dtype={cfg.dtype}, device={device}")
    print(f"Perturbations: {cfg.perturbations_per_msg}/msg, max_delta={cfg.max_delta}")

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed, continuing without: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Linear warmup then cosine decay
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

        # Generate batch of perturbation data
        data = generate_jacobian_batch(
            batch_size=cfg.batch_size,
            perturbations_per_msg=cfg.perturbations_per_msg,
            max_delta=cfg.max_delta,
            device=device,
        )

        messages = data["messages"]
        positions = data["positions"]
        deltas = data["deltas"]
        hash_changes = data["hash_changes"]
        targets = compute_jacobian_targets(messages, positions, deltas, hash_changes)

        # Forward: predict full Jacobian
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            J = model(messages)  # (B, 64, 16)

            B, K = positions.shape

            # Gather predicted Jacobian rows for perturbed positions
            pos_idx = positions.unsqueeze(-1).expand(B, K, 16)  # (B, K, 16)
            J_pred = torch.gather(J, 1, pos_idx)  # (B, K, 16)

            # Predicted change = J_pred * delta
            predicted_change = J_pred * deltas.float().unsqueeze(-1)
            actual_change = hash_changes.float()

            # MSE loss between predicted and actual hash changes
            loss = F.mse_loss(predicted_change, actual_change)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            lr = optimizer.param_groups[0]["lr"]

            # Quick cosine similarity for monitoring
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

        # Validation
        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                report = run_validation(model, cfg, device, device_type, amp_dtype, use_amp)
            print(
                f"  METRICS | mean_cosine={report['mean_cosine']:.4f} | "
                f"pct_positive={report['pct_positive']:.1f}%"
            )

        # Periodic checkpoint
        if cfg.save_every > 0 and step % cfg.save_every == 0:
            path = ckpt_dir / f"step_{step}.pt"
            _save_checkpoint(model, optimizer, cfg, step, path)
            print(f"  Saved checkpoint to {path}")

    # Final save
    path = ckpt_dir / "final.pt"
    _save_checkpoint(model, optimizer, cfg, step, path)
    print(f"Saved final checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train learned Jacobian network")
    parser.add_argument("config", nargs="?", default=None, help="Path to TOML config file")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Load from file or use defaults
    if args.config:
        cfg = JacobianConfig.from_toml(args.config)
    else:
        cfg = JacobianConfig()

    # CLI overrides take precedence
    cfg = cfg.override(
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        device=args.device,
        compile=False if args.no_compile else None,
    )

    train(cfg)


if __name__ == "__main__":
    main()
