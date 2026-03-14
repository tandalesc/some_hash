#!/usr/bin/env python3
"""Train the MD5 surrogate — learn MD5's forward pass round by round.

Training modes:
  Phase 1 (steps 0 - rollout_start): Pure teacher forcing. Each round gets
    the real intermediate state and learns the local transform.
  Phase 2 (rollout_start - end): Scheduled rollout. Gradually reduce teacher
    forcing injection — every 2 rounds, then 4, then 8, then full autoregressive.
    This teaches the network to handle its own errors.

Usage:
    uv run scripts/train_surrogate.py --device cuda:1 --num-rounds 4
    uv run scripts/train_surrogate.py --device cuda:1 --num-rounds 64 --bit-level
    uv run scripts/train_surrogate.py --device cuda:1 --num-rounds 4 --bit-level --steps 2000
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

from src.md5_gpu import _SHIFT as _SHIFT_LOOKUP
from src.md5_gpu import md5_intermediates
from src.md5_surrogate import (
    MD5Surrogate,
    words_to_state_bits,
    words_to_state_bytes,
)


def _get_rollout_interval(step: int, max_steps: int, num_rounds: int) -> int:
    """Scheduled rollout: how often to inject the real state.

    Returns inject_every: inject real state every N rounds.
    1 = full teacher forcing, num_rounds = full autoregressive.

    Schedule: teacher forcing for first 40% of training, then
    gradually increase rollout interval.
    """
    progress = step / max_steps
    if progress < 0.4:
        return 1  # pure teacher forcing
    elif progress < 0.55:
        return 2
    elif progress < 0.7:
        return 4
    elif progress < 0.85:
        return max(8, num_rounds // 4)
    else:
        return num_rounds  # full autoregressive


def train(
    num_rounds: int = 64,
    d_hidden: int = 256,
    shared_weights: bool = False,
    bit_level: bool = False,
    batch_size: int = 512,
    max_steps: int = 20_000,
    lr: float = 3e-4,
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 5000,
    checkpoint_dir: str = "checkpoints/surrogate",
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    torch.manual_seed(42)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = MD5Surrogate(
        num_rounds=num_rounds, d_hidden=d_hidden,
        shared_weights=shared_weights, bit_level=bit_level,
    ).to(device)

    state_convert = words_to_state_bits if bit_level else words_to_state_bytes
    repr_name = "bit" if bit_level else "byte"
    state_dim = 128 if bit_level else 16
    word_dim = 32 if bit_level else 4

    param_count = sum(p.numel() for p in model.parameters())
    print(f"MD5 Surrogate: {num_rounds} rounds, d_hidden={d_hidden}, "
          f"repr={repr_name}, shared={shared_weights}, params={param_count:,}")
    print(f"Device: {device}, batch_size={batch_size}, lr={lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_schedule(step):
        if step < 500:
            return step / 500
        progress = (step - 500) / max(1, max_steps - 500)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print(f"Training for {max_steps} steps (scheduled rollout)")
    t0 = time.time()

    for step in range(1, max_steps + 1):
        model.train()

        messages = torch.randint(0, 256, (batch_size, 64), dtype=torch.int64, device=device)
        with torch.no_grad():
            real_states = md5_intermediates(messages, num_rounds=num_rounds, num_blocks=1)
            real_converted = [state_convert(s) for s in real_states]

        msg_normalized = messages.float() / 255.0
        msg_words = model._prepare_message(msg_normalized)

        inject_every = _get_rollout_interval(step, max_steps, num_rounds)

        # Run rounds with scheduled teacher forcing
        state = real_converted[0]  # always start from real initial state
        total_loss = torch.tensor(0.0, device=device)

        for i in range(num_rounds):
            # Inject real state at scheduled intervals
            if inject_every > 1 and i > 0 and i % inject_every == 0:
                state = real_converted[i].detach()

            g = model._schedule[i]
            word = msg_words[:, g, :]
            round_info = torch.tensor(
                [i / 64.0, _SHIFT_LOOKUP[i] / 25.0],
                device=device, dtype=torch.float32
            ).unsqueeze(0).expand(batch_size, -1)

            state = model.rounds[i](state, word, round_info)
            target = real_converted[i + 1]
            total_loss = total_loss + F.mse_loss(state, target)

        loss = total_loss / num_rounds

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:>6d} | loss {loss.item():.6f} | "
                  f"inject_every={inject_every} | lr {current_lr:.2e} | "
                  f"{steps_per_sec:.1f} steps/s")

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_msgs = torch.randint(0, 256, (256, 64), dtype=torch.int64, device=device)
                test_real = md5_intermediates(test_msgs, num_rounds=num_rounds, num_blocks=1)
                real_final = state_convert(test_real[-1])

                msg_norm = test_msgs.float() / 255.0
                pred_final = model(msg_norm)
                autoreg_mse = F.mse_loss(pred_final, real_final).item()

                # Per-round teacher-forced MSE
                real_conv_test = [state_convert(s) for s in test_real]
                msg_words_test = model._prepare_message(msg_norm)
                round_mses = []
                for i in range(min(num_rounds, 4)):
                    g = model._schedule[i]
                    word = msg_words_test[:, g, :]
                    ri = torch.tensor(
                        [i / 64.0, _SHIFT_LOOKUP[i] / 25.0],
                        device=device, dtype=torch.float32
                    ).unsqueeze(0).expand(256, -1)
                    pred = model.rounds[i](real_conv_test[i], word, ri)
                    round_mses.append(F.mse_loss(pred, real_conv_test[i + 1]).item())

            # Jacobian quality via autograd
            test_msg = test_msgs[:32]
            msg_f = test_msg.float() / 255.0
            msg_f.requires_grad_(True)
            pred_hash = model(msg_f)
            pred_hash.sum().backward()
            surrogate_grad = msg_f.grad.detach()

            # Finite-difference through real MD5
            base_hash = state_convert(md5_intermediates(test_msg, num_rounds, 1)[-1])
            cos_sims = []
            for byte_idx in range(0, 64, 4):
                perturbed = test_msg.clone()
                perturbed[:, byte_idx] = (perturbed[:, byte_idx] + 1) % 256
                perturbed_hash = state_convert(
                    md5_intermediates(perturbed, num_rounds, 1)[-1])
                fd = (perturbed_hash - base_hash).flatten(1)
                sg = surrogate_grad[:, byte_idx].unsqueeze(-1).expand_as(fd)
                mask = fd.abs().sum(dim=-1) > 0
                if mask.any():
                    cos = F.cosine_similarity(sg[mask], fd[mask], dim=-1).mean().item()
                    cos_sims.append(cos)

            mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0

            round_str = ", ".join(f"{m:.6f}" for m in round_mses)
            print(f"  METRICS | autoreg_mse={autoreg_mse:.6f} | "
                  f"round_mse=[{round_str}] | jacobian_cos={mean_cos:.4f}")

        if save_every > 0 and step % save_every == 0:
            path = ckpt_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "num_rounds": num_rounds,
                         "d_hidden": d_hidden, "bit_level": bit_level,
                         "step": step}, path)
            print(f"  Saved checkpoint to {path}")

    path = ckpt_dir / "final.pt"
    torch.save({"model": model.state_dict(), "num_rounds": num_rounds,
                 "d_hidden": d_hidden, "bit_level": bit_level, "step": step}, path)
    print(f"Saved final checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train MD5 surrogate")
    parser.add_argument("--num-rounds", type=int, default=64)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--shared", action="store_true", help="Share weights across rounds")
    parser.add_argument("--bit-level", action="store_true", help="Use 128-bit state representation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/surrogate")
    args = parser.parse_args()

    train(
        num_rounds=args.num_rounds,
        d_hidden=args.d_hidden,
        shared_weights=args.shared,
        bit_level=args.bit_level,
        batch_size=args.batch_size,
        max_steps=args.steps,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
