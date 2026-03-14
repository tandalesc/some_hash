import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .data import generate_batch, make_val_set
from .diffusion import compute_loss, mask_message, sample, sample_mask_ratio
from .eval import metrics_report
from .model import Denoiser


def _save_checkpoint(model, optimizer, cfg, step, path):
    state = model._orig_mod.state_dict() if cfg.compile else model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step}, path)


def train(cfg: Config):
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")
    device_type = device.split(":")[0]  # "cuda:1" -> "cuda" for autocast

    torch.manual_seed(cfg.seed)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_dtype = dtype_map.get(cfg.dtype, torch.float32)
    use_amp = amp_dtype != torch.float32

    model = Denoiser(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Config: bs={cfg.batch_size}, lr={cfg.lr}, layers={cfg.n_layers}, "
          f"d_model={cfg.d_model}, dtype={cfg.dtype}, device={device}")

    if cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
            print(f"Model compiled with torch.compile (mode={cfg.compile_mode})")
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
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Fixed validation set
    val_msgs, val_hashes = make_val_set(min(cfg.val_size, 1000))

    print(f"Training for {cfg.max_steps} steps, batch_size={cfg.batch_size}")
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        model.train()

        # Generate batch
        messages, hashes = generate_batch(cfg.batch_size, device=device)

        # Sample mask ratios and apply masking
        mask_ratios = sample_mask_ratio(cfg.batch_size, device=device)
        x_noisy, mask = mask_message(messages, mask_ratios, cfg.mask_token)

        # Forward pass with AMP
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x_noisy, hashes, mask_ratios)
            loss = compute_loss(logits, messages, mask)

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
            print(
                f"step {step:>6d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {steps_per_sec:.1f} steps/s"
            )

        # Evaluation
        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_h = val_hashes[:256].to(device)
                samples = sample(model, val_h, cfg)
                report = metrics_report(samples, val_h)
            print(
                f"  METRICS | exact_match={report['exact_match_rate']:.6f} | "
                f"bit_agree={report['bit_agreement']:.4f} | "
                f"byte_agree={report['byte_agreement']:.4f}"
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
