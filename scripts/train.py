#!/usr/bin/env python3
"""CLI entrypoint for training."""
import argparse
import sys

sys.path.insert(0, ".")

from src.config import Config
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Train MD5 preimage diffusion model")
    parser.add_argument("config", nargs="?", default=None, help="Path to TOML config file")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    # Load from file or use defaults
    if args.config:
        cfg = Config.from_toml(args.config)
    else:
        cfg = Config()

    # CLI overrides take precedence
    cfg = cfg.override(
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.layers,
        d_model=args.d_model,
        device=args.device,
        compile=False if args.no_compile else None,
    )

    train(cfg)


if __name__ == "__main__":
    main()
