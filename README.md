# MD5 Preimage Generation via Conditional Discrete Diffusion

A byte-level discrete diffusion model that learns to generate plausible MD5
preimages. Given a 128-bit target hash, the model iteratively denoises a
64-byte message block conditioned on that hash.

## Architecture

```
Input: 64 masked bytes ──→ Token Embed + Pos Embed
                                    │
                          ┌─────────▼───────────┐
                          │  Transformer Block  │ ×N
                          │  ┌───────────────┐  │
  Hash (16 bytes) ───┐    │  │  adaLN-Zero   │  │
                     ├──→ │  │  Self-Attn    │  │
  Timestep t ────────┘    │  │  adaLN-Zero   │  │
                          │  │  FFN          │  │
                          │  └───────────────┘  │
                          └─────────┬───────────┘
                                    │
Output: 64 × 256 logits ◀──── LayerNorm + Linear
```

**Conditioning**: adaLN-Zero. The MD5 hash and noise level are projected to
per-layer scale, shift, and gate parameters — no cross-attention needed since
the conditioning signal is fixed-size.

**Noise process**: Discrete masking. Random byte positions are replaced with a
`[MASK]` token at training time; the model predicts the original byte values.
Mask ratio follows a cosine schedule biased toward heavier corruption.

**Sampling**: Iterative unmasking. Start fully masked, run T forward passes,
unmask the highest-confidence positions each step.

## Quickstart

```bash
uv sync
# debug run (CPU, tiny model, 100 steps)
uv run scripts/train.py configs/debug.toml
# full training
uv run scripts/train.py configs/3090.toml
# sample preimages
uv run scripts/sample.py --checkpoint checkpoints/3090/final.pt --random
# benchmark
uv run scripts/benchmark.py --checkpoint checkpoints/3090/final.pt --device cuda:1
```

## Config System

TOML files with `[model]`, `[training]`, `[diffusion]`, `[system]` sections.
CLI flags override config values.

```bash
# config file + CLI override
uv run scripts/train.py configs/3090.toml --steps 10000 --batch-size 4096
```

| Config | Target | Batch | Layers | d_model |
|---|---|---|---|---|
| `configs/3090.toml` | RTX 3090 | 2048 | 2 | 256 |
| `configs/mac.toml` | Apple Silicon | 256 | 6 | 256 |
| `configs/debug.toml` | CPU | 32 | 2 | 128 |

## Model Sizing Sweep

```bash
uv run scripts/sweep.py configs/sweep/3090_*.toml --device cuda:1 --steps 5000
```

Runs short training for each config and prints a comparison table.

## Evaluation Metrics

| Metric | Random baseline | What it tells you |
|---|---|---|
| Exact hash match | ~0 (1/2¹²⁸) | Did we actually find a preimage? |
| Bit agreement | 50% | Does the model know anything about MD5? |
| Byte agreement | 0.39% (1/256) | Coarser but more interpretable |

## How It Compares to Shor's Algorithm

|  | Shor's | Ours |
|---|---|---|
| Samples | O(1) via quantum interference | O(T) via iterative denoising |
| Hardware | Fault-tolerant quantum computer | GPU |
| Mechanism | Quantum Fourier transform | Gradient descent |
| Status | Needs ~4000 logical qubits for RSA-2048 | Needs a mass batch_size and a prayer |
| Similarity | Both sample from a structured distribution conditioned on the target |  |

The gap is interference vs gradient descent. Same destination, different physics.

## Limitations

This does not break MD5. To be precise:

- MD5 has a 128-bit output space. Random chance of an exact match is 1/2¹²⁸ ≈
  2.9 × 10⁻³⁹. Our model would need to be cosmologically better than random to
  find a single preimage on purpose.
- The avalanche property of MD5 means flipping one input bit flips ~50% of output
  bits. There is no local gradient signal — you either get the hash right or you
  get noise. The model must memorize or learn global structure, not exploit smooth
  shortcuts.
- Even if bit agreement moves above 50%, that doesn't help find preimages.
  Getting 60% of bits right is not 60% of the way to a preimage — it's
  approximately 0% of the way. Hash functions are all-or-nothing.

What this project *actually* does is measure how much statistical regularity a
neural network can extract from a cryptographic mixing function via brute-force
generative modeling. The answer is likely "not much," and that's the point.

## Dependencies

- PyTorch ≥ 2.0
- Python ≥ 3.10
- That's it. MD5 comes from `hashlib` (stdlib).
