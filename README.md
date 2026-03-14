# MD5 Preimage Generation via Conditional Discrete Diffusion

A byte-level discrete diffusion model that learns to generate plausible MD5
preimages. Given a 128-bit target hash, the model iteratively denoises a
64-byte message block conditioned on that hash.

The interesting part: **learned Jacobian guidance**. A second network predicts
MD5's local sensitivity at each operating point, providing a directional signal
during sampling. The diffusion model provides the prior ("what looks plausible"),
the Jacobian provides the compass ("which direction reduces hash error").

## Architecture

### Diffusion Model (the prior)

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

**Sampling**: Iterative unmasking. Start fully masked, run T forward passes,
unmask the highest-confidence positions each step.

### Learned Jacobian Network (the compass)

A small transformer that predicts how MD5's output changes when the input
changes, at a specific operating point:

```
Input:  x (64-byte message)
Output: J (64 × 16 sensitivity matrix)
        J[i][o] ≈ "how much does hash byte o change when input byte i changes?"
```

Trained on unlimited free data: perturb random messages, observe actual hash
changes. The key metric is **cosine similarity** between predicted and actual
perturbation directions.

### Guided Sampling (combining both)

```python
for t in T → 0:
    score      = diffusion_model(x_noisy, target_hash, t)   # prior
    J          = jacobian_net(x_current)                      # compass
    hash_error = target_hash - MD5(x_current)                 # where we are
    guidance   = J.T @ hash_error                             # which way to go
    x_next     = denoise(score + anneal(t) * guidance)        # combine
```

Strong guidance at high noise (coarse direction), zero guidance at low noise
(the prior handles fine details).

## Why Not Just Differentiate Through MD5?

We tried. MD5 uses ARX (Addition-Rotation-XOR) — two algebraic structures
with no common smooth basis. We implemented a differentiable MD5 with three
STE variants:

| STE Mode | Approach | Gradient quality |
|---|---|---|
| XOR | Treat addition as XOR in backward pass | Decorrelated at 4+ rounds |
| Sinusoidal + dither | Periodic bit extraction with noise | Extends to ~4 rounds |
| Complex quadrature | sin+cos components, never-zero gradient | Cleanest at 1-2 rounds |

**Key finding**: All hand-crafted STEs achieve cosine similarity ~0 against the
true finite-difference gradient on full MD5 (128 rounds). The avalanche property
destroys gradient signal regardless of the backward-pass engineering. Reduced-round
sweep shows gradient-based inversion works at 1-2 rounds, dies by 4.

```bash
# Reproduce the gradient analysis
uv run scripts/gradient_check.py
uv run scripts/round_sweep.py --device cuda:1 --rounds 1,2,3,4,6,8,16,64 --ste all
```

Hence the learned Jacobian: instead of engineering the backward pass, learn it.

## Quickstart

```bash
uv sync

# Phase 1: Train the Jacobian network
uv run scripts/train_jacobian.py configs/jacobian_3090.toml

# Phase 2: Train the diffusion model
uv run scripts/train.py configs/3090.toml

# Phase 3: Guided sampling (after both are trained)
# (coming soon)

# Analysis tools
uv run scripts/gradient_check.py                                      # soft MD5 correctness + gradient quality
uv run scripts/direct_optimize.py --random --device cuda:1            # pure gradient descent baseline
uv run scripts/round_sweep.py --device cuda:1 --ste all              # STE comparison across round counts
uv run scripts/sweep.py configs/sweep/3090_*.toml --device cuda:1    # model sizing sweep
```

## Config System

TOML files with `[model]`, `[training]`, `[diffusion]`, `[system]` sections.
CLI flags override config values.

| Config | Purpose |
|---|---|
| `configs/3090.toml` | Diffusion model training on RTX 3090 |
| `configs/jacobian_3090.toml` | Jacobian network training on RTX 3090 |
| `configs/mac.toml` | Diffusion model on Apple Silicon |
| `configs/debug.toml` | Quick CPU debug run |

## Evaluation Metrics

| Metric | Random baseline | What it tells you |
|---|---|---|
| Exact hash match | ~0 (1/2^128) | Did we actually find a preimage? |
| Bit agreement | 50% | Does the model know anything about MD5? |
| Byte agreement | 0.39% (1/256) | Coarser but more interpretable |
| Jacobian cosine sim | ~0 | Does the compass point roughly right? |

## How It Compares to Shor's Algorithm

|  | Shor's | Ours |
|---|---|---|
| Samples | O(1) via quantum interference | O(T) via iterative denoising |
| Hardware | Fault-tolerant quantum computer | GPU |
| Mechanism | Quantum Fourier transform | Gradient descent + learned compass |
| Status | Needs ~4000 logical qubits for RSA-2048 | Needs a mass batch_size and a prayer |
| Similarity | Both sample from a structured distribution conditioned on the target |  |

The gap is interference vs gradient descent. Same destination, different physics.

## Limitations

This does not break MD5. To be precise:

- MD5 has a 128-bit output space. Random chance of an exact match is 1/2^128 ~=
  2.9 x 10^-39. Our model would need to be cosmologically better than random to
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

## What Success Looks Like

- Jacobian network achieves higher cosine similarity than all hand-crafted STEs.
  ("A poor learned map beats an engineered map.")
- Guided diffusion achieves higher bit agreement than unguided.
  ("A poor map beats no map.")
- The guidance ablation shows diminishing returns as weight increases.
  (That's the interesting science.)

## Dependencies

- PyTorch >= 2.0
- Python >= 3.10
- That's it. MD5 comes from `hashlib` (stdlib) and a pure PyTorch GPU implementation.
