from dataclasses import dataclass, fields
from pathlib import Path
import tomllib


@dataclass
class Config:
    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.0

    # Data
    seq_len: int = 64       # byte-level: 512 bits / 8
    vocab_size: int = 257   # 256 byte values + 1 MASK token
    mask_token: int = 256
    hash_bits: int = 128
    hash_bytes: int = 16

    # Training
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    eval_every: int = 1000
    log_every: int = 100
    val_size: int = 10_000
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10_000

    # Diffusion
    sampling_steps: int = 64
    guidance_scale: float = 0.0  # 0 = pure diffusion, >0 = soft MD5 guidance

    # System
    device: str = "cuda"
    compile: bool = True
    compile_mode: str = "default"
    dtype: str = "bfloat16"
    seed: int = 42
    num_workers: int = 0

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load config from a TOML file. Supports [model], [training],
        [diffusion], and [system] sections, or flat keys."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        # Flatten sections into a single dict
        flat = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        # Only keep keys that match dataclass fields
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in valid}
        return cls(**filtered)

    def override(self, **kwargs) -> "Config":
        """Return a new Config with the given fields overridden.
        None values are skipped."""
        updates = {k: v for k, v in kwargs.items() if v is not None}
        valid = {f.name for f in fields(self)}
        for k in updates:
            if k not in valid:
                raise ValueError(f"Unknown config field: {k}")
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(updates)
        return Config(**current)


@dataclass
class JacobianConfig:
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    use_ste_features: bool = False
    use_intermediate_features: bool = False
    ste_mode: str = "complex"
    n_snapshots: int = 7

    # MD5 variant
    num_rounds: int = 64
    num_blocks: int = 2

    # Training
    batch_size: int = 256
    perturbations_per_msg: int = 8
    max_delta: int = 3
    loss_fn: str = "cosine"  # "mse" or "cosine"
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000
    eval_every: int = 500
    log_every: int = 50
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/jacobian"

    # System
    device: str = "cuda"
    compile: bool = True
    dtype: str = "bfloat16"
    seed: int = 42

    @classmethod
    def from_toml(cls, path: str | Path) -> "JacobianConfig":
        """Load config from a TOML file. Supports [model], [training],
        and [system] sections, or flat keys."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        # Flatten sections into a single dict
        flat = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

        # Only keep keys that match dataclass fields
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in valid}
        return cls(**filtered)

    def override(self, **kwargs) -> "JacobianConfig":
        """Return a new JacobianConfig with the given fields overridden.
        None values are skipped."""
        updates = {k: v for k, v in kwargs.items() if v is not None}
        valid = {f.name for f in fields(self)}
        for k in updates:
            if k not in valid:
                raise ValueError(f"Unknown config field: {k}")
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(updates)
        return JacobianConfig(**current)
