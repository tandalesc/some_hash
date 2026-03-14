import torch
import torch.nn.functional as F

from .config import Config
from .soft_md5 import SoftMD5, probs_to_soft_bits, bytes_to_bits


def mask_message(
    x: torch.Tensor, mask_ratio: float | torch.Tensor, mask_token: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking to message bytes.

    Args:
        x: (B, L) clean message bytes
        mask_ratio: scalar or (B,) tensor of per-sample mask ratios in [0, 1]
        mask_token: token id for MASK

    Returns:
        x_noisy: (B, L) with some positions replaced by mask_token
        mask: (B, L) bool tensor — True where masked
    """
    B, L = x.shape
    if isinstance(mask_ratio, (int, float)):
        mask_ratio = torch.full((B,), mask_ratio, device=x.device)
    # Per-sample random mask
    rand = torch.rand(B, L, device=x.device)
    mask = rand < mask_ratio.unsqueeze(1)
    x_noisy = x.clone()
    x_noisy[mask] = mask_token
    return x_noisy, mask


def sample_mask_ratio(batch_size: int, device: str = "cpu") -> torch.Tensor:
    """Sample mask ratios from a cosine schedule (biased toward higher masking)."""
    u = torch.rand(batch_size, device=device)
    # Cosine schedule: more weight on higher mask ratios
    return torch.cos(u * (torch.pi / 2)).clamp(0.05, 1.0)


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy loss only on masked positions.

    Args:
        logits: (B, L, 256)
        targets: (B, L) original clean bytes
        mask: (B, L) bool — True where masked
    Returns:
        scalar loss
    """
    # Flatten to (N, 256) and (N,) for masked positions only
    logits_masked = logits[mask]
    targets_masked = targets[mask]
    if logits_masked.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(logits_masked, targets_masked)


@torch.no_grad()
def sample(
    model,
    hash_bytes: torch.Tensor,
    cfg: Config,
    steps: int | None = None,
) -> torch.Tensor:
    """Iterative unmasking sampler.

    Starts from fully masked, progressively unmasks positions with highest
    model confidence over `steps` iterations.

    Args:
        model: the denoiser
        hash_bytes: (B, 16) target hashes
        cfg: config
        steps: number of sampling steps (default: cfg.sampling_steps)

    Returns:
        (B, 64) sampled message bytes
    """
    steps = steps or cfg.sampling_steps
    B = hash_bytes.shape[0]
    device = hash_bytes.device
    L = cfg.seq_len

    # Start fully masked
    x = torch.full((B, L), cfg.mask_token, dtype=torch.long, device=device)
    # Track which positions are still masked
    is_masked = torch.ones(B, L, dtype=torch.bool, device=device)

    for step in range(steps):
        # Current mask ratio (decreasing)
        t_val = 1.0 - step / steps
        t = torch.full((B,), t_val, device=device)

        logits = model(x, hash_bytes, t)
        probs = logits.softmax(dim=-1)

        # Sample from predicted distribution at all positions
        candidates = torch.multinomial(probs.view(-1, 256), 1).view(B, L)
        confidence = probs.max(dim=-1).values

        # How many tokens to unmask this step
        n_to_unmask = max(1, int(L * (1.0 / steps)))
        # Only consider currently masked positions
        confidence[~is_masked] = -1.0

        # Pick top-confidence masked positions to unmask
        _, indices = confidence.topk(min(n_to_unmask, L), dim=-1)
        # Unmask those positions
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(indices)
        x[batch_idx, indices] = candidates[batch_idx, indices]
        is_masked[batch_idx, indices] = False

    # Fill any remaining masked positions (shouldn't happen, but safety)
    if is_masked.any():
        t = torch.zeros(B, device=device)
        logits = model(x, hash_bytes, t)
        final = logits.argmax(dim=-1)
        x[is_masked] = final[is_masked]

    return x


def sample_guided(
    model,
    hash_bytes: torch.Tensor,
    cfg: Config,
    guidance_scale: float = 1.0,
    steps: int | None = None,
) -> torch.Tensor:
    """Iterative unmasking with classifier guidance from differentiable MD5.

    At each step:
    1. Model predicts byte logits (diffusion prior)
    2. Convert to soft bits, forward through differentiable MD5
    3. Compute gradient of hash match loss w.r.t. logits
    4. Nudge logits toward better hash match

    Guidance is annealed: strong at high noise (early steps), zero at low noise
    (late steps), since the approximate gradient is most useful for coarse
    structure and the diffusion prior handles fine details.

    Args:
        model: the denoiser
        hash_bytes: (B, 16) target hashes
        cfg: config
        guidance_scale: strength of guidance (0 = pure diffusion)
        steps: number of sampling steps
    Returns:
        (B, 64) sampled message bytes
    """
    steps = steps or cfg.sampling_steps
    B = hash_bytes.shape[0]
    device = hash_bytes.device
    L = cfg.seq_len

    soft_md5 = SoftMD5().to(device)
    # Target hash as bits: (B, 128)
    target_bits = bytes_to_bits(hash_bytes).to(device)

    x = torch.full((B, L), cfg.mask_token, dtype=torch.long, device=device)
    is_masked = torch.ones(B, L, dtype=torch.bool, device=device)

    for step in range(steps):
        t_val = 1.0 - step / steps
        t = torch.full((B,), t_val, device=device)

        # --- Guidance: compute gradient w.r.t. logits ---
        # Anneal guidance: linear decay, strong early, zero at end
        guidance_weight = guidance_scale * t_val

        if guidance_weight > 0:
            # Need gradients for this part
            with torch.enable_grad():
                logits = model(x, hash_bytes, t)
                logits_for_grad = logits.detach().requires_grad_(True)
                probs = logits_for_grad.softmax(dim=-1)
                soft_bits = probs_to_soft_bits(probs)
                soft_hash = soft_md5(soft_bits)
                hash_loss = ((soft_hash - target_bits) ** 2).sum(dim=-1).mean()
                hash_loss.backward()
                grad = logits_for_grad.grad

            # Guided logits: subtract gradient (gradient points toward increasing loss)
            logits = logits - guidance_weight * grad
        else:
            with torch.no_grad():
                logits = model(x, hash_bytes, t)

        with torch.no_grad():
            probs = logits.softmax(dim=-1)
            candidates = torch.multinomial(probs.view(-1, 256), 1).view(B, L)
            confidence = probs.max(dim=-1).values

            n_to_unmask = max(1, int(L * (1.0 / steps)))
            confidence[~is_masked] = -1.0

            _, indices = confidence.topk(min(n_to_unmask, L), dim=-1)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(indices)
            x[batch_idx, indices] = candidates[batch_idx, indices]
            is_masked[batch_idx, indices] = False

    # Fill remaining
    if is_masked.any():
        with torch.no_grad():
            t = torch.zeros(B, device=device)
            logits = model(x, hash_bytes, t)
            final = logits.argmax(dim=-1)
            x[is_masked] = final[is_masked]

    return x
