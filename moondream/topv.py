"""TopV: training-free visual token pruning via Optimal Transport.

Implements the pruning method from:
  TopV: Compatible Token Pruning with Inference Time Optimization
  for Fast and Low-Memory Multimodal Vision Language Model (Yang et al., 2025)

Key design:
  - Prune once during prefill at LLM layer Li (default 2).
  - Source tokens  = input visual tokens of layer Li.
  - Target tokens  = output of layer Li (block output, analogous to Post-LN).
  - Importance is formulated as an Optimal Transport problem solved by
    Sinkhorn iterations, using a visual-aware cost function combining:
      * feature similarity (L2),
      * relative spatial distance (Gaussian),
      * absolute central distance.
  - After TopK selection, a uniform recovery step re-introduces a subset of
    pruned tokens to prevent visual collapse (important for OCR).
  - The pruned set is fixed for all subsequent layers and decode steps,
    keeping KV cache compact and FlashAttention/SDPA compatible.

This module is pure-PyTorch and introduces no CPU synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class PruningConfig:
    """Configuration for TopV visual token pruning."""

    enabled: bool = False
    # Layer index at which to prune (TopV/FastV convention: Li = 2).
    prune_layer: int = 2
    # Fraction of visual tokens to *keep* (before recovery).
    keep_ratio: float = 0.5
    # Cost function weights (alpha: feature, beta: spatial, gamma: central).
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.01
    # Spatial distance bandwidth.
    sigma: float = 10.0
    # Sinkhorn regularization temperature.
    sinkhorn_eps: float = 0.1
    # Sinkhorn iterations (paper: 3).
    sinkhorn_iters: int = 3
    # Recovery: uniformly sample every `recovery_interval`-th pruned token
    # and re-introduce it. Set to 0 to disable recovery.
    recovery_interval: int = 4


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------


def _spatial_grid(n_tokens: int, device: torch.device) -> torch.Tensor:
    """Return (n_tokens, 2) grid coordinates for a square visual token grid.

    Token i maps to (x, y) = (i % grid_w, i // grid_w) where
    grid_w = grid_h = sqrt(n_tokens).
    """
    grid_w = int(n_tokens**0.5)
    assert grid_w * grid_w == n_tokens, (
        f"visual token count {n_tokens} is not a perfect square; "
        "spatial grid cannot be inferred"
    )
    idx = torch.arange(n_tokens, device=device)
    coords = torch.stack([idx % grid_w, idx // grid_w], dim=-1).float()  # (N, 2)
    return coords


def _normalize_01(x: torch.Tensor, dim=None) -> torch.Tensor:
    """Normalize a tensor to [0, 1] range."""
    x_min = x.min(dim=dim, keepdim=True).values if dim is not None else x.min()
    x_max = x.max(dim=dim, keepdim=True).values if dim is not None else x.max()
    rng = x_max - x_min
    rng = torch.where(rng == 0, torch.ones_like(rng), rng)
    return (x - x_min) / rng


def compute_cost_matrix(
    source: torch.Tensor,
    target: torch.Tensor,
    config: PruningConfig,
) -> torch.Tensor:
    """Build the visual-aware cost matrix Cv ∈ R^{N×N}.

    Args:
        source: (N, D) source visual token features (input to layer Li).
        target: (N, D) target visual token features (output of layer Li).
        config: pruning configuration.

    Returns:
        Cv: (N, N) cost matrix, each factor normalized to [0, 1].
    """
    n = source.shape[0]
    device = source.device

    # --- Feature similarity factor: L2 distance (chunked over targets) ---
    # Cf(i, j) = ||s_i - t_j||^2
    # Compute in chunks to avoid materializing (N, N, D) at once.
    cf = torch.empty(n, n, device=device, dtype=source.dtype)
    chunk = max(1, 8192 // max(1, source.shape[-1]))  # targets per chunk
    for j_start in range(0, n, chunk):
        j_end = min(j_start + chunk, n)
        diff = source.unsqueeze(1) - target[j_start:j_end].unsqueeze(0)  # (N, C, D)
        cf[:, j_start:j_end] = diff.pow(2).sum(dim=-1)  # (N, C)
    cf = _normalize_01(cf)

    # --- Relative spatial distance factor: Gaussian ---
    # Cs(i, j) = 1 - exp(-((x_i - x_j)^2 + (y_i - y_j)^2) / (2*sigma^2))
    coords = _spatial_grid(n, device)  # (N, 2)
    coord_diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)
    sq_dist = coord_diff.pow(2).sum(dim=-1)  # (N, N)
    cs = 1.0 - torch.exp(-sq_dist / (2.0 * config.sigma**2))
    cs = _normalize_01(cs)

    # --- Absolute central distance factor ---
    # Ce(i) = sqrt((x_i - x_c)^2 + (y_i - y_c)^2), only depends on source i.
    center = coords[n // 2]  # (2,) — center token coordinates
    ce = (coords - center).pow(2).sum(dim=-1).sqrt()  # (N,)
    ce = _normalize_01(ce)
    ce = ce.unsqueeze(1).expand(-1, n)  # (N, N) — broadcast over targets

    # --- Combined cost ---
    cv = config.alpha * cf + config.beta * cs + config.gamma * ce
    return cv


# ---------------------------------------------------------------------------
# Sinkhorn solver
# ---------------------------------------------------------------------------


def sinkhorn(
    cost: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    """Solve the OT problem via Sinkhorn iterations.

    Args:
        cost: (N, M) cost matrix.
        p: (N,) source marginal distribution.
        q: (M,) target marginal distribution.
        epsilon: regularization temperature.
        n_iters: number of iterations.

    Returns:
        P: (N, M) optimal transport plan (contribution matrix).
    """
    # Kernel matrix
    log_k = -cost / epsilon
    # Log-domain Sinkhorn for numerical stability
    log_p = torch.log(p + 1e-30)
    log_q = torch.log(q + 1e-30)

    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)

    for _ in range(n_iters):
        # log_v = log_q - logsumexp_j(log_k + log_u)
        log_v = log_q - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0)
        # log_u = log_p - logsumexp_i(log_k + log_v)
        log_u = log_p - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)

    log_p_plan = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    return torch.exp(log_p_plan)


# ---------------------------------------------------------------------------
# Token selection
# ---------------------------------------------------------------------------


def select_tokens(
    importance: torch.Tensor,
    keep_ratio: float,
    recovery_interval: int,
) -> torch.Tensor:
    """Select which visual tokens to keep (TopK + uniform recovery).

    Args:
        importance: (N,) importance scores (higher = more important).
        keep_ratio: fraction of tokens to keep via TopK (before recovery).
        recovery_interval: if > 0, uniformly sample every i-th *pruned* token
            and re-introduce it. This prevents visual collapse.

    Returns:
        keep_indices: (K,) sorted indices of tokens to keep.
    """
    n = importance.shape[0]
    n_keep = max(1, int(n * keep_ratio))

    # TopK selection
    _, topk_indices = torch.topk(importance, n_keep, dim=0)
    topk_mask = torch.zeros(n, dtype=torch.bool, device=importance.device)
    topk_mask[topk_indices] = True

    if recovery_interval > 0:
        # Pruned tokens
        pruned_indices = torch.where(~topk_mask)[0]
        # Uniformly sample from pruned tokens
        recovery = pruned_indices[::recovery_interval]
        topk_mask[recovery] = True

    keep_indices = torch.where(topk_mask)[0]
    return keep_indices


# ---------------------------------------------------------------------------
# High-level pruning entry point
# ---------------------------------------------------------------------------


def compute_token_importance(
    source: torch.Tensor,
    target: torch.Tensor,
    config: PruningConfig,
) -> torch.Tensor:
    """Compute visual token importance via OT/Sinkhorn.

    Args:
        source: (N, D) source visual token features.
        target: (N, D) target visual token features.
        config: pruning configuration.

    Returns:
        importance: (N,) importance scores.
    """
    n = source.shape[0]

    # Cost matrix
    cv = compute_cost_matrix(source, target, config)  # (N, N)

    # Marginal distributions: use token L2 norms as non-uniform marginals
    # so that tokens carrying more information receive higher transport mass.
    p = torch.norm(source.float(), dim=-1)
    p = p / (p.sum() + 1e-30)
    q = torch.norm(target.float(), dim=-1)
    q = q / (q.sum() + 1e-30)

    # Sinkhorn → contribution matrix
    p_plan = sinkhorn(
        cv.float(), p, q, config.sinkhorn_eps, config.sinkhorn_iters
    )  # (N, N)

    # Importance = row-sum of contribution matrix
    importance = p_plan.sum(dim=1)  # (N,)
    return importance


def prune_visual_tokens(
    source: torch.Tensor,
    target: torch.Tensor,
    config: PruningConfig,
) -> torch.Tensor:
    """Determine which visual tokens to keep.

    Args:
        source: (N, D) input visual tokens at layer Li.
        target: (N, D) output visual tokens at layer Li.
        config: pruning configuration.

    Returns:
        keep_indices: (K,) sorted indices of visual tokens to keep.
    """
    importance = compute_token_importance(source, target, config)
    keep_indices = select_tokens(
        importance, config.keep_ratio, config.recovery_interval
    )
    return keep_indices
