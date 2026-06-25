# Ethically sourced from https://github.com/xjdr-alt/entropix

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int,
    rot_dim: int = 32,
    interleave: bool = False,
) -> torch.Tensor:
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert num_heads == x.shape[1]

    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    if interleave:
        xq_r = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 0]
        xq_i = x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)[..., 1]
    else:
        d_q = x_rot.shape[-1] // 2
        xq_r, xq_i = x_rot[..., :d_q], x_rot[..., d_q:]

    freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)

    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)

    return torch.cat([xq_out.to(x.dtype), x_pass], dim=-1)


def rerotate_rope(
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    n_kv_heads: int,
    rot_dim: int = 32,
) -> torch.Tensor:
    """Re-rotate cached k vectors when token positions change.

    ``apply_rotary_emb`` produces an *interleaved* layout
    (``[r0, i0, r1, i1, ...]``) via ``stack((r, i), dim=-1).flatten(-2)``.
    This function matches that layout so re-rotated k vectors are consistent
    with what ``apply_rotary_emb`` would have produced at ``new_positions``.

    RoPE is a rotation: k @ R(p_old) @ R(delta) = k @ R(p_new), where
    delta = p_new - p_old (can be negative).

    Args:
        k: (1, n_kv_heads, seq_len, head_dim) — already rotated by old_positions.
        freqs_cis: (max_context, rot_dim//2, 2) — precomputed RoPE table.
        old_positions: (seq_len,) original position IDs.
        new_positions: (seq_len,) target position IDs.
        n_kv_heads: number of KV heads.
        rot_dim: rotary dimension (must match precomputed freqs_cis).

    Returns:
        k re-rotated to new_positions, same shape as input.
    """
    assert rot_dim == freqs_cis.shape[-2] * 2
    assert n_kv_heads == k.shape[1]

    k_rot = k[..., :rot_dim].float()
    k_pass = k[..., rot_dim:]

    # Interleaved layout: pairs are (r, i) at dims (2j, 2j+1).
    d = rot_dim // 2
    k_r = k_rot[..., 0::2]  # (1, H, S, d)
    k_i = k_rot[..., 1::2]  # (1, H, S, d)

    # Base frequencies: w_i = atan2(freqs_cis[1, i, 1], freqs_cis[1, i, 0]).
    w = torch.atan2(freqs_cis[1, :, 1], freqs_cis[1, :, 0])  # (d,)

    delta = (new_positions - old_positions).float()  # (S,)
    angles = delta.unsqueeze(-1) * w.unsqueeze(0)  # (S, d)
    cos_d = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, S, d)
    sin_d = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    k_new_r = k_r * cos_d - k_i * sin_d
    k_new_i = k_r * sin_d + k_i * cos_d
    # Re-interleave: [r0, i0, r1, i1, ...]
    k_out = torch.stack((k_new_r, k_new_i), dim=-1).flatten(-2)
    return torch.cat([k_out.to(k.dtype), k_pass], dim=-1)
