"""Mixed post-training quantization utilities for moondream-snap.

Current policy:
  - vision encoder/projector: optional INT4/INT8 weight-only
  - text decoder Linear layers: optional INT4/INT8 weight-only
  - embeddings/lm_head: BF16 (unchanged)
  - activations: BF16

No calibration, no training, no weight updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

try:
    from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
except ImportError:  # pragma: no cover
    quantize_ = None
    int4_weight_only = None
    int8_weight_only = None


QuantMode = Literal["none", "int4", "int8"]


@dataclass
class QuantizationConfig:
    mode: QuantMode = "none"
    vision_mode: QuantMode = "none"
    group_size: int = 128

    @property
    def enabled(self) -> bool:
        return self.mode != "none" or self.vision_mode != "none"


def _iter_text_linear_modules(model: nn.Module):
    """Yield only LLM decoder Linear modules, excluding embeddings/lm_head."""
    for block_idx, block in enumerate(model.text.blocks):
        yield f"text.blocks.{block_idx}.attn.qkv", block["attn"]["qkv"]
        yield f"text.blocks.{block_idx}.attn.proj", block["attn"]["proj"]
        yield f"text.blocks.{block_idx}.mlp.fc1", block["mlp"]["fc1"]
        yield f"text.blocks.{block_idx}.mlp.fc2", block["mlp"]["fc2"]


def _iter_vision_linear_modules(model: nn.Module):
    """Yield vision-side Linear modules."""
    yield "vision.patch_emb", model.vision["patch_emb"]
    for block_idx, block in enumerate(model.vision.blocks):
        yield f"vision.blocks.{block_idx}.attn.qkv", block["attn"]["qkv"]
        yield f"vision.blocks.{block_idx}.attn.proj", block["attn"]["proj"]
        yield f"vision.blocks.{block_idx}.mlp.fc1", block["mlp"]["fc1"]
        yield f"vision.blocks.{block_idx}.mlp.fc2", block["mlp"]["fc2"]
    yield "vision.proj_mlp.fc1", model.vision["proj_mlp"]["fc1"]
    yield "vision.proj_mlp.fc2", model.vision["proj_mlp"]["fc2"]


def _quantize_modules(root: nn.Module, iterator, quantizer) -> tuple[int, list[str]]:
    count = 0
    names = []
    for name, module in iterator(root):
        if not isinstance(module, nn.Linear):
            continue
        quantize_(module, quantizer)
        count += 1
        names.append(name)
    return count, names


def apply_mixed_quantization(model: nn.Module, config: QuantizationConfig) -> dict:
    """Apply mixed PTQ to a loaded MoondreamModel in-place."""
    if not config or not config.enabled:
        return {"mode": "none", "vision_mode": "none", "quantized_modules": 0}
    if quantize_ is None:
        raise ImportError("torchao is required for quantization. Install torchao.")

    def make_quantizer(mode: QuantMode):
        if mode == "int4":
            return int4_weight_only(group_size=config.group_size)
        if mode == "int8":
            return int8_weight_only()
        raise ValueError(f"Unsupported quantization mode: {mode}")

    count = 0
    names = []

    if config.mode != "none":
        q = make_quantizer(config.mode)
        c, n = _quantize_modules(model, _iter_text_linear_modules, q)
        count += c
        names.extend(n)

    if config.vision_mode != "none":
        q = make_quantizer(config.vision_mode)
        c, n = _quantize_modules(model, _iter_vision_linear_modules, q)
        count += c
        names.extend(n)

    return {
        "mode": config.mode,
        "vision_mode": config.vision_mode,
        "group_size": config.group_size,
        "quantized_modules": count,
        "first_modules": names[:8],
    }
