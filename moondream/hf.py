"""High-level loader shim for moondream-snap.

This mirrors the public surface of the legacy ``moondream`` PyPI package
(``from moondream.hf import LATEST_REVISION, Moondream, detect_device``) but
loads weights into the editable model code vendored from the
``vikhyatk/moondream2`` checkpoint (revision :data:`LATEST_REVISION`).

Unlike ``trust_remote_code=True`` (which pulls code into the HF cache and is
not editable), this loader always uses the *installed* (editable) model code
from this package, so hooks / pruning / quantization modifications are picked
up directly.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
from transformers import AutoTokenizer

from .hf_moondream import HfConfig, HfMoondream
from .topv import PruningConfig
from .weights import load_weights_into_model

#: Hugging Face Hub revision (tag) this vendored code is pinned to.
LATEST_REVISION = "2025-06-21"

#: Default hub model id whose weights match this code.
DEFAULT_MODEL_ID = "vikhyatk/moondream2"


def detect_device() -> tuple[torch.device, torch.dtype]:
    """Pick the best available device and a matching dtype.

    Returns ``(device, dtype)``: CUDA/MPS use bfloat16, CPU uses float32.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32


class Moondream(HfMoondream):
    """Drop-in replacement for the legacy ``moondream.Moondream`` class.

    Adds a :meth:`from_pretrained` that downloads the safetensors weights for
    :data:`DEFAULT_MODEL_ID` at :data:`LATEST_REVISION` and loads them into the
    editable :class:`HfMoondream` model code (no ``trust_remote_code``).
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: Optional[str] = LATEST_REVISION,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        pruning_config: Optional[PruningConfig] = None,
        **kwargs,
    ) -> "Moondream":
        # Build the model with the default MoondreamConfig (the hub config.json
        # ships an empty `config` field, so HfConfig falls back to dataclass
        # defaults that match the 2025-06-21 checkpoint).
        model = cls(HfConfig(), pruning_config=pruning_config)
        dtype = torch_dtype if torch_dtype is not None else model.model.vision.pos_emb.dtype

        # Download weights only (code comes from this editable package).
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            model_id, "model.safetensors", revision=revision
        )
        load_weights_into_model(weights_path, model.model)

        model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID, revision: Optional[str] = LATEST_REVISION):
    """Convenience: load the matching tokenizer for the pinned revision."""
    return AutoTokenizer.from_pretrained(model_id, revision=revision)
