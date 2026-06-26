"""moondream-snap: editable moondream2 (revision 2025-06-21) for snap-vlm research.

This package vendors the model code shipped alongside the
`vikhyatk/moondream2` checkpoint (revision 2025-06-21) so that the model
internals can be modified in place (token pruning, quantization, scheduling).

Use :mod:`moondream.hf` for the high-level ``from_pretrained`` entry point
that mirrors the original ``moondream`` PyPI shim.
"""

from .hf import LATEST_REVISION, Moondream, detect_device
from .quantization import QuantizationConfig
from .topv import PruningConfig

__all__ = [
    "LATEST_REVISION",
    "Moondream",
    "detect_device",
    "PruningConfig",
    "QuantizationConfig",
]
__version__ = "0.1.0"
