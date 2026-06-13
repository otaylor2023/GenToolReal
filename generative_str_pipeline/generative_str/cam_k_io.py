"""Load DexToolBench ``cam_K.txt`` (3×3 intrinsics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cam_k_txt(path: Path) -> np.ndarray:
    """Return float32 array of shape ``(3, 3)``."""
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").replace(",", " ")
    vals = [float(x) for x in text.split() if x.strip()]
    if len(vals) != 9:
        raise ValueError(f"Expected 9 numbers in {path}, got {len(vals)}")
    return np.array(vals, dtype=np.float32).reshape(3, 3)
