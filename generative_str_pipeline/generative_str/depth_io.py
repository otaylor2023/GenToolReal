"""Save monocular depth maps as uint16 PNG for ``mde_align`` (linear in raw values)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def save_mde_depth_png(path: Path, depth: np.ndarray) -> None:
    """Write a single-channel depth map; affine scale in ``mde_align`` recovers metric."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float64).squeeze()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = d - float(np.min(d))
    mx = float(np.max(d))
    if mx < 1e-8:
        u16 = np.zeros(d.shape, dtype=np.uint16)
    else:
        scaled = d / mx * 60000.0 + 1.0
        u16 = np.clip(scaled, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(u16).save(path)


def sorted_frame_stems(rgb_dir: Path) -> List[str]:
    """Return sorted ``frame_XXXX`` stems under ``rgb_dir``."""
    paths = sorted(rgb_dir.glob("frame_*.png"))
    stems: List[str] = []
    for p in paths:
        m = re.match(r"^(frame_\d+)$", p.stem)
        if m:
            stems.append(m.group(1))
    return stems
