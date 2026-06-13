"""Static-camera / consistency metrics for generated video vs frame 0."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def masked_mae(ref: np.ndarray, cur: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if mask is None:
        return float(np.mean(np.abs(ref - cur)))
    m = mask.astype(bool)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(ref[m] - cur[m])))


def frame_consistency_report(
    frame0: Path,
    frames: list[Path],
    mask: Optional[Path] = None,
) -> dict:
    """Mean absolute RGB difference on optional mask (static region)."""
    r0 = load_rgb(frame0)
    mask_arr = None
    if mask is not None and mask.is_file():
        mask_arr = np.array(Image.open(mask).convert("L")) > 127
        if mask_arr.shape[:2] != r0.shape[:2]:
            raise ValueError("Mask shape must match RGB")

    rows = []
    for p in frames:
        rt = load_rgb(p)
        if rt.shape != r0.shape:
            rows.append({"frame": p.name, "mae": float("nan"), "error": "shape_mismatch"})
            continue
        rows.append({"frame": p.name, "mae": masked_mae(r0, rt, mask_arr)})
    return {
        "frame0": str(frame0),
        "mean_mae_over_frames": float(
            np.nanmean([x["mae"] for x in rows if np.isfinite(x["mae"])]) or np.nan
        ),
        "per_frame": rows,
    }
