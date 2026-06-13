"""Compare aligned MDE depth to dense ground-truth depth (fixture runs)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def _load_m(path: Path) -> np.ndarray:
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    a = a.astype(np.float32)
    if a.max() > 100:
        return a * 0.001
    return a


def depth_rmse(
    pred_dir: Path,
    gt_dir: Path,
    pattern: str = "frame_*.png",
    mask: Optional[Path] = None,
) -> dict:
    """RMSE in meters per frame where GT exists."""
    mask_arr = None
    if mask is not None and mask.is_file():
        mask_arr = np.array(Image.open(mask).convert("L")) > 127

    stats = []
    for gt_p in sorted(gt_dir.glob(pattern)):
        name = gt_p.name
        pr_p = pred_dir / name
        if not pr_p.is_file():
            continue
        g = _load_m(gt_p)
        p = _load_m(pr_p)
        if g.shape != p.shape:
            stats.append({"frame": name, "rmse": float("nan"), "error": "shape"})
            continue
        m = (g > 1e-6) if mask_arr is None else (g > 1e-6) & mask_arr
        if not np.any(m):
            continue
        diff = p[m] - g[m]
        stats.append({"frame": name, "rmse": float(np.sqrt(np.mean(diff**2)))})

    rmses = [s["rmse"] for s in stats if np.isfinite(s["rmse"])]
    return {
        "per_frame": stats,
        "mean_rmse_m": float(np.mean(rmses)) if rmses else float("nan"),
    }
