"""Load and apply global XYZ mean/std normalization (see normalization_stats.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def load_xyz_normalization_stats(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(raw["xyz_mean"], dtype=np.float64)
    std = np.asarray(raw["xyz_std"], dtype=np.float64)
    eps = float(raw.get("norm_eps", 1e-8))
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(f"Invalid xyz_mean/xyz_std shapes in {path}")
    return mean, std, eps


def normalize_xyz_np(xyz: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float) -> np.ndarray:
    return (xyz.astype(np.float64) - mean) / (std + eps)


def denormalize_xyz_torch(
    xyz_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return xyz_norm * (std + eps) + mean
