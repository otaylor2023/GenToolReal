"""Compare two DexToolBench trajectory JSONs (world-frame goals)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


def _poses_to_array(traj: dict) -> np.ndarray:
    goals: List[List[float]] = traj["goals"]
    return np.array(goals, dtype=np.float64)


def compare_trajectories(path_a: Path, path_b: Path) -> dict:
    with open(path_a, "r", encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        b = json.load(f)
    ga = _poses_to_array(a)
    gb = _poses_to_array(b)
    n = min(len(ga), len(gb))
    if n == 0:
        return {"error": "empty goals", "n": 0}
    ga = ga[:n, :3]
    gb = gb[:n, :3]
    err = np.linalg.norm(ga - gb, axis=1)
    return {
        "n_compared": n,
        "mean_position_error_m": float(np.mean(err)),
        "max_position_error_m": float(np.max(err)),
    }
