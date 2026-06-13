# SPDX-License-Identifier: Apache-2.0
# Vendored from Generative_STR `generative_str_pipeline/generative_str/mde_align.py`
# (scale monocular depth to metric GT at frame 0). Do not import sibling packages at runtime.

"""Scale monocular depth to metric ground truth at frame 0."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class AlignmentResult:
    scale: float
    shift: float
    num_pixels: int
    rmse_before: float
    rmse_after: float


def _load_depth_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _depth_to_meters(d: np.ndarray) -> np.ndarray:
    """Heuristic: uint16-style mm if max > 100, else assume meters."""
    if d.max() > 100.0:
        return d * 0.001
    return d


def _save_depth_png_meters(path: Path, depth_meters: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.clip(depth_meters, 0.0, None)
    mm = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(mm).save(path)


def fit_scale_shift(
    mde0: np.ndarray,
    gt0_m: np.ndarray,
    mask: Optional[np.ndarray] = None,
    robust: bool = False,
) -> Tuple[float, float, AlignmentResult]:
    """Fit ``s, b`` so ``s * mde0 + b ≈ gt0_m`` (both in meters internally)."""
    if mde0.shape != gt0_m.shape:
        raise ValueError(f"Shape mismatch mde0 {mde0.shape} vs gt0 {gt0_m.shape}")

    if mask is None:
        mask = gt0_m > 1e-6
    else:
        mask = np.asarray(mask, dtype=bool) & (gt0_m > 1e-6)

    m = mde0[mask].reshape(-1)
    g = gt0_m[mask].reshape(-1)
    if m.size < 10:
        raise ValueError(f"Too few valid pixels for alignment: {m.size}")

    A = np.stack([m, np.ones_like(m)], axis=1)
    if not robust:
        sol, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        s, b = float(sol[0]), float(sol[1])
    else:
        s, b = 1.0, 0.0
        for _ in range(5):
            pred = s * m + b
            r = np.abs(pred - g)
            sigma = np.median(r) + 1e-6
            w = np.minimum(1.0, sigma / (r + 1e-6))
            Aw = A * w[:, None]
            gw = g * w
            sol, _, _, _ = np.linalg.lstsq(Aw, gw, rcond=None)
            s, b = float(sol[0]), float(sol[1])

    pred = s * m + b
    rmse_after = float(np.sqrt(np.mean((pred - g) ** 2)))
    rmse_before = float(np.sqrt(np.mean((m - g) ** 2)))

    meta = AlignmentResult(
        scale=s,
        shift=b,
        num_pixels=int(m.size),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
    )
    return s, b, meta


def align_run_directory(
    capture_dir: Path,
    mde_raw_dir: Path,
    aligned_dir: Path,
    *,
    mask_path: Optional[Path] = None,
    robust: bool = False,
    rgb_glob: str = "frame_*.png",
) -> AlignmentResult:
    """Build FoundationPose-style ``aligned_rgbd`` from capture + raw MDE."""
    gt0_path = capture_dir / "depth" / "frame_0000.png"
    mde0_path = mde_raw_dir / "frame_0000.png"
    if not gt0_path.is_file():
        raise FileNotFoundError(gt0_path)
    if not mde0_path.is_file():
        raise FileNotFoundError(f"{mde0_path} missing — run MDE on frame_0000 for alignment")

    gt0_raw = _load_depth_png(gt0_path)
    gt0_m = _depth_to_meters(gt0_raw)
    mde0 = _load_depth_png(mde0_path)
    mask = None
    if mask_path is not None and mask_path.is_file():
        mask = np.array(Image.open(mask_path).convert("L")) > 127

    s, b, meta = fit_scale_shift(mde0, gt0_m, mask=mask, robust=robust)

    aligned_rgb = aligned_dir / "rgb"
    aligned_depth = aligned_dir / "depth"
    aligned_rgb.mkdir(parents=True, exist_ok=True)
    aligned_depth.mkdir(parents=True, exist_ok=True)

    for name in ("cam_K.txt", "T_RC.txt"):
        src = capture_dir / name
        if src.is_file():
            (aligned_dir / name).write_bytes(src.read_bytes())

    for p in sorted((capture_dir / "rgb").glob(rgb_glob)):
        shutil.copy2(p, aligned_rgb / p.name)

    video_gen = capture_dir.parent / "video_gen"
    vrgb = video_gen / "rgb"
    if vrgb.is_dir():
        for p in sorted(vrgb.glob(rgb_glob)):
            shutil.copy2(p, aligned_rgb / p.name)

    _save_depth_png_meters(aligned_depth / "frame_0000.png", gt0_m)

    for mde_p in sorted(mde_raw_dir.glob("frame_*.png")):
        stem = mde_p.stem
        if stem == "frame_0000":
            continue
        raw = _load_depth_png(mde_p)
        aligned_m = s * raw + b
        _save_depth_png_meters(aligned_depth / f"{stem}.png", aligned_m)

    meta_path = aligned_dir.parent / "meta" / "mde_alignment.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "scale": meta.scale,
                "shift": meta.shift,
                "num_pixels": meta.num_pixels,
                "rmse_before": meta.rmse_before,
                "rmse_after": meta.rmse_after,
                "robust": robust,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return meta
