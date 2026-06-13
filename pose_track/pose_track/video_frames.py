from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def extract_video_to_rgb_frames(
    video_path: Path,
    out_dir: Path,
    *,
    start_index: int = 1,
    target_hw: Tuple[int, int] | None = None,
) -> int:
    """Write ``frame_{idx:04d}.png`` under ``out_dir`` (indices starting at ``start_index``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise RuntimeError("imageio is required for video decode: pip install imageio imageio-ffmpeg") from e

    count = 0
    for frame in iio.imiter(str(video_path), plugin="pyav"):
        idx = start_index + count
        rgb = np.asarray(frame[:, :, :3])
        if target_hw is not None:
            th, tw = target_hw
            pil = Image.fromarray(rgb).resize((tw, th), Image.Resampling.LANCZOS)
            rgb = np.asarray(pil)
        Image.fromarray(rgb).save(out_dir / f"frame_{idx:04d}.png")
        count += 1
    return count
