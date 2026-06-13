"""Build a coarse binary mask from metric depth (uint16 mm or float meters in PNG)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def mask_from_depth_png(depth_path: Path, out_mask: Path, *, dilate: int = 5) -> None:
    arr = np.array(Image.open(depth_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    d = arr.astype(np.float32)
    if d.max() > 100.0:
        d = d * 0.001
    m = (d > 0.05) & (d < 5.0) & np.isfinite(d)
    m = m.astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        m = cv2.dilate(m, k)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m).save(out_mask)
