from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pose_track.vendor.mde_align import align_run_directory, fit_scale_shift


def test_fit_scale_shift_identity() -> None:
    g = np.linspace(0.5, 2.0, 100, dtype=np.float32).reshape(10, 10)
    m = g * 2.0 + 0.1
    s, b, meta = fit_scale_shift(m, g)
    assert abs(s - 0.5) < 1e-3
    assert abs(b - (-0.05)) < 1e-2
    assert meta.num_pixels == 100


def test_align_run_directory(tmp_path: Path) -> None:
    cap = tmp_path / "capture"
    (cap / "rgb").mkdir(parents=True)
    (cap / "depth").mkdir(parents=True)
    (cap / "masks").mkdir(parents=True)
    rgb0 = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb0[:, :] = (128, 64, 32)
    Image.fromarray(rgb0).save(cap / "rgb" / "frame_0000.png")
    gt_mm = (np.ones((4, 4), dtype=np.float32) * 1500.0).astype(np.uint16)
    Image.fromarray(gt_mm).save(cap / "depth" / "frame_0000.png")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(cap / "masks" / "frame_0000.png")
    (cap / "cam_K.txt").write_text("100 0 2\n0 100 2\n0 0 1\n", encoding="utf-8")

    vg = tmp_path / "video_gen" / "rgb"
    vg.mkdir(parents=True)
    Image.fromarray(rgb0).save(vg / "frame_0001.png")

    mde = tmp_path / "mde_raw"
    mde.mkdir(parents=True)
    mde0 = np.ones((4, 4), dtype=np.float32) * 3000
    Image.fromarray(mde0.astype(np.uint16)).save(mde / "frame_0000.png")
    Image.fromarray((mde0 * 1.1).astype(np.uint16)).save(mde / "frame_0001.png")

    aligned = tmp_path / "aligned_rgbd"
    meta = align_run_directory(cap, mde, aligned, mask_path=cap / "masks" / "frame_0000.png")
    assert meta.scale > 0
    assert (aligned / "rgb" / "frame_0000.png").is_file()
    assert (aligned / "rgb" / "frame_0001.png").is_file()
    assert (aligned / "depth" / "frame_0000.png").is_file()
    j = json.loads((tmp_path / "meta" / "mde_alignment.json").read_text(encoding="utf-8"))
    assert "scale" in j
