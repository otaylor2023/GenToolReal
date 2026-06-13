"""Run Depth Anything V2 (Hugging Face) over ``capture/rgb`` + ``video_gen/rgb`` → ``mde_raw``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _relative_depth_to_uint16(rel: np.ndarray) -> np.ndarray:
    r = np.asarray(rel, dtype=np.float32)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.zeros(rel.shape, dtype=np.uint16)
    lo, hi = float(np.percentile(r, 2)), float(np.percentile(r, 98))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    u = (np.asarray(rel, dtype=np.float32) - lo) / (hi - lo)
    u = np.clip(u, 0.0, 1.0)
    return (u * 65535.0).astype(np.uint16)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m pose_track.da2_worker <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    if os.environ.get("POSE_TRACK_MOCK_MDE", "").lower() in ("1", "true", "yes"):
        return _mock_mde(run_dir)

    try:
        from transformers import pipeline  # type: ignore
        import torch  # type: ignore
    except ImportError:
        print(
            "Install DA2 deps: pip install 'pose-track[da2]' plus CUDA torch from pytorch.org; "
            "or set POSE_TRACK_MOCK_MDE=1 for tests.",
            file=sys.stderr,
        )
        return 1

    raw_dev = os.environ.get("POSE_TRACK_DA2_DEVICE")
    if raw_dev is not None and str(raw_dev).strip() != "":
        torch_device = int(raw_dev)
    else:
        torch_device = 0 if torch.cuda.is_available() else -1
    model_id = os.environ.get(
        "POSE_TRACK_DA2_MODEL",
        "depth-anything/Depth-Anything-V2-Small-hf",
    )
    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        device=torch_device,
    )

    cap_rgb = run_dir / "capture" / "rgb"
    v_rgb = run_dir / "video_gen" / "rgb"
    out = run_dir / "mde_raw"
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(cap_rgb.glob("frame_*.png")) + sorted(v_rgb.glob("frame_*.png"))
    if not frames:
        print("no RGB frames found", file=sys.stderr)
        return 1

    for fp in frames:
        im = Image.open(fp).convert("RGB")
        pred = pipe(im)["depth"]
        arr = np.array(pred, dtype=np.float32)
        u16 = _relative_depth_to_uint16(arr)
        Image.fromarray(u16).save(out / fp.name)
        print(fp.name, flush=True)
    return 0


def _mock_mde(run_dir: Path) -> int:
    """Deterministic fake MDE for CI (linear ramp)."""
    cap_rgb = run_dir / "capture" / "rgb"
    v_rgb = run_dir / "video_gen" / "rgb"
    out = run_dir / "mde_raw"
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(cap_rgb.glob("frame_*.png")) + sorted(v_rgb.glob("frame_*.png"))
    for fp in frames:
        rgb = np.array(Image.open(fp).convert("RGB"))
        g = np.mean(rgb.astype(np.float32), axis=2)
        u16 = (g / 255.0 * 65535.0).astype(np.uint16)
        Image.fromarray(u16).save(out / fp.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
