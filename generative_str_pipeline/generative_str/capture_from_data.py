"""Build a ``capture/`` bundle from DexToolBench-style ``data/`` tree."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def find_first_task_with_frame_zero(data_root: Path) -> Tuple[Path, Path, Path]:
    """Return ``(rgb_path, depth_path, cam_k_path)`` for lexicographically first ``rgb/frame_0000.png``."""
    candidates = sorted(data_root.rglob("rgb/frame_0000.png"))
    for rgb in candidates:
        task_dir = rgb.parent.parent
        depth = task_dir / "depth" / "frame_0000.png"
        cam_k = task_dir / "cam_K.txt"
        if depth.is_file() and cam_k.is_file():
            return rgb, depth, cam_k
    raise FileNotFoundError(
        f"No task under {data_root} has rgb/frame_0000.png, depth/frame_0000.png, and cam_K.txt"
    )


@dataclass
class CaptureFromDataArgs:
    data_root: Path
    """Repository ``data/`` root (DexToolBench layout)."""

    capture_out: Path
    """Directory to create with ``rgb/``, ``depth/``, ``cam_K.txt``."""

    t_rc_source: Optional[Path] = None
    """Optional 4×4 ``T_RC.txt`` from robot calibration; required for FoundationPose on hardware."""


def materialize_capture(args: CaptureFromDataArgs) -> Path:
    rgb, depth, cam_k = find_first_task_with_frame_zero(args.data_root)
    cap = args.capture_out
    (cap / "rgb").mkdir(parents=True, exist_ok=True)
    (cap / "depth").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rgb, cap / "rgb" / "frame_0000.png")
    shutil.copy2(depth, cap / "depth" / "frame_0000.png")
    shutil.copy2(cam_k, cap / "cam_K.txt")
    if args.t_rc_source is not None:
        if not args.t_rc_source.is_file():
            raise FileNotFoundError(args.t_rc_source)
        shutil.copy2(args.t_rc_source, cap / "T_RC.txt")
    meta = cap.parent / "meta" if cap.name == "capture" else cap / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "source_task.txt").write_text(
        f"rgb={rgb}\ndepth={depth}\ncam_K={cam_k}\n",
        encoding="utf-8",
    )
    return cap
