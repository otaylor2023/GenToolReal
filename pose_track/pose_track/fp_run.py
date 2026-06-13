from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pose_track.layout import repo_root


def run_foundation_pose(
    run_dir: Path,
    *,
    fp_python: str | None = None,
) -> None:
    """Subprocess ``FoundationPose/extract_poses.py`` on ``aligned_rgbd``."""
    root = repo_root()
    fp_py = fp_python or sys.executable
    script = root / "FoundationPose" / "extract_poses.py"
    if not script.is_file():
        raise FileNotFoundError(f"FoundationPose script missing: {script}")

    aligned = run_dir / "aligned_rgbd"
    mesh = run_dir / "mesh" / "object.obj"
    mask = run_dir / "capture" / "masks" / "frame_0000.png"
    out = run_dir / "foundationpose" / "poses.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        fp_py,
        str(script),
        "--video_dir",
        str(aligned),
        "--mesh_path",
        str(mesh),
        "--mask_path",
        str(mask),
        "--output_path",
        str(out),
        "--est_refine_iter",
        "5",
        "--track_refine_iter",
        "2",
        "--debug",
        "0",
    ]
    cal = run_dir / "capture" / "T_RC.npy"
    if cal.is_file():
        cmd += ["--calibration", str(cal)]
    subprocess.run(cmd, check=True)
