from __future__ import annotations

import os
import shutil
from pathlib import Path

from pose_track.layout import default_run_dir, ensure_stages, project_dir, repo_root
from pose_track.mask_from_depth import mask_from_depth_png


def _find_any_mp4() -> Path | None:
    root = repo_root()
    globs = [
        root / "videos",
        root / "cosmos_vlm" / "runs",
    ]
    for base in globs:
        if base.is_dir():
            for p in sorted(base.rglob("*.mp4")):
                if "r4p8" in p.name.lower() or "r4p8" in str(p).lower():
                    return p
            for p in sorted(base.rglob("output_*.mp4")):
                return p
    return None


def _copy_gt_from_session(gt_session: Path, run_dir: Path) -> None:
    """Copy ``00_depth_uint16mm.png`` → ``capture/depth/frame_0000.png``, ``cam_K.txt``."""
    depth_src = gt_session / "00_depth_uint16mm.png"
    if not depth_src.is_file():
        depth_src = gt_session / "00_depth.png"
    if depth_src.is_file():
        shutil.copy2(depth_src, run_dir / "capture" / "depth" / "frame_0000.png")
    k_src = gt_session / "cam_K.txt"
    if k_src.is_file():
        shutil.copy2(k_src, run_dir / "capture" / "cam_K.txt")
    rgb_src = gt_session / "00_main.png"
    if rgb_src.is_file():
        shutil.copy2(rgb_src, run_dir / "capture" / "rgb" / "frame_0000.png")


def bootstrap_run(
    run_dir: Path | None = None,
    *,
    gt_session_dir: Path | None = None,
) -> Path:
    rd = (run_dir or default_run_dir()).resolve()
    ensure_stages(rd)
    root = repo_root()

    mesh_src = (
        root
        / "simtoolreal"
        / "assets"
        / "urdf"
        / "dextoolbench"
        / "brush"
        / "blue_brush"
        / "blue_brush.obj"
    )
    if mesh_src.is_file():
        shutil.copy2(mesh_src, rd / "mesh" / "object.obj")

    env_gt = gt_session_dir or (
        Path(os.environ["POSE_TRACK_GT_SESSION_DIR"]).resolve()
        if os.environ.get("POSE_TRACK_GT_SESSION_DIR")
        else None
    )
    if env_gt is not None and env_gt.is_dir():
        _copy_gt_from_session(env_gt, rd)
        dp = rd / "capture" / "depth" / "frame_0000.png"
        if dp.is_file():
            mask_from_depth_png(dp, rd / "capture" / "masks" / "frame_0000.png")

    vid_dst_dir = project_dir() / "assets" / "videos" / "r4p8"
    vid_dst_dir.mkdir(parents=True, exist_ok=True)
    mp4 = _find_any_mp4()
    if mp4 is not None:
        shutil.copy2(mp4, vid_dst_dir / "clip.mp4")
        shutil.copy2(mp4, rd / "meta" / "source_video.mp4")

    (rd / "meta" / "bootstrap.json").write_text(
        '{"note": "see README for POSE_TRACK_GT_SESSION_DIR and SAVE_VLM_GT_DEPTH"}\n',
        encoding="utf-8",
    )
    return rd
