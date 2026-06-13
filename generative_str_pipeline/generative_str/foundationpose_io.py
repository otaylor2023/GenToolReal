"""Load and normalize FoundationPose / DexToolBench pose JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Union

PoseList = List[List[float]]


def load_robot_frame_poses(path: Path) -> PoseList:
    """Load robot-frame poses for `process_poses`.

    Accepts:
    - Flat JSON list of [x,y,z,qx,qy,qz,qw] (DexToolBench raw format).
    - FoundationPose dict with ``poses_robot`` (and optionally ``poses_cam``).

    Returns:
        List of 7-float poses in **robot frame** (same convention as SimToolReal data).
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, list):
        return _validate_pose_list(data, path)

    if isinstance(data, dict):
        if "poses_robot" in data:
            return _validate_pose_list(data["poses_robot"], path)
        raise ValueError(
            f"{path}: expected top-level list or dict with 'poses_robot', got dict keys {list(data.keys())}"
        )

    raise ValueError(f"{path}: unsupported JSON root type {type(data)}")


def save_robot_frame_poses(path: Path, poses: PoseList) -> None:
    """Write DexToolBench-compatible flat list."""
    _validate_pose_list(poses, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)


def flatten_foundationpose_export(src: Path, dst: Path) -> None:
    """Convert FoundationPose ``poses.json`` to flat ``poses_robot`` list file."""
    poses = load_robot_frame_poses(src)
    save_robot_frame_poses(dst, poses)


def _validate_pose_list(data: Any, path: Path) -> PoseList:
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{path}: pose list must be non-empty")
    out: PoseList = []
    for i, row in enumerate(data):
        if not isinstance(row, (list, tuple)) or len(row) != 7:
            raise ValueError(
                f"{path}: pose {i} must be length-7 [x,y,z,qx,qy,qz,qw], got {row!r}"
            )
        out.append([float(x) for x in row])
    return out
