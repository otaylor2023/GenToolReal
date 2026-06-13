"""Render the brush mesh at trajectory poses to verify the pose convention (CPU/EGL).

Places the blue_brush mesh at the start pose and at evenly spaced goal poses in a
single scene (opacity ramps from start to end), with the table plane for
reference. Use this to confirm the bristle face lands near the table and points
into it before launching an IsaacGym rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import trimesh

# importing this module sets PYOPENGL_PLATFORM=egl and pulls in pyrender helpers
from generative_str_pipeline.visualize_brush_trajectories import (
    RENDER_HEIGHT,
    RENDER_WIDTH,
    _look_at,
    _make_table,
)
import pyrender
from PIL import Image

from generative_str_pipeline.sim_rollout.waypoint_to_pose import matrix_from_quat_xyzw

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pose_to_matrix(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat_xyzw(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def _load_brush_mesh(control_frame_path: Path) -> trimesh.Trimesh:
    data = json.loads(control_frame_path.read_text(encoding="utf-8"))
    obj_path = Path(data["obj_path"])
    if not obj_path.is_file():
        raise FileNotFoundError(f"Brush mesh not found: {obj_path}")
    mesh = trimesh.load(str(obj_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")
    return mesh


def _add_brush(scene: pyrender.Scene, mesh: trimesh.Trimesh, T: np.ndarray, rgba) -> None:
    m = mesh.copy()
    m.apply_transform(T)
    material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        baseColorFactor=[rgba[0], rgba[1], rgba[2], rgba[3]],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        doubleSided=True,
    )
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False, material=material))


def render_preview(traj_path: Path, control_frame_path: Path, out_path: Path, max_poses: int) -> None:
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    mesh = _load_brush_mesh(control_frame_path)

    start = np.asarray(traj["start_pose"], dtype=np.float64)
    goals = np.asarray(traj["goals"], dtype=np.float64).reshape(-1, 7)

    poses: List[np.ndarray] = [start]
    if goals.shape[0] > 0:
        if goals.shape[0] > max_poses:
            idx = np.linspace(0, goals.shape[0] - 1, max_poses).round().astype(int)
        else:
            idx = np.arange(goals.shape[0])
        poses.extend(goals[i] for i in idx)

    table_z = float(min(start[2], goals[:, 2].min() if goals.shape[0] else start[2]))
    table_z = min(table_z, 0.53)

    scene = pyrender.Scene(bg_color=[26, 30, 46, 255], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(_make_table(table_z), smooth=False))

    n = len(poses)
    for i, pose in enumerate(poses):
        u = i / max(1, n - 1)
        # start = cyan, end = orange; opacity ramps up
        rgba = (0.3 + 0.6 * u, 0.6 - 0.2 * u, 0.9 - 0.7 * u, 0.30 + 0.55 * u)
        _add_brush(scene, mesh, _pose_to_matrix(pose), rgba)

    center = np.array([start[0], start[1], table_z], dtype=np.float64)
    eye = center + np.array([0.55, -0.65, 0.55])
    cam_pose = _look_at(eye, center)
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(45.0), aspectRatio=float(RENDER_WIDTH) / float(RENDER_HEIGHT)
    )
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)
    fill = _look_at(center + np.array([0.0, 0.0, 1.5]), center)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill)

    renderer = pyrender.OffscreenRenderer(viewport_width=RENDER_WIDTH, viewport_height=RENDER_HEIGHT)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(out_path)
    print(f"Wrote {out_path} ({n} brush poses)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview brush mesh at trajectory poses.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument(
        "--control_frame",
        type=str,
        default="generative_str_pipeline/assets/object_control_points/blue_brush.json",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_poses", type=int, default=8)
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    render_preview(_resolve(args.trajectory), _resolve(args.control_frame), _resolve(args.output), int(args.max_poses))


if __name__ == "__main__":
    main()
