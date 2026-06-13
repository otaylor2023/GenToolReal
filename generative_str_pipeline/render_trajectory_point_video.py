"""Render a brush trajectory one frame per trajectory point, then stitch to MP4.

For each datapoint the trajectory is the start (tool home contact) followed by
the 6 predicted/labeled waypoints (7 points total). We render one frame per
point, progressively revealing the path and highlighting the point added at
that frame. Frames are concatenated into a per-datapoint MP4 so the trajectory
shape can be inspected step-by-step.

Output goes to ``training/verification/<output_subdir>/``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrender
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_str_pipeline.visualize_brush_trajectories import (
    COLOR_CONTACT,
    COLOR_DESTINATION,
    COLOR_MATERIAL,
    COLOR_NORMAL,
    COLOR_SURFACE,
    COLOR_SURFDIR,
    COLOR_TOOL_BODY,
    FONT_PATH_BOLD,
    FONT_PATH_REGULAR,
    OUTPUT_HEIGHT,
    PANEL_HEIGHT,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    _add_mesh,
    _add_translucent_mesh,
    _load_font,
    _look_at,
    _make_arrow,
    _make_destination_surface,
    _make_sphere,
    _make_table,
    _make_trajectory_tube,
)
from training.action_trajectory.dataset import (
    WaypointTrajectorySample,
    load_waypoint_samples,
)

COLOR_PATH_DONE = (150, 150, 175, 255)
COLOR_HIGHLIGHT = (255, 235, 90, 255)
COLOR_START = COLOR_TOOL_BODY


def _trajectory_points(sample: WaypointTrajectorySample) -> list[np.ndarray]:
    """Start contact followed by the 6 waypoint contacts (7 points)."""
    pts = [np.asarray(sample.tool_contact_xyz_world, dtype=np.float64).reshape(3)]
    n_wp = int(sample.waypoints.shape[0])
    for i in range(n_wp):
        pts.append(np.asarray(sample.waypoints[i, 0:3], dtype=np.float64).reshape(3))
    return pts


def _add_static_scene(
    scene: pyrender.Scene,
    sample: WaypointTrajectorySample,
    *,
    draw_destination_surface: bool,
) -> None:
    table_z = float(sample.table_xyz_world[2])
    _add_mesh(scene, _make_table(table_z))

    if (
        draw_destination_surface
        and sample.has_destination
        and sample.destination_xyz_world is not None
    ):
        dest_normal = sample.destination_normal
        if dest_normal is None:
            dest_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _add_translucent_mesh(
            scene,
            _make_destination_surface(
                np.asarray(sample.destination_xyz_world, dtype=np.float64),
                np.asarray(dest_normal, dtype=np.float64),
            ),
            COLOR_SURFACE,
        )

    if sample.has_material and sample.material_xyz_world is not None:
        _add_mesh(
            scene,
            _make_sphere(
                np.asarray(sample.material_xyz_world, dtype=np.float64), 0.024, COLOR_MATERIAL
            ),
        )

    if sample.has_destination and sample.destination_xyz_world is not None:
        _add_mesh(
            scene,
            _make_sphere(
                np.asarray(sample.destination_xyz_world, dtype=np.float64),
                0.024,
                COLOR_DESTINATION,
            ),
        )

    # Tool home pose marker (also point 0 of the trajectory).
    _add_mesh(
        scene,
        _make_sphere(
            np.asarray(sample.tool_contact_xyz_world, dtype=np.float64), 0.018, COLOR_START
        ),
    )


def _add_camera_and_lights(scene: pyrender.Scene) -> None:
    target = np.array([0.0, 0.0, 0.62], dtype=np.float64)
    eye = target + np.array([0.35, -0.55, 0.30], dtype=np.float64)
    cam_pose = _look_at(eye, target)
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(45.0),
        aspectRatio=float(RENDER_WIDTH) / float(RENDER_HEIGHT),
    )
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)
    fill_pose = _look_at(target + np.array([0.0, 0.0, 1.5]), target)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill_pose)


def build_progressive_scene(
    sample: WaypointTrajectorySample,
    points: list[np.ndarray],
    up_to_idx: int,
    *,
    draw_destination_surface: bool = True,
) -> pyrender.Scene:
    """Scene revealing trajectory points 0..up_to_idx (inclusive)."""
    scene = pyrender.Scene(bg_color=[26, 30, 46, 255], ambient_light=[0.4, 0.4, 0.4])
    _add_static_scene(scene, sample, draw_destination_surface=draw_destination_surface)

    revealed = points[: up_to_idx + 1]
    if len(revealed) >= 2:
        _add_mesh(scene, _make_trajectory_tube(revealed, color=COLOR_PATH_DONE))

    # Waypoint contacts (point idx >= 1 maps to waypoints[idx-1]).
    for idx in range(1, up_to_idx + 1):
        contact = points[idx]
        is_current = idx == up_to_idx
        color = COLOR_HIGHLIGHT if is_current else COLOR_CONTACT
        radius = 0.018 if is_current else 0.012
        _add_mesh(scene, _make_sphere(contact, radius, color))
        if is_current:
            wp = sample.waypoints[idx - 1]
            normal = np.asarray(wp[3:6], dtype=np.float64)
            surface_dir = np.asarray(wp[6:9], dtype=np.float64)
            _add_mesh(
                scene,
                _make_arrow(contact, normal, COLOR_NORMAL, length=0.07, base_gap=radius),
            )
            _add_mesh(
                scene,
                _make_arrow(
                    contact, surface_dir, COLOR_SURFDIR, length=0.05, base_gap=radius
                ),
            )

    _add_camera_and_lights(scene)
    return scene


def _make_frame_panel(
    sample: WaypointTrajectorySample,
    movement_token: str,
    up_to_idx: int,
    n_points: int,
) -> Image.Image:
    panel = Image.new("RGB", (RENDER_WIDTH, PANEL_HEIGHT), color=(20, 24, 40))
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(FONT_PATH_BOLD, 26)
    body_font = _load_font(FONT_PATH_REGULAR, 20)

    text_x = 24
    if up_to_idx == 0:
        point_label = "point 0 / {} (start / tool home)".format(n_points - 1)
    else:
        point_label = "point {} / {} (waypoint {})".format(
            up_to_idx, n_points - 1, up_to_idx - 1
        )
    draw.text(
        (text_x, 16),
        f"{movement_token}   dp {sample.datapoint_index}   {point_label}",
        fill=(220, 220, 240),
        font=title_font,
    )
    draw.text(
        (text_x, 60),
        f'"{sample.instruction}"',
        fill=(190, 190, 210),
        font=body_font,
    )

    pt = np.asarray(
        sample.tool_contact_xyz_world if up_to_idx == 0 else sample.waypoints[up_to_idx - 1, 0:3],
        dtype=np.float64,
    ).reshape(3)
    draw.text(
        (text_x, 96),
        f"xyz = [{pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f}]",
        fill=(160, 160, 180),
        font=body_font,
    )

    # Progress bar across revealed points.
    bar_x0, bar_y0, bar_w, bar_h = text_x, 150, RENDER_WIDTH - 2 * text_x, 18
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(80, 80, 100))
    if n_points > 1:
        filled = int(bar_w * (up_to_idx / (n_points - 1)))
        draw.rectangle(
            [bar_x0, bar_y0, bar_x0 + filled, bar_y0 + bar_h], fill=COLOR_HIGHLIGHT[:3]
        )
    return panel


def render_progressive_frame(
    sample: WaypointTrajectorySample,
    movement_token: str,
    points: list[np.ndarray],
    up_to_idx: int,
    *,
    renderer: pyrender.OffscreenRenderer,
    draw_destination_surface: bool = True,
) -> Image.Image:
    scene = build_progressive_scene(
        sample, points, up_to_idx, draw_destination_surface=draw_destination_surface
    )
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    img = Image.fromarray(color)
    panel = _make_frame_panel(sample, movement_token, up_to_idx, len(points))
    full = Image.new("RGB", (RENDER_WIDTH, OUTPUT_HEIGHT), color=(20, 24, 40))
    full.paste(img, (0, 0))
    full.paste(panel, (0, RENDER_HEIGHT))
    return full


def render_datapoint_video(
    sample: WaypointTrajectorySample,
    movement_token: str,
    out_path: Path,
    *,
    renderer: pyrender.OffscreenRenderer,
    fps: float = 2.0,
    hold_last: int = 2,
    draw_destination_surface: bool = True,
) -> None:
    points = _trajectory_points(sample)
    frames: list[np.ndarray] = []
    for up_to_idx in range(len(points)):
        frame = render_progressive_frame(
            sample,
            movement_token,
            points,
            up_to_idx,
            renderer=renderer,
            draw_destination_surface=draw_destination_surface,
        )
        frames.append(np.asarray(frame))
    # Hold the final (full trajectory) frame a little longer.
    for _ in range(max(0, hold_last)):
        frames.append(frames[-1])

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one frame per trajectory point and concat into MP4."
    )
    parser.add_argument("--shard_path", type=str, required=True, help="Path to *_shard.json")
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="dataset_0011_trajectory_point_videos",
        help="Subdir under training/verification/ for output MP4 files",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=8,
        help="Max datapoints to render (0 = all)",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second")
    parser.add_argument(
        "--also_save_frames",
        action="store_true",
        help="Also save individual PNG frames next to each MP4",
    )
    args = parser.parse_args()

    shard_path = _resolve_path(Path(args.shard_path))
    out_dir = _resolve_path(Path("training/verification") / args.output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_waypoint_samples(shard_path)
    raw = json.loads(shard_path.read_text(encoding="utf-8"))
    movement_tokens = [str(dp.get("movement_token", "")) for dp in raw.get("datapoints", [])]
    if len(movement_tokens) != len(samples):
        raise RuntimeError(
            f"movement_token count {len(movement_tokens)} != sample count {len(samples)}"
        )

    if int(args.max_datapoints) > 0:
        samples = samples[: int(args.max_datapoints)]
        movement_tokens = movement_tokens[: int(args.max_datapoints)]

    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH, viewport_height=RENDER_HEIGHT
    )
    try:
        for sample, mtoken in zip(samples, movement_tokens):
            stem = f"traj_{mtoken or 'move'}_{sample.datapoint_index:06d}"
            out_path = out_dir / f"{stem}.mp4"
            render_datapoint_video(
                sample, mtoken, out_path, renderer=renderer, fps=float(args.fps)
            )
            print(f"Wrote {out_path}")
            if args.also_save_frames:
                frame_dir = out_dir / stem
                frame_dir.mkdir(parents=True, exist_ok=True)
                points = _trajectory_points(sample)
                for up_to_idx in range(len(points)):
                    img = render_progressive_frame(
                        sample, mtoken, points, up_to_idx, renderer=renderer
                    )
                    img.save(frame_dir / f"frame_{up_to_idx:02d}.png")
    finally:
        renderer.delete()

    print(f"Done: {len(samples)} video(s) in {out_dir}")


if __name__ == "__main__":
    main()
