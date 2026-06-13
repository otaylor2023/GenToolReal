"""Reactive closed-loop rollout verification visualizations (two versions).

Version 1 (executed): per scene, concatenate the first ``chunk_size`` waypoints
from each generation's 6-point plan into the full executed trajectory. Render one
frame per executed waypoint as an MP4; material ball position/color (viridis)
updates only at generation boundaries.

Version 2 (plans): per scene, one static PNG per generation showing the full
6-point plan from that generation's observed state.

Output: ``training/verification/<output_subdir>/executed/`` and ``plans/``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import sys
from dataclasses import dataclass
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
    COLOR_NORMAL,
    COLOR_PAN_RIM,
    COLOR_PAN_WALL,
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
    _make_material_mesh,
    _make_pan_rim,
    _make_pan_wall,
    _make_sphere,
    _make_table,
    _make_trajectory_tube,
    _viridis_rgba,
    pan_center_for_sample,
    pan_viz_settings,
)
from training.action_trajectory.dataset import (
    WaypointTrajectorySample,
    load_waypoint_samples,
)

COLOR_PATH_DONE = (150, 150, 175, 255)
COLOR_HIGHLIGHT = (255, 235, 90, 255)
COLOR_START = COLOR_TOOL_BODY


@dataclass
class GenerationMeta:
    scene_index: int
    window_index: int
    rollout_step: int
    material_after_xyz: np.ndarray | None = None
    material_trace_xyz: np.ndarray | None = None
    material_quat_xyzw: np.ndarray | None = None
    material_after_quat_xyzw: np.ndarray | None = None
    material_trace_quat_xyzw: np.ndarray | None = None
    material_size: np.ndarray | None = None


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _group_scenes(
    samples: list[WaypointTrajectorySample],
    raw_datapoints: list[dict],
) -> dict[int, list[tuple[WaypointTrajectorySample, GenerationMeta]]]:
    if len(samples) != len(raw_datapoints):
        raise RuntimeError(
            f"sample count {len(samples)} != raw datapoint count {len(raw_datapoints)}"
        )
    by_scene: dict[int, list[tuple[WaypointTrajectorySample, GenerationMeta]]] = {}
    for sample, dp in zip(samples, raw_datapoints):
        scene_idx = int(dp.get("scene_index", 0))
        after = dp.get("material_xyz_after_world")
        trace = dp.get("material_xyz_executed_world")
        meta = GenerationMeta(
            scene_index=scene_idx,
            window_index=int(dp.get("window_index", 0)),
            rollout_step=int(dp.get("rollout_step", 0)),
            material_after_xyz=(
                np.asarray(after, dtype=np.float64).reshape(3)
                if after is not None
                else None
            ),
            material_trace_xyz=(
                np.asarray(trace, dtype=np.float64).reshape(-1, 3)
                if trace is not None
                else None
            ),
        )
        by_scene.setdefault(scene_idx, []).append((sample, meta))
    for scene_idx in by_scene:
        by_scene[scene_idx].sort(key=lambda x: x[1].window_index)
    return by_scene


def _viridis_for_gen(gen_idx: int, n_gen: int) -> tuple[int, int, int, int]:
    t = float(gen_idx) / float(max(n_gen - 1, 1))
    return _viridis_rgba(t)


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


def _add_static_environment(
    scene: pyrender.Scene,
    sample: WaypointTrajectorySample,
    *,
    material_xyz: np.ndarray | None = None,
    material_quat_xyzw: np.ndarray | None = None,
    material_size: np.ndarray | None = None,
    material_color: tuple[int, int, int, int] | None = None,
    draw_destination_surface: bool = True,
    draw_pan: bool = False,
    pan_center_xy: tuple[float, float] | None = None,
) -> None:
    _ = draw_destination_surface  # surface intentionally not drawn (table only)
    table_z = float(sample.table_xyz_world[2])
    _add_mesh(scene, _make_table(table_z))

    # Handleless pan: translucent wall ring + bright rim circle (radius marker).
    # Only drawn for pan tasks (e.g. spatula flip / spoon pan); off by default so
    # non-pan tasks like the brush sweep don't get a spurious pan.
    if draw_pan:
        if pan_center_xy is None:
            _, pan_center_xy = pan_center_for_sample(sample)
        _add_translucent_mesh(scene, _make_pan_wall(table_z, pan_center_xy), COLOR_PAN_WALL)
        _add_mesh(scene, _make_pan_rim(table_z, pan_center_xy))

    if sample.has_material and sample.material_xyz_world is not None:
        mat = (
            np.asarray(material_xyz, dtype=np.float64).reshape(3)
            if material_xyz is not None
            else np.asarray(sample.material_xyz_world, dtype=np.float64).reshape(3)
        )
        mat_color = material_color if material_color is not None else _viridis_rgba(0.5)
        _add_mesh(
            scene,
            _make_material_mesh(
                mat,
                mat_color,
                quat_xyzw=material_quat_xyzw,
                size=material_size,
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


def _build_executed_points(
    generations: list[tuple[WaypointTrajectorySample, GenerationMeta]],
    *,
    chunk_size: int,
) -> list[np.ndarray]:
    """P0 = gen0 tool_contact, then first chunk_size waypoints per generation."""
    if not generations:
        return []
    points: list[np.ndarray] = [
        np.asarray(generations[0][0].tool_contact_xyz_world, dtype=np.float64).reshape(3)
    ]
    for sample, _ in generations:
        n_take = min(chunk_size, int(sample.waypoints.shape[0]))
        for i in range(n_take):
            points.append(
                np.asarray(sample.waypoints[i, 0:3], dtype=np.float64).reshape(3)
            )
    return points


def _generation_for_frame(frame_idx: int, chunk_size: int) -> int:
    """Map executed frame index to generation index (ball updates at gen boundaries)."""
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // chunk_size


def build_executed_scene(
    generations: list[tuple[WaypointTrajectorySample, GenerationMeta]],
    executed_points: list[np.ndarray],
    up_to_idx: int,
    *,
    chunk_size: int,
    draw_destination_surface: bool = True,
    draw_pan: bool = False,
    pan_center_xy: tuple[float, float] | None = None,
) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[26, 30, 46, 255], ambient_light=[0.4, 0.4, 0.4])
    n_gen = len(generations)
    gen_idx = _generation_for_frame(up_to_idx, chunk_size)
    ref_sample, ref_meta = generations[gen_idx]
    mat_before = np.asarray(ref_sample.material_xyz_world, dtype=np.float64).reshape(3)
    if up_to_idx == 0:
        mat_xyz = mat_before
    else:
        local = (up_to_idx - 1) % chunk_size
        trace = ref_meta.material_trace_xyz
        if trace is not None and len(trace) > 0:
            mat_xyz = trace[min(local, len(trace) - 1)]
        else:
            # Backward-compatible fallback for older shards that only stored the
            # final material position. New shards store per-executed-step values.
            mat_after = (
                ref_meta.material_after_xyz
                if ref_meta.material_after_xyz is not None
                else mat_before
            )
            frac = float(local + 1) / float(chunk_size)
            mat_xyz = (1.0 - frac) * mat_before + frac * mat_after
    mat_color = _viridis_for_gen(gen_idx, n_gen)

    _add_static_environment(
        scene,
        ref_sample,
        material_xyz=mat_xyz,
        material_color=mat_color,
        draw_destination_surface=draw_destination_surface,
        draw_pan=draw_pan,
        pan_center_xy=pan_center_xy,
    )

    revealed = executed_points[: up_to_idx + 1]
    if len(revealed) >= 2:
        seg_colors: list[tuple[int, int, int, int]] = []
        for seg_i in range(len(revealed) - 1):
            g = _generation_for_frame(seg_i + 1, chunk_size)
            seg_colors.append(_viridis_for_gen(g, n_gen))
        _add_mesh(
            scene,
            _make_trajectory_tube(revealed, segment_colors=seg_colors),
        )

    for idx, pt in enumerate(revealed):
        is_current = idx == up_to_idx
        g = _generation_for_frame(idx, chunk_size)
        color = COLOR_HIGHLIGHT if is_current else _viridis_for_gen(g, n_gen)
        radius = 0.018 if is_current else 0.012
        _add_mesh(scene, _make_sphere(pt, radius, color))
        if idx == 0:
            _add_mesh(scene, _make_sphere(pt, 0.014, COLOR_START))

    _add_camera_and_lights(scene)
    return scene


def build_plan_scene(
    sample: WaypointTrajectorySample,
    *,
    gen_idx: int,
    n_gen: int,
    meta: GenerationMeta | None = None,
    draw_destination_surface: bool = True,
    draw_pan: bool = False,
    pan_center_xy: tuple[float, float] | None = None,
) -> pyrender.Scene:
    """Full planned trajectory (all waypoints) from the generation's state."""
    scene = pyrender.Scene(bg_color=[26, 30, 46, 255], ambient_light=[0.4, 0.4, 0.4])
    mat_color = _viridis_for_gen(gen_idx, n_gen)
    _add_static_environment(
        scene,
        sample,
        material_xyz=np.asarray(sample.material_xyz_world, dtype=np.float64).reshape(3),
        material_quat_xyzw=meta.material_quat_xyzw if meta is not None else None,
        material_size=meta.material_size if meta is not None else None,
        material_color=mat_color,
        draw_destination_surface=draw_destination_surface,
        draw_pan=draw_pan,
        pan_center_xy=pan_center_xy,
    )

    start = np.asarray(sample.tool_contact_xyz_world, dtype=np.float64).reshape(3)
    n_wp = int(sample.waypoints.shape[0])
    wp_contacts = [
        np.asarray(sample.waypoints[i, 0:3], dtype=np.float64).reshape(3)
        for i in range(n_wp)
    ]
    path = [start] + wp_contacts
    _add_mesh(scene, _make_trajectory_tube(path, color=COLOR_PATH_DONE))

    _add_mesh(scene, _make_sphere(start, 0.016, COLOR_START))
    draw_arrows = True
    for i, contact in enumerate(wp_contacts):
        _add_mesh(scene, _make_sphere(contact, 0.009, COLOR_CONTACT))
        if draw_arrows:
            normal = np.asarray(sample.waypoints[i, 3:6], dtype=np.float64)
            surface_dir = np.asarray(sample.waypoints[i, 6:9], dtype=np.float64)
            _add_mesh(
                scene,
                _make_arrow(contact, normal, COLOR_NORMAL, length=0.07, base_gap=0.013),
            )
            _add_mesh(
                scene,
                _make_arrow(
                    contact, surface_dir, COLOR_SURFDIR, length=0.05, base_gap=0.013
                ),
            )

    _add_camera_and_lights(scene)
    return scene


def _make_executed_panel(
    generations: list[tuple[WaypointTrajectorySample, GenerationMeta]],
    frame_idx: int,
    n_frames: int,
    *,
    chunk_size: int,
) -> Image.Image:
    panel = Image.new("RGB", (RENDER_WIDTH, PANEL_HEIGHT), color=(20, 24, 40))
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(FONT_PATH_BOLD, 26)
    body_font = _load_font(FONT_PATH_REGULAR, 20)
    small_font = _load_font(FONT_PATH_REGULAR, 16)

    ref_sample = generations[0][0]
    n_gen = len(generations)
    gen_idx = _generation_for_frame(frame_idx, chunk_size)
    meta = generations[gen_idx][1]

    text_x = 24
    draw.text(
        (text_x, 16),
        f"scene {meta.scene_index}   executed step {frame_idx} / {n_frames - 1}   "
        f"gen {gen_idx} / {n_gen - 1}",
        fill=(220, 220, 240),
        font=title_font,
    )
    draw.text(
        (text_x, 60),
        f'"{ref_sample.instruction}"',
        fill=(190, 190, 210),
        font=body_font,
    )
    pt = generations[0][0].tool_contact_xyz_world if frame_idx == 0 else None
    if frame_idx > 0:
        g = _generation_for_frame(frame_idx, chunk_size)
        wp_local = (frame_idx - 1) % chunk_size
        pt = generations[g][0].waypoints[wp_local, 0:3]
    pt_arr = np.asarray(pt, dtype=np.float64).reshape(3)
    draw.text(
        (text_x, 96),
        f"xyz = [{pt_arr[0]:+.3f}, {pt_arr[1]:+.3f}, {pt_arr[2]:+.3f}]   "
        f"rollout_step={meta.rollout_step}",
        fill=(160, 160, 180),
        font=body_font,
    )

    # Viridis legend: generation swatches.
    legend_x = 760
    draw.text((legend_x, 12), "generation (viridis)", fill=(220, 220, 240), font=small_font)
    sw = 18
    for g in range(min(n_gen, 8)):
        col = _viridis_for_gen(g, n_gen)
        draw.rectangle(
            [legend_x + g * (sw + 6), 36, legend_x + g * (sw + 6) + sw, 36 + sw],
            fill=col[:3],
            outline=(80, 80, 100),
        )
    if n_gen > 8:
        draw.text((legend_x + 8 * (sw + 6), 36), f"+{n_gen - 8}", fill=(160, 160, 180), font=small_font)

    bar_x0, bar_y0, bar_w, bar_h = text_x, 150, RENDER_WIDTH - 2 * text_x, 18
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(80, 80, 100))
    if n_frames > 1:
        filled = int(bar_w * (frame_idx / (n_frames - 1)))
        draw.rectangle(
            [bar_x0, bar_y0, bar_x0 + filled, bar_y0 + bar_h],
            fill=_viridis_for_gen(gen_idx, n_gen)[:3],
        )
    return panel


def _make_plan_panel(
    sample: WaypointTrajectorySample,
    meta: GenerationMeta,
    gen_idx: int,
    n_gen: int,
) -> Image.Image:
    panel = Image.new("RGB", (RENDER_WIDTH, PANEL_HEIGHT), color=(20, 24, 40))
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(FONT_PATH_BOLD, 26)
    body_font = _load_font(FONT_PATH_REGULAR, 20)

    text_x = 24
    draw.text(
        (text_x, 16),
        f"scene {meta.scene_index}   gen {gen_idx} / {n_gen - 1}   "
        f"rollout_step={meta.rollout_step}",
        fill=(220, 220, 240),
        font=title_font,
    )
    draw.text(
        (text_x, 60),
        f'"{sample.instruction}"',
        fill=(190, 190, 210),
        font=body_font,
    )
    mat = np.asarray(sample.material_xyz_world, dtype=np.float64).reshape(3)
    n_wp = int(sample.waypoints.shape[0])
    draw.text(
        (text_x, 96),
        f"material xyz = [{mat[0]:+.3f}, {mat[1]:+.3f}, {mat[2]:+.3f}]   "
        f"(full {n_wp}-step plan)",
        fill=(160, 160, 180),
        font=body_font,
    )

    sw = 18
    legend_x = 760
    draw.rectangle(
        [legend_x, 36, legend_x + sw, 36 + sw],
        fill=_viridis_for_gen(gen_idx, n_gen)[:3],
        outline=(80, 80, 100),
    )
    draw.text(
        (legend_x + sw + 8, 36),
        f"gen {gen_idx}",
        fill=(200, 200, 220),
        font=body_font,
    )
    return panel


def _render_scene_to_image(
    scene: pyrender.Scene,
    panel: Image.Image,
    *,
    renderer: pyrender.OffscreenRenderer,
) -> Image.Image:
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    img = Image.fromarray(color)
    full = Image.new("RGB", (RENDER_WIDTH, OUTPUT_HEIGHT), color=(20, 24, 40))
    full.paste(img, (0, 0))
    full.paste(panel, (0, RENDER_HEIGHT))
    return full


def render_executed_video(
    generations: list[tuple[WaypointTrajectorySample, GenerationMeta]],
    out_path: Path,
    *,
    renderer: pyrender.OffscreenRenderer,
    chunk_size: int = 2,
    fps: float = 2.0,
    hold_last: int = 2,
    draw_pan: bool = False,
    pan_center_xy: tuple[float, float] | None = None,
) -> None:
    executed = _build_executed_points(generations, chunk_size=chunk_size)
    if not executed:
        return

    n_gen = len(generations)
    # Cache the right-side full-plan render per generation (static within a gen).
    plan_cache: dict[int, Image.Image] = {}

    def _plan_image(gen_idx: int) -> Image.Image:
        if gen_idx not in plan_cache:
            sample, meta = generations[gen_idx]
            plan_scene = build_plan_scene(
                sample,
                gen_idx=gen_idx,
                n_gen=n_gen,
                meta=meta,
                draw_pan=draw_pan,
                pan_center_xy=pan_center_xy,
            )
            plan_panel = _make_plan_panel(sample, meta, gen_idx, n_gen)
            plan_cache[gen_idx] = _render_scene_to_image(
                plan_scene, plan_panel, renderer=renderer
            )
        return plan_cache[gen_idx]

    frames: list[np.ndarray] = []
    for up_to_idx in range(len(executed)):
        scene = build_executed_scene(
            generations,
            executed,
            up_to_idx,
            chunk_size=chunk_size,
            draw_pan=draw_pan,
            pan_center_xy=pan_center_xy,
        )
        panel = _make_executed_panel(
            generations, up_to_idx, len(executed), chunk_size=chunk_size
        )
        left_img = _render_scene_to_image(scene, panel, renderer=renderer)

        gen_idx = _generation_for_frame(up_to_idx, chunk_size)
        right_img = _plan_image(gen_idx)

        combined = Image.new(
            "RGB", (RENDER_WIDTH * 2, OUTPUT_HEIGHT), color=(20, 24, 40)
        )
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (RENDER_WIDTH, 0))
        frames.append(np.asarray(combined))

    for _ in range(max(0, hold_last)):
        frames.append(frames[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reactive rollout viz: side-by-side executed video "
        "(left) + full planned trajectory (right)."
    )
    parser.add_argument(
        "--shard_path",
        type=str,
        default="training/datasets/dataset_0012_spatula_flip_reactive/shards/"
        "spatula_flip_reactive_0000_shard.json",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="dataset_0012_spatula_flip_reactive_viz",
        help="Subdir under training/verification/",
    )
    parser.add_argument("--num_scenes", type=int, default=4, help="Scenes to render (0=all)")
    parser.add_argument(
        "--scene_ids",
        type=str,
        default="",
        help="Comma-separated scene indices to render (overrides --num_scenes)",
    )
    parser.add_argument("--chunk_size", type=int, default=5, help="Executed waypoints per gen")
    parser.add_argument("--fps", type=float, default=10.0, help="Executed video FPS")
    parser.add_argument(
        "--draw_pan",
        action="store_true",
        help="Draw the handleless pan (wall + rim). Off by default; only enable "
        "for pan tasks like the spatula flip / spoon pan.",
    )
    args = parser.parse_args()

    shard_path = _resolve_path(Path(args.shard_path))
    out_root = _resolve_path(Path("training/verification") / args.output_subdir)
    executed_dir = out_root / "executed"

    samples = load_waypoint_samples(shard_path)
    raw = json.loads(shard_path.read_text(encoding="utf-8"))
    raw_dps = raw.get("datapoints", [])
    by_scene = _group_scenes(samples, raw_dps)

    if args.scene_ids.strip():
        requested = [int(x) for x in args.scene_ids.split(",") if x.strip() != ""]
        scene_ids = [s for s in requested if s in by_scene]
        missing = [s for s in requested if s not in by_scene]
        if missing:
            print(f"Warning: scene ids not in shard, skipping: {missing}")
    else:
        scene_ids = sorted(by_scene.keys())
        if int(args.num_scenes) > 0:
            scene_ids = scene_ids[: int(args.num_scenes)]

    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH, viewport_height=RENDER_HEIGHT
    )
    try:
        for scene_idx in scene_ids:
            gens = by_scene[scene_idx]
            video_path = executed_dir / f"scene_{scene_idx:04d}_executed.mp4"
            render_executed_video(
                gens,
                video_path,
                renderer=renderer,
                chunk_size=int(args.chunk_size),
                fps=float(args.fps),
                draw_pan=bool(args.draw_pan),
            )
            print(f"Wrote {video_path} ({len(gens)} generations)")
    finally:
        renderer.delete()

    print(f"Done: {len(scene_ids)} scene(s) in {out_root}")


if __name__ == "__main__":
    main()
