"""W&B trajectory overlays using pyrender brush visualizer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.action_trajectory.dataset import WaypointTrajectorySample


_LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_label_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_LABEL_FONT_PATH, size=size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def add_label_banner(
    image: Image.Image,
    label: str,
    *,
    color: tuple[int, int, int] = (40, 60, 90),
    text_color: tuple[int, int, int] = (235, 235, 245),
    height_px: int = 56,
) -> Image.Image:
    """Add a labeled banner at the top of an image."""
    img = image.convert("RGB")
    banner = Image.new("RGB", (img.width, height_px), color=color)
    draw = ImageDraw.Draw(banner)
    font = _load_label_font(28)
    text = str(label).upper()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    tx = max(16, (img.width - tw) // 2)
    ty = max(8, (height_px - th) // 2 - 4)
    draw.text((tx, ty), text, fill=text_color, font=font)
    out = Image.new("RGB", (img.width, img.height + height_px), color=color)
    out.paste(banner, (0, 0))
    out.paste(img, (0, height_px))
    return out


def pick_fixed_viz_indices(samples: Sequence[Any], *, n: int, seed: int) -> list[int]:
    """Deterministic shuffle of row indices into `samples` (stable across epochs)."""
    if not samples or n <= 0:
        return []
    order = sorted(
        range(len(samples)),
        key=lambda i: (
            str(getattr(samples[i], "scene_id", "")),
            str(getattr(samples[i], "shard_path", "")),
            int(getattr(samples[i], "datapoint_index", 0)),
        ),
    )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    return order[: min(int(n), len(order))]


def pick_fixed_viz_indices_by_movement(
    samples: Sequence[Any],
    *,
    per_movement: int,
    seed: int,
) -> list[int]:
    """Pick up to `per_movement` stable examples for each movement_token.

    This keeps W&B qualitative panels balanced, so rare/random split effects do
    not hide a trajectory family such as stroke_sweep.
    """
    if not samples or per_movement <= 0:
        return []
    movement_order = ("stroke_sweep", "paint_dip", "paint_stroke", "scrub", "press")
    grouped: dict[str, list[int]] = {m: [] for m in movement_order}
    for i, sample in enumerate(samples):
        token = str(getattr(sample, "movement_token", ""))
        grouped.setdefault(token, []).append(i)

    rng = np.random.default_rng(int(seed))
    out: list[int] = []
    for token in movement_order:
        indices = grouped.get(token, [])
        indices = sorted(
            indices,
            key=lambda i: (
                str(getattr(samples[i], "scene_id", "")),
                str(getattr(samples[i], "shard_path", "")),
                int(getattr(samples[i], "datapoint_index", 0)),
            ),
        )
        rng.shuffle(indices)
        out.extend(indices[: int(per_movement)])
    return out


def sample_with_waypoints(
    base: WaypointTrajectorySample,
    waypoints: np.ndarray,
) -> WaypointTrajectorySample:
    wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 9)
    return replace(base, waypoints=wp)


def compose_gt_pred_pair(
    gt_image: Image.Image,
    pred_image: Image.Image,
    *,
    gap_px: int = 8,
) -> Image.Image:
    gt = add_label_banner(
        gt_image.convert("RGB"),
        "Ground Truth",
        color=(40, 90, 60),
    )
    pred = add_label_banner(
        pred_image.convert("RGB"),
        "Predicted",
        color=(90, 50, 120),
    )
    w = gt.width + gap_px + pred.width
    h = max(gt.height, pred.height)
    canvas = Image.new("RGB", (w, h), color=(20, 24, 40))
    canvas.paste(gt, (0, 0))
    canvas.paste(pred, (gt.width + gap_px, 0))
    return canvas


def render_trajectory_panel(
    sample: WaypointTrajectorySample,
    movement_token: str,
) -> Image.Image:
    from generative_str_pipeline.visualize_brush_trajectories import render_datapoint

    return render_datapoint(sample, movement_token)


def render_reactive_rollout_video_for_sample(
    sample: WaypointTrajectorySample,
    *,
    out_dir: Path,
    chunk_size: int = 5,
    fps: float = 10.0,
) -> Path | None:
    """Render the dataset reactive rollout video for ``sample``'s full scene.

    The model trains on individual generation datapoints, but the verification
    video needs the surrounding scene generations to show executed chunks and
    boundary perturbations. We recover those generations from the sample's shard.
    """
    from generative_str_pipeline.render_reactive_rollout_viz import (
        RENDER_HEIGHT,
        RENDER_WIDTH,
        _group_scenes,
        render_executed_video,
    )
    from generative_str_pipeline.visualize_brush_trajectories import pan_center_for_sample
    import pyrender
    from training.action_trajectory.dataset import load_waypoint_samples

    shard_path = Path(sample.shard_path)
    if not shard_path.is_absolute():
        shard_path = Path.cwd() / shard_path
    if not shard_path.is_file():
        return None

    raw = json.loads(shard_path.read_text(encoding="utf-8"))
    raw_dps = raw.get("datapoints", [])
    scene_idx: int | None = None
    for dp in raw_dps:
        if int(dp.get("datapoint_index", -1)) == int(sample.datapoint_index):
            scene_idx = int(dp.get("scene_index", 0))
            break
    if scene_idx is None:
        return None

    shard_samples = load_waypoint_samples(shard_path)
    by_scene = _group_scenes(shard_samples, raw_dps)
    generations = by_scene.get(scene_idx)
    if not generations:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"scene_{scene_idx:04d}_dp_{int(sample.datapoint_index):06d}_reactive.mp4"
    )
    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH, viewport_height=RENDER_HEIGHT
    )
    draw_pan, pan_center_xy = pan_center_for_sample(sample)
    try:
        render_executed_video(
            generations,
            out_path,
            renderer=renderer,
            chunk_size=int(chunk_size),
            fps=float(fps),
            draw_pan=draw_pan,
            pan_center_xy=pan_center_xy,
        )
    finally:
        renderer.delete()
    return out_path
