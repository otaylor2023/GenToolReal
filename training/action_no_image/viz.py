"""W&B / offline overlays: all keypoints + goal-region panel (reuses action_expert eval helpers)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from training.action_no_image.dataset import NoImageActionSample


def _kp_color(i: int) -> tuple[int, int, int]:
    palette = [
        (244, 67, 54),
        (33, 150, 243),
        (76, 175, 80),
        (255, 152, 0),
        (156, 39, 176),
        (0, 188, 212),
        (255, 87, 34),
        (63, 81, 181),
        (139, 195, 74),
        (121, 85, 72),
    ]
    return palette[i % len(palette)]


@lru_cache(maxsize=64)
def _load_shard_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_all_keypoints_with_xyz(shard_path: Path) -> Image.Image:
    """RGB workspace with every valid keypoint: label + xyz_world at uv_px."""
    sp = str(shard_path.resolve())
    shard = _load_shard_json(sp)
    rgb_path = Path(str(shard.get("image", "")))
    if not rgb_path.is_file():
        raise FileNotFoundError(f"Shard image missing: {rgb_path}")
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    keypoints = shard.get("keypoints", {}) or {}
    for idx, (_kp_id, kp) in enumerate(keypoints.items()):
        if not kp.get("valid", False):
            continue
        uv = kp.get("uv_px")
        xyz = kp.get("xyz_world")
        if not isinstance(uv, list) or len(uv) != 2:
            continue
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            continue
        x, y = float(uv[0]), float(uv[1])
        color = _kp_color(idx)
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, fill=color, width=2)
        lbl = str(kp.get("label", "")).strip()
        obj = str(kp.get("object_name", "")).strip()
        head = f"{lbl} of {obj}" if obj else lbl
        line = f"{head} | ({float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f})"
        draw.text((x + 8, y - 10), line, fill=color, font=font)
    return img


def pick_fixed_viz_indices(samples: Sequence[Any], *, n: int, seed: int) -> List[int]:
    """Deterministic shuffle of row indices into `samples` (stable across epochs)."""
    if not samples or n <= 0:
        return []
    order = sorted(
        range(len(samples)),
        key=lambda i: (
            str(getattr(samples[i], "scene_id", "")),
            str(getattr(samples[i], "shard_path", "")),
            int(getattr(samples[i], "datapoint_index", 0)),
            int(getattr(samples[i], "instruction_variant_index", 0)),
        ),
    )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    return order[: min(int(n), len(order))]


def compose_sample_comparison_pair(
    *,
    all_keypoint_image: Image.Image,
    success_region_image: Image.Image,
) -> Image.Image:
    """Return a single sample image with 2 stacked rows.

    Top: all keypoints + xyz.
    Bottom: focused success-region render.
    """
    top = all_keypoint_image.convert("RGB")
    bot = success_region_image.convert("RGB")
    gutter = 10
    w = max(top.width, bot.width)
    out = Image.new("RGB", (w + gutter * 2, top.height + bot.height + gutter * 3), color=(18, 18, 18))
    out.paste(top, (gutter, gutter))
    out.paste(bot, (gutter, top.height + gutter * 2))
    return out


def _scatter_uv(ax: Any, uv: Any, **kwargs: Any) -> None:
    if uv is None:
        return
    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
        ax.scatter(float(uv[0]), float(uv[1]), **kwargs)


def _keypoints_for_region_plot(shard: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _kid, kp in (shard.get("keypoints") or {}).items():
        if not kp.get("valid", False):
            continue
        xyz = kp.get("xyz_world")
        uv = kp.get("uv_px")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            continue
        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
            continue
        out.append(
            {
                "position_xyz_world": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "uv_px": [float(uv[0]), float(uv[1])],
            }
        )
    return out


def render_success_region_panel(
    shard_path: Path,
    sample: Any,
    predicted_xyz_world: np.ndarray | torch.Tensor | Sequence[float],
    region_cfg: Any,
    *,
    sampled_goal_xyz_world: np.ndarray | torch.Tensor | Sequence[float] | None = None,
    depth_eps_m: float = 0.025,
) -> plt.Figure:
    """Focused RGB + goal-region cylinder (eval_fixed_subset style) + tool/ref/pred markers."""
    from training.action_expert.eval_fixed_subset import (  # noqa: PLC0415 — heavy deps
        _build_goal_region_plot,
        _filter_goal_region_plot_for_depth,
        _load_depth_map_npy,
        _project_world_to_uv,
        _resolve_depth_npy_path,
        _resize_depth_to_image,
        _uv_or_fallback,
    )
    from training.action_expert.losses import goal_region_contains  # noqa: PLC0415

    sp = str(shard_path.resolve())
    shard = _load_shard_json(sp)
    cam = shard.get("camera") or {}
    intr_raw = cam.get("intrinsics") or {}
    intr = {
        "fx": float(intr_raw["fx"]),
        "fy": float(intr_raw["fy"]),
        "cx": float(intr_raw["cx"]),
        "cy": float(intr_raw["cy"]),
    }
    w_from_c = np.asarray(cam["world_from_camera"], dtype=np.float64).reshape(4, 4)
    kps_plot = _keypoints_for_region_plot(shard)
    dps = shard.get("datapoints") or []
    di = int(getattr(sample, "datapoint_index", 0))
    if di < 0 or di >= len(dps):
        raise IndexError(f"datapoint_index {di} out of range for shard {sp}")
    dp = dps[di]
    tool_id = str(dp.get("tool_keypoint_id", ""))
    refs = list(dp.get("ref_keypoint_ids") or [])
    keypoints = shard.get("keypoints", {}) or {}

    g_world = np.asarray(getattr(sample, "goal_xyz_world"), dtype=np.float64).reshape(3)
    plot = _build_goal_region_plot(
        g_world.tolist(),
        movement_token=str(getattr(sample, "movement_token", "")),
        constraint_type=str(getattr(sample, "constraint_type", "")),
        constraint_params=dict(getattr(sample, "constraint_params", {}) or {}),
        region_cfg=region_cfg,
        intr=intr,
        w_from_c=w_from_c,
        keypoints=kps_plot,
        n_volume=240,
        n_rim=64,
        n_meridian_z=24,
        n_table_ring=64,
        rng=np.random.default_rng(0),
    )

    rgb_path = Path(str(shard.get("image", "")))
    rgb = plt.imread(str(rgb_path))
    depth_path = _resolve_depth_npy_path(sp, shard)
    depth_map = _load_depth_map_npy(depth_path) if depth_path is not None else None
    if isinstance(depth_map, np.ndarray) and depth_map.ndim == 2:
        depth_map = _resize_depth_to_image(depth_map, rgb)
        plot = _filter_goal_region_plot_for_depth(plot, depth_map, float(depth_eps_m))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb)

    su, sv = plot.get("shading_u"), plot.get("shading_v")
    if isinstance(su, np.ndarray) and isinstance(sv, np.ndarray) and su.size > 8:
        ax.scatter(
            su,
            sv,
            s=3.0,
            c="#00FF88",
            alpha=0.14,
            linewidths=0,
            zorder=2,
            rasterized=True,
            label="goal region (volume projection)",
        )
    tru, trv = plot.get("table_ring_u"), plot.get("table_ring_v")
    if isinstance(tru, np.ndarray) and isinstance(trv, np.ndarray) and tru.size > 4:
        ax.scatter(
            tru,
            trv,
            s=6.5,
            c="#00FF44",
            alpha=0.93,
            linewidths=0,
            edgecolors="none",
            zorder=4,
            rasterized=True,
            label="goal XY @ table (dot ring)",
        )
    for key, color, lw, alpha in (
        ("rim_bottom_uv_px", "lime", 2.1, 0.92),
        ("rim_top_uv_px", "cyan", 2.1, 0.92),
        ("rim_mid_uv_px", "yellow", 1.15, 0.5),
    ):
        du, dv = plot.get(key + "_draw_u"), plot.get(key + "_draw_v")
        if isinstance(du, list) and isinstance(dv, list) and len(du) >= 3:
            ax.plot(du, dv, color=color, linewidth=lw, alpha=alpha, zorder=3)
            continue
        poly = plot.get(key)
        if isinstance(poly, list) and len(poly) >= 3:
            us = [float(p[0]) for p in poly]
            vs = [float(p[1]) for p in poly]
            ax.plot(us, vs, color=color, linewidth=lw, alpha=alpha, zorder=3)
    md = plot.get("_meridian_uv_draw")
    if isinstance(md, list) and md:
        for du, dv in md:
            if isinstance(du, list) and isinstance(dv, list) and len(du) >= 2:
                ax.plot(du, dv, color="white", linewidth=0.85, alpha=0.38, zorder=3)
    elif isinstance(plot.get("meridian_uv_px"), list):
        for seg in plot["meridian_uv_px"]:
            if isinstance(seg, list) and len(seg) >= 2:
                us = [float(p[0]) for p in seg]
                vs = [float(p[1]) for p in seg]
                ax.plot(us, vs, color="white", linewidth=0.85, alpha=0.38, zorder=3)

    def _uv_world(xyz: np.ndarray) -> List[float] | None:
        uv = _project_world_to_uv(xyz, intr, w_from_c)
        if uv is not None:
            return [float(uv[0]), float(uv[1])]
        return _uv_or_fallback(xyz.tolist(), intr, w_from_c, kps_plot)

    tool_uv = None
    if tool_id in keypoints:
        tuv = keypoints[tool_id].get("uv_px")
        if isinstance(tuv, list) and len(tuv) == 2:
            tool_uv = [float(tuv[0]), float(tuv[1])]
    if tool_uv is None:
        tool_uv = _uv_world(np.asarray(getattr(sample, "tool_xyz_world"), dtype=np.float64))

    ref_uv = None
    if refs and refs[0] in keypoints:
        ruv = keypoints[refs[0]].get("uv_px")
        if isinstance(ruv, list) and len(ruv) == 2:
            ref_uv = [float(ruv[0]), float(ruv[1])]
    if ref_uv is None:
        ref_uv = _uv_world(np.asarray(getattr(sample, "ref_xyz_world"), dtype=np.float64))

    sec_uv = None
    if len(refs) > 1 and refs[1] in keypoints:
        suv = keypoints[refs[1]].get("uv_px")
        if isinstance(suv, list) and len(suv) == 2:
            sec_uv = [float(suv[0]), float(suv[1])]
    sec_w = getattr(sample, "secondary_ref_xyz_world", None)
    if sec_uv is None and sec_w is not None:
        sec_uv = _uv_world(np.asarray(sec_w, dtype=np.float64))

    pred = torch.as_tensor(predicted_xyz_world, dtype=torch.float32).reshape(3)
    dataset_g = torch.as_tensor(g_world, dtype=torch.float32).reshape(3)
    sampled_g = (
        torch.as_tensor(sampled_goal_xyz_world, dtype=torch.float32).reshape(3)
        if sampled_goal_xyz_world is not None
        else dataset_g
    )
    pred_uv = _uv_world(pred.detach().cpu().numpy())
    dataset_goal_uv = _uv_world(dataset_g.detach().cpu().numpy())
    sampled_goal_uv = _uv_world(sampled_g.detach().cpu().numpy())

    ok = goal_region_contains(
        pred,
        dataset_g,
        str(getattr(sample, "movement_token", "")),
        str(getattr(sample, "constraint_type", "")),
        dict(getattr(sample, "constraint_params", {}) or {}),
        region_cfg,
    )
    l2_dataset_m = float(torch.linalg.norm(pred - dataset_g).item())
    l2_sampled_m = float(torch.linalg.norm(pred - sampled_g).item())

    _scatter_uv(ax, tool_uv, c="tab:blue", s=170, marker="*", label="tool", zorder=5)
    _scatter_uv(ax, ref_uv, c="tab:orange", s=95, marker="^", label="ref", zorder=4)
    _scatter_uv(ax, sec_uv, c="tab:red", s=95, marker="^", label="ref2", zorder=4)
    _scatter_uv(
        ax,
        dataset_goal_uv,
        c="tab:green",
        s=120,
        marker="x",
        label="dataset goal (region center)",
        zorder=6,
    )
    _scatter_uv(
        ax,
        sampled_goal_uv,
        c="deepskyblue",
        s=120,
        marker="x",
        label="sampled training goal",
        zorder=6,
    )
    _scatter_uv(ax, pred_uv, c="tab:purple", s=120, marker="o", label="pred", zorder=7)

    instr = str(getattr(sample, "instruction", "")).strip()
    head = instr[:80] + ("…" if len(instr) > 80 else "")
    tok = str(getattr(sample, "movement_token", ""))
    status = "OK" if ok else "MISS"
    status_color = "tab:green" if ok else "tab:red"
    ax.text(
        0.02,
        0.98,
        f"{status} | l2_sampled={l2_sampled_m:.3f}m | l2_dataset={l2_dataset_m:.3f}m",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=status_color,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": status_color},
    )
    ax.set_title(
        f"{tok} | {head}",
        fontsize=9,
    )
    ax.set_axis_off()
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
