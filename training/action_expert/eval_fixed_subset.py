"""Evaluate one 4-NL scene per token and render diagnostics.

Success and renders use **goal region** only: `goal_region_contains(pred, dataset_goal_xyz_world)`
from `training/action_expert/losses.py` (same neighborhood radii as training fixed eval), not
`satisfies_constraint` (reference geometry).

Renders project the full **3D** tolerance volume: horizontal disk (``xy_r``) × vertical slab
(|dz| ≤ ``z_r``) around the dataset goal — one connected cylinder, not multiple disjoint areas.

If ``shard["depth"]`` resolves to ``depth.npy`` (``distance_to_camera``), goal-region shading and
rim/meridian lines are masked so samples behind the rasterized scene depth are not drawn.
Tool/ref/pred/dataset-goal keypoints are drawn on top and are **not** depth-occluded.

A separate **bright green dotted ring** lies on the inferred tabletop plane (5th percentile of
keypoint Z in the scene, IsaacLab-style ~0.53 m) at the goal XY radius; it uses the same depth test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[misc, assignment]

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.config import load_config
from training.action_expert.dataset import (
    ActionExpertDataset,
    ActionSample,
    action_expert_collate,
    load_action_samples_from_shards,
    split_shards_85_10_5,
)
from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.losses import _goal_region_xy_z_radius, goal_region_contains
from training.action_expert.train_action_expert import _region_cfg, _rollout_prediction, _batch_to_device
from training.action_expert.vlm import PaliGemmaContextEncoder
from training.action_expert.xyz_normalization import denormalize_xyz_torch, load_xyz_normalization_stats
from training.action_expert.model import ActionExpertModel


@lru_cache(maxsize=2048)
def _load_shard_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _camera_distance_m(world_xyz: Sequence[float], world_from_camera: np.ndarray) -> float:
    """Euclidean distance from camera origin to `world_xyz` in camera frame.

    Matches Omniverse / Isaac ``distance_to_camera`` style depth buffers used in ``depth.npy``.
    """
    w_from_c = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)
    c_from_w = np.linalg.inv(w_from_c)
    p_w = np.array(
        [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0],
        dtype=np.float64,
    )
    p_c = (c_from_w @ p_w)[:3]
    return float(np.linalg.norm(p_c))


def _project_world_to_uv(
    xyz_world: np.ndarray,
    camera_intrinsics: Dict[str, Any],
    world_from_camera: np.ndarray,
) -> Tuple[float, float] | None:
    """Project world XYZ to image UV using Isaac/OpenGL camera convention (forward is -Z).

    Matches `training/gemini/gemini_position_dataset.py` overlay projection (not OpenCV +Z).
    """
    xyz = np.asarray(xyz_world, dtype=np.float64).reshape(3)
    w_from_c = np.asarray(world_from_camera, dtype=np.float64).reshape(4, 4)
    c_from_w = np.linalg.inv(w_from_c)
    p_w = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    p_c = c_from_w @ p_w
    d = -float(p_c[2])
    if d <= 1e-8:
        return None
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    u = float((float(p_c[0]) / d) * fx + cx)
    v = float(cy - (float(p_c[1]) / d) * fy)
    return u, v


def _uv_from_keypoints_near_xyz(
    keypoints: List[Dict[str, Any]],
    xyz_world: Sequence[float],
    *,
    tol_m: float = 2e-4,
) -> List[float] | None:
    """If a keypoint in the scene matches `xyz_world`, reuse its shard `uv_px` (robust fallback)."""
    t = np.asarray(xyz_world, dtype=np.float64).reshape(3)
    for kp in keypoints:
        pos = kp.get("position_xyz_world")
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            continue
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(p - t)) <= tol_m:
            uv = kp.get("uv_px")
            if isinstance(uv, (list, tuple)) and len(uv) == 2:
                return [float(uv[0]), float(uv[1])]
    return None


def _build_goal_region_plot(
    dataset_goal_xyz: Sequence[float],
    *,
    movement_token: str,
    constraint_type: str,
    constraint_params: Dict[str, Any],
    region_cfg: Any,
    intr: Dict[str, Any],
    w_from_c: np.ndarray,
    keypoints: List[Dict[str, Any]],
    n_volume: int,
    n_rim: int,
    n_meridian_z: int,
    n_table_ring: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """2D render payload for the full 3D goal region from `goal_region_contains`.

    In world space this is a **single** right cylinder: disk radius ``xy_r`` in the XY plane
    around ``(gx,gy)``, height ``2*z_r`` along Z about ``gz`` (i.e. |dx,dy| in disk, |dz| in slab).
    It is not multiple disjoint areas for this loss.

    The image shows: (1) semi-transparent random interior samples (volume), (2) top/bottom rim
    polylines, (3) optional vertical seam lines on the side surface.
    """
    xy_r, z_r = _goal_region_xy_z_radius(
        movement_token, constraint_type, constraint_params, region_cfg
    )
    g = np.asarray(dataset_goal_xyz, dtype=np.float64).reshape(3)
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])

    def rim(z_plane: float) -> Tuple[List[List[float]] | None, List[float] | None]:
        poly: List[List[float]] = []
        dists: List[float] = []
        for i in range(n_rim + 1):
            th = 2.0 * np.pi * float(i) / float(n_rim)
            pw = (
                gx + float(xy_r) * float(np.cos(th)),
                gy + float(xy_r) * float(np.sin(th)),
                float(z_plane),
            )
            uv = _uv_or_fallback(pw, intr, w_from_c, keypoints)
            if uv is None:
                return None, None
            poly.append(uv)
            dists.append(_camera_distance_m(pw, w_from_c))
        return poly, dists

    rim_bottom, rim_bottom_d = rim(gz - float(z_r))
    rim_top, rim_top_d = rim(gz + float(z_r))
    rim_mid, rim_mid_d = rim(gz)

    us: List[float] = []
    vs: List[float] = []
    ds: List[float] = []
    attempts = 0
    max_attempts = max(5000, n_volume * 12)
    while len(us) < n_volume and attempts < max_attempts:
        attempts += 1
        z = float(rng.uniform(gz - float(z_r), gz + float(z_r)))
        rr = float(np.sqrt(rng.uniform(0.0, 1.0))) * float(xy_r)
        th = float(rng.uniform(0.0, 2.0 * np.pi))
        pw = (gx + rr * float(np.cos(th)), gy + rr * float(np.sin(th)), z)
        uv = _uv_or_fallback(pw, intr, w_from_c, keypoints)
        if uv is not None:
            us.append(float(uv[0]))
            vs.append(float(uv[1]))
            ds.append(_camera_distance_m(pw, w_from_c))

    meridians: List[List[List[float]]] = []
    meridian_dists: List[List[float]] = []
    for k in range(4):
        theta = (0.5 * np.pi) * float(k)
        seg: List[List[float]] = []
        sd: List[float] = []
        ok = True
        for j in range(n_meridian_z + 1):
            z = float(gz - z_r + (2.0 * float(z_r)) * float(j) / float(n_meridian_z))
            pw = (
                gx + float(xy_r) * float(np.cos(theta)),
                gy + float(xy_r) * float(np.sin(theta)),
                z,
            )
            uv = _uv_or_fallback(pw, intr, w_from_c, keypoints)
            if uv is None:
                ok = False
                break
            seg.append(uv)
            sd.append(_camera_distance_m(pw, w_from_c))
        if ok and len(seg) >= 2:
            meridians.append(seg)
            meridian_dists.append(sd)

    table_z = _estimate_table_top_z_m(keypoints)
    ntr = max(8, int(n_table_ring))
    tr_u: List[float] = []
    tr_v: List[float] = []
    tr_d: List[float] = []
    for i in range(ntr + 1):
        th = 2.0 * np.pi * float(i) / float(ntr)
        pw = (
            gx + float(xy_r) * float(np.cos(th)),
            gy + float(xy_r) * float(np.sin(th)),
            float(table_z),
        )
        uv = _uv_or_fallback(pw, intr, w_from_c, keypoints)
        if uv is None:
            continue
        tr_u.append(float(uv[0]))
        tr_v.append(float(uv[1]))
        tr_d.append(_camera_distance_m(pw, w_from_c))

    return {
        "xy_radius_m": float(xy_r),
        "z_half_extent_m": float(z_r),
        "table_top_z_m": float(table_z),
        "table_ring_n_samples": int(ntr),
        "shading_u": np.asarray(us, dtype=np.float64),
        "shading_v": np.asarray(vs, dtype=np.float64),
        "shading_dist_m": np.asarray(ds, dtype=np.float64),
        "table_ring_u": np.asarray(tr_u, dtype=np.float64),
        "table_ring_v": np.asarray(tr_v, dtype=np.float64),
        "table_ring_dist_m": np.asarray(tr_d, dtype=np.float64),
        "rim_bottom_uv_px": rim_bottom,
        "rim_bottom_dist_m": rim_bottom_d,
        "rim_top_uv_px": rim_top,
        "rim_top_dist_m": rim_top_d,
        "rim_mid_uv_px": rim_mid,
        "rim_mid_dist_m": rim_mid_d,
        "meridian_uv_px": meridians,
        "meridian_dist_m": meridian_dists,
    }


def _estimate_table_top_z_m(keypoints: List[Dict[str, Any]]) -> float:
    """Infer tabletop height from keypoint world Z (low percentile ~= resting surfaces).

    IsaacLab tables are often ~0.38–0.55 m depending on scene; shard JSON does not store table_top_z.
    """
    zs: List[float] = []
    for kp in keypoints:
        p = kp.get("position_xyz_world")
        if isinstance(p, (list, tuple)) and len(p) == 3:
            zs.append(float(p[2]))
    if not zs:
        return 0.53
    arr = np.asarray(zs, dtype=np.float64)
    return float(np.percentile(arr, 5.0))


def _goal_region_meta_only(plot: Dict[str, Any]) -> Dict[str, Any]:
    su = plot.get("shading_u")
    n_pts = int(su.shape[0]) if isinstance(su, np.ndarray) else 0
    meta: Dict[str, Any] = {
        "world_shape": "cylinder_xy_disk_z_slab",
        "matches_loss": "goal_region_contains (training/action_expert/losses.py)",
        "xy_radius_m": float(plot["xy_radius_m"]),
        "z_half_extent_m": float(plot["z_half_extent_m"]),
        "n_shading_points_rendered": n_pts,
        "note": "Single connected region; not satisfies_constraint reference geometry.",
    }
    if "table_top_z_m" in plot:
        meta["table_top_z_m"] = float(plot["table_top_z_m"])
    if "table_ring_n_samples" in plot:
        meta["table_ring_n_samples"] = int(plot["table_ring_n_samples"])
    return meta


def _uv_or_fallback(
    xyz_world: Sequence[float],
    intr: Dict[str, Any],
    w_from_c: np.ndarray,
    keypoints: List[Dict[str, Any]],
) -> List[float] | None:
    uv = _project_world_to_uv(np.asarray(xyz_world, dtype=np.float64), intr, w_from_c)
    if uv is not None:
        return [float(uv[0]), float(uv[1])]
    fb = _uv_from_keypoints_near_xyz(keypoints, xyz_world)
    if fb is not None:
        return fb
    return None


def _resolve_depth_npy_path(shard_path: str, shard_obj: Dict[str, Any]) -> Path | None:
    raw = shard_obj.get("depth")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p
    base = Path(shard_path).resolve().parent
    for cand in (base / Path(str(raw)).name, base / str(raw)):
        if cand.is_file():
            return cand
    return None


def _load_depth_map_npy(path: Path) -> np.ndarray | None:
    try:
        arr = np.load(str(path))
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            return None
        return arr
    except OSError:
        return None


def _sample_depth_nn(depth: np.ndarray, u: float, v: float) -> float:
    h, w = int(depth.shape[0]), int(depth.shape[1])
    ui = int(np.clip(round(float(u)), 0, w - 1))
    vi = int(np.clip(round(float(v)), 0, h - 1))
    return float(depth[vi, ui])


def _depth_visible(dist_m: float, depth: np.ndarray, u: float, v: float, eps_m: float) -> bool:
    """True if the world point is not farther than the stored scene depth at this pixel (in front)."""
    ds = _sample_depth_nn(depth, u, v)
    if not np.isfinite(ds) or ds <= 1e-7:
        return True
    return float(dist_m) <= float(ds) + float(eps_m)


def _filter_goal_region_plot_for_depth(
    plot: Dict[str, Any],
    depth: np.ndarray,
    eps_m: float,
) -> Dict[str, Any]:
    """Drop occluded samples / insert NaNs in polylines. Keypoints are NOT passed here."""
    out = dict(plot)
    su, sv = plot.get("shading_u"), plot.get("shading_v")
    sd = plot.get("shading_dist_m")
    if (
        isinstance(su, np.ndarray)
        and isinstance(sv, np.ndarray)
        and isinstance(sd, np.ndarray)
        and sd.size == su.size
    ):
        mask = np.array(
            [
                _depth_visible(float(sd[i]), depth, float(su[i]), float(sv[i]), eps_m)
                for i in range(int(su.size))
            ],
            dtype=bool,
        )
        out["shading_u"] = su[mask]
        out["shading_v"] = sv[mask]

    tru, trv, trd = (
        plot.get("table_ring_u"),
        plot.get("table_ring_v"),
        plot.get("table_ring_dist_m"),
    )
    if (
        isinstance(tru, np.ndarray)
        and isinstance(trv, np.ndarray)
        and isinstance(trd, np.ndarray)
        and trd.size == tru.size == trv.size
    ):
        mask_tr = np.array(
            [
                _depth_visible(float(trd[i]), depth, float(tru[i]), float(trv[i]), eps_m)
                for i in range(int(tru.size))
            ],
            dtype=bool,
        )
        out["table_ring_u"] = tru[mask_tr]
        out["table_ring_v"] = trv[mask_tr]

    for rim_key, dist_key in (
        ("rim_bottom_uv_px", "rim_bottom_dist_m"),
        ("rim_top_uv_px", "rim_top_dist_m"),
        ("rim_mid_uv_px", "rim_mid_dist_m"),
    ):
        poly = plot.get(rim_key)
        dists = plot.get(dist_key)
        if (
            isinstance(poly, list)
            and isinstance(dists, list)
            and len(poly) == len(dists)
            and len(poly) >= 1
        ):
            us: List[float] = []
            vs: List[float] = []
            for i, p in enumerate(poly):
                if _depth_visible(float(dists[i]), depth, float(p[0]), float(p[1]), eps_m):
                    us.append(float(p[0]))
                    vs.append(float(p[1]))
                else:
                    us.append(float("nan"))
                    vs.append(float("nan"))
            out[rim_key + "_draw_u"] = us
            out[rim_key + "_draw_v"] = vs

    mers = plot.get("meridian_uv_px")
    mdists = plot.get("meridian_dist_m")
    md_out: List[Tuple[List[float], List[float]]] = []
    if isinstance(mers, list) and isinstance(mdists, list) and len(mers) == len(mdists):
        for seg, sd in zip(mers, mdists):
            if (
                not isinstance(seg, list)
                or not isinstance(sd, list)
                or len(seg) != len(sd)
                or len(seg) < 2
            ):
                continue
            us, vs = [], []
            for i, p in enumerate(seg):
                if _depth_visible(float(sd[i]), depth, float(p[0]), float(p[1]), eps_m):
                    us.append(float(p[0]))
                    vs.append(float(p[1]))
                else:
                    us.append(float("nan"))
                    vs.append(float("nan"))
            md_out.append((us, vs))
    out["_meridian_uv_draw"] = md_out
    return out


def _next_eval_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in base.iterdir():
        if not p.is_dir() or not p.name.startswith("eval_"):
            continue
        suffix = p.name.split("_", 1)[1]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    out = base / f"eval_{max_idx + 1:04d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _select_one_group_per_token(samples: List[ActionSample]) -> Dict[Tuple[str, str, int], List[ActionSample]]:
    tok_groups: Dict[str, Dict[Tuple[str, str, int], List[ActionSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in samples:
        tok = str(s.movement_token).strip().lower()
        gk = (str(s.scene_id), str(s.shard_path), int(s.goal_sample_group_index))
        tok_groups[tok][gk].append(s)

    out: Dict[Tuple[str, str, int], List[ActionSample]] = {}
    for tok in sorted(tok_groups.keys()):
        groups = tok_groups[tok]
        chosen_key: Tuple[str, str, int] | None = None
        chosen_rows: List[ActionSample] = []
        for gk in sorted(groups.keys()):
            rows = groups[gk]
            by_iv: Dict[int, ActionSample] = {}
            for r in rows:
                by_iv[int(r.instruction_variant_index)] = r
            if len(by_iv) >= 4:
                chosen_key = gk
                chosen_rows = [by_iv[k] for k in sorted(by_iv.keys())[:4]]
                break
        if chosen_key is None:
            # Fallback: choose densest group for token.
            best = sorted(
                groups.items(),
                key=lambda kv: (-len({int(x.instruction_variant_index) for x in kv[1]}), kv[0]),
            )[0]
            chosen_key = best[0]
            rows = best[1]
            by_iv = {}
            for r in rows:
                by_iv[int(r.instruction_variant_index)] = r
            chosen_rows = [by_iv[k] for k in sorted(by_iv.keys())[:4]]
        if len(chosen_rows) == 4:
            out[chosen_key] = chosen_rows
    return out


def _compute_prediction_for_sample(
    *,
    cfg: Any,
    region_cfg: Any,
    vlm: PaliGemmaContextEncoder,
    action_model: ActionExpertModel,
    device: torch.device,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
    sample: ActionSample,
    sample_index: int,
    dataset: ActionExpertDataset,
) -> Dict[str, Any]:
    raw = dataset[int(sample_index)]
    batch = _batch_to_device(action_expert_collate([raw]), device, cfg)
    ctx = vlm.forward_context(
        images_uint8=batch["image"],
        system_prompts=batch["system_prompt"],
        instructions=batch["instruction_text"],
        object_labels=batch["object_labels"],
    )
    label_emb = vlm.embed_labels(batch["keypoint_labels"])
    sample_positions = _rollout_prediction(
        action_model=action_model,
        context=ctx["context"],
        context_mask=ctx["attention_mask"],
        label_embeddings=label_emb,
        keypoint_positions=batch["keypoint_positions"],
        steps=int(cfg.integration_steps),
        n_samples=int(cfg.inference_samples),
    )  # [1, S, 3]
    valid_rows: List[torch.Tensor] = []
    for sidx in range(sample_positions.shape[1]):
        p = denormalize_xyz_torch(sample_positions[0, sidx], xyz_mean, xyz_std, float(norm_eps))
        ok = goal_region_contains(
            pred_xyz=p,
            goal_xyz=batch["dataset_goal_xyz_world"][0],
            movement_token=batch["movement_token"][0],
            constraint_type=batch["constraint_type"][0],
            constraint_params=batch["constraint_params"][0],
            cfg=region_cfg,
        )
        if ok:
            valid_rows.append(sample_positions[0, sidx])
    if valid_rows:
        pred_n = torch.stack(valid_rows, dim=0).mean(dim=0)
    else:
        pred_n = sample_positions[0].mean(dim=0)
    pred_w = denormalize_xyz_torch(pred_n, xyz_mean, xyz_std, float(norm_eps))
    dataset_goal = batch["dataset_goal_xyz_world"][0]
    training_goal = batch["goal_xyz_world"][0]
    success_gr = goal_region_contains(
        pred_xyz=pred_w,
        goal_xyz=dataset_goal,
        movement_token=batch["movement_token"][0],
        constraint_type=batch["constraint_type"][0],
        constraint_params=batch["constraint_params"][0],
        cfg=region_cfg,
    )
    l2 = float(torch.linalg.norm(pred_w - training_goal, dim=-1).item())
    l2_to_dataset_goal_m = float(torch.linalg.norm(pred_w - dataset_goal, dim=-1).item())
    return {
        "instruction": str(sample.instruction),
        "instruction_variant_index": int(sample.instruction_variant_index),
        "datapoint_index": int(sample.datapoint_index),
        "predicted_xyz_world": [float(x) for x in pred_w.detach().cpu().tolist()],
        "goal_xyz_world": [float(x) for x in training_goal.detach().cpu().tolist()],
        "dataset_goal_xyz_world": [float(x) for x in dataset_goal.detach().cpu().tolist()],
        "sampled_goal_in_region": bool(batch["sampled_goal_in_region"][0]),
        "success": bool(success_gr),
        "success_metric": "goal_region_contains_vs_dataset_goal_xyz_world",
        "l2_error_m": float(l2),
        "l2_error_to_dataset_goal_m": float(l2_to_dataset_goal_m),
    }


def _render_scene_all(scene: Dict[str, Any], out_path: Path) -> None:
    rgb = plt.imread(scene["image_path"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb)
    kps = scene["all_keypoints"]
    for kp in kps:
        uv = kp.get("uv_px")
        if not uv:
            continue
        ax.scatter(float(uv[0]), float(uv[1]), c="yellow", s=28, alpha=0.9, edgecolors="black", linewidths=0.5)
    tool_uv = scene.get("tool_uv_px")
    goal_uv = scene.get("goal_uv_px")
    if tool_uv is not None:
        ax.scatter(float(tool_uv[0]), float(tool_uv[1]), c="tab:blue", s=170, marker="*", label="tool")
    if goal_uv is not None:
        ax.scatter(float(goal_uv[0]), float(goal_uv[1]), c="tab:green", s=110, marker="x", label="goal")
    if scene.get("reference_uv_px") is not None:
        ref = scene["reference_uv_px"]
        ax.scatter(float(ref[0]), float(ref[1]), c="tab:orange", s=90, marker="^", label="ref")
    if scene.get("secondary_reference_uv_px") is not None:
        ref2 = scene["secondary_reference_uv_px"]
        ax.scatter(float(ref2[0]), float(ref2[1]), c="tab:red", s=90, marker="^", label="ref2")
    vs = sorted(
        scene.get("variants") or [],
        key=lambda x: int(x.get("instruction_variant_index", 0)),
    )
    base = (
        f"All keypoints | token={scene['movement_token']} | "
        f"scene={scene['scene_id']} | group={scene['goal_sample_group_index']}"
    )
    if vs:
        ins0 = str(vs[0].get("instruction", "")).strip()
        ext = ins0 if len(vs) == 1 else f"{ins0} (+{len(vs) - 1} other NL variants)"
        title = f"{base} | {ext}"
        fs = 8.5 if len(title) > 130 else 10.0
        ax.set_title(title, fontsize=fs)
    else:
        ax.set_title(base)
    ax.set_axis_off()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_uv(ax: Any, uv: Any, **kwargs: Any) -> None:
    if uv is None:
        return
    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
        ax.scatter(float(uv[0]), float(uv[1]), **kwargs)


def _resize_depth_to_image(depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Bilinear resize depth to (H,W) of RGB if shapes differ."""
    dh, dw = int(depth.shape[0]), int(depth.shape[1])
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if (dh, dw) == (h, w):
        return depth
    if PILImage is None:
        return depth
    z = PILImage.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    z = z.resize((w, h), PILImage.BILINEAR)
    return np.asarray(z, dtype=np.float64)


def _render_scene_variant(scene: Dict[str, Any], variant: Dict[str, Any], out_path: Path) -> None:
    rgb = plt.imread(scene["image_path"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb)
    plot = variant.get("_goal_region_plot")
    depth_map = scene.get("_depth_map_m")
    if isinstance(depth_map, np.ndarray) and depth_map.ndim == 2:
        depth_map = _resize_depth_to_image(depth_map, rgb)
    eps_m = float(scene.get("depth_occlusion_eps_m", 0.025))
    no_depth_occ = bool(scene.get("no_depth_occlusion", False))
    if isinstance(plot, dict):
        if (
            isinstance(depth_map, np.ndarray)
            and depth_map.ndim == 2
            and not no_depth_occ
        ):
            plot = _filter_goal_region_plot_for_depth(plot, depth_map, eps_m)

        su = plot.get("shading_u")
        sv = plot.get("shading_v")
        if isinstance(su, np.ndarray) and isinstance(sv, np.ndarray) and su.size > 8:
            lbl = "goal region (volume projection)"
            if isinstance(depth_map, np.ndarray) and not no_depth_occ:
                lbl += ", depth-occluded"
            ax.scatter(
                su,
                sv,
                s=3.0,
                c="#00FF88",
                alpha=0.14,
                linewidths=0,
                zorder=2,
                rasterized=True,
                label=lbl,
            )
        tru, trv = plot.get("table_ring_u"), plot.get("table_ring_v")
        if isinstance(tru, np.ndarray) and isinstance(trv, np.ndarray) and tru.size > 4:
            tr_lbl = "goal XY @ table (dot ring)"
            if isinstance(depth_map, np.ndarray) and not no_depth_occ:
                tr_lbl += ", depth-occluded"
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
                label=tr_lbl,
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
                    ax.plot(
                        du,
                        dv,
                        color="white",
                        linewidth=0.85,
                        alpha=0.38,
                        zorder=3,
                    )
        elif isinstance(plot.get("meridian_uv_px"), list):
            for seg in plot["meridian_uv_px"]:
                if isinstance(seg, list) and len(seg) >= 2:
                    us = [float(p[0]) for p in seg]
                    vs = [float(p[1]) for p in seg]
                    ax.plot(
                        us,
                        vs,
                        color="white",
                        linewidth=0.85,
                        alpha=0.38,
                        zorder=3,
                    )
    _scatter_uv(
        ax,
        scene.get("tool_uv_px"),
        c="tab:blue",
        s=170,
        marker="*",
        label="tool",
        zorder=5,
    )
    _scatter_uv(
        ax,
        variant.get("dataset_goal_uv_px"),
        c="tab:green",
        s=120,
        marker="x",
        label="dataset goal (region center)",
        zorder=6,
    )
    _scatter_uv(
        ax,
        variant.get("predicted_uv_px"),
        c="tab:purple",
        s=120,
        marker="o",
        label="pred",
        zorder=7,
    )
    _scatter_uv(
        ax,
        scene.get("reference_uv_px"),
        c="tab:orange",
        s=95,
        marker="^",
        label="ref",
        zorder=4,
    )
    _scatter_uv(
        ax,
        scene.get("secondary_reference_uv_px"),
        c="tab:red",
        s=95,
        marker="^",
        label="ref2",
        zorder=4,
    )
    succ_txt = "SUCCESS" if bool(variant["success"]) else "FAIL"
    color = "tab:green" if bool(variant["success"]) else "tab:red"
    ax.text(
        0.02,
        0.98,
        f"{succ_txt} | l2={variant['l2_error_m']:.4f}m\niv={variant['instruction_variant_index']}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=color,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": color},
    )
    tok = str(scene["movement_token"])
    instr = str(variant.get("instruction", "")).strip()
    head = f"Focused view | token={tok}"
    if instr:
        full_one = f"{head} | {instr}"
        if len(full_one) > 118:
            ax.set_title(f"{head}\n{instr}", fontsize=9)
        else:
            ax.set_title(full_one, fontsize=10)
    else:
        ax.set_title(head)
    ax.set_axis_off()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    apply_hf_env()
    parser = argparse.ArgumentParser(description="Evaluate one 4-NL scene per movement token.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training/eval"))
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="If >0, only evaluate this many scenes (one scene group per movement token each).",
    )
    parser.add_argument(
        "--goal-region-points",
        type=int,
        default=4500,
        help="Random interior samples for shaded cylinder projection (full XY and Z extent).",
    )
    parser.add_argument(
        "--no-depth-occlusion",
        action="store_true",
        help="Draw full goal region without masking against shard depth.npy.",
    )
    parser.add_argument(
        "--depth-occlusion-eps",
        type=float,
        default=0.025,
        help="Meters of slack when comparing point distance-to-camera vs depth map.",
    )
    parser.add_argument(
        "--table-ring-samples",
        type=int,
        default=260,
        help="Dots on the bright tabletop ring (same XY radius as goal region, inferred table Z).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_hf_cache(str(cfg.hf_cache_dir))
    region_cfg = _region_cfg(cfg)
    device = torch.device(cfg.device)

    stats_path = Path(cfg.normalization_stats_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    xyz_mean_np, xyz_std_np, norm_eps_f = load_xyz_normalization_stats(stats_path)
    xyz_mean_t = torch.as_tensor(xyz_mean_np, dtype=torch.float32, device=device)
    xyz_std_t = torch.as_tensor(xyz_std_np, dtype=torch.float32, device=device)

    splits = split_shards_85_10_5(
        Path(cfg.dataset_dir),
        seed=int(cfg.seed),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
    )
    val_samples = load_action_samples_from_shards(
        splits["val"],
        max_keypoints=int(cfg.max_keypoints),
        region_cfg=region_cfg,
        explode_instruction_variants=bool(cfg.explode_instruction_variants),
    )
    selected = _select_one_group_per_token(val_samples)
    if int(args.max_scenes) > 0:
        selected = dict(sorted(selected.items())[: int(args.max_scenes)])
    selected_samples = [s for rows in selected.values() for s in rows]
    if not selected_samples:
        raise RuntimeError("No selected samples for token subset eval.")

    dataset = ActionExpertDataset(
        selected_samples,
        image_size=(cfg.image_size, cfg.image_size),
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
        region_cfg=region_cfg,
        sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
        goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
    )
    sample_id_to_idx = {id(s): i for i, s in enumerate(selected_samples)}

    vlm = PaliGemmaContextEncoder(
        model_id=str(cfg.paligemma_model_id),
        device=device,
        cache_dir=str(cfg.hf_cache_dir),
        local_files_only=bool(cfg.local_files_only),
        lora_rank=int(cfg.lora_rank),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        enable_gradient_checkpointing=bool(cfg.enable_gradient_checkpointing),
    ).to(device)
    action_model = ActionExpertModel(
        d_model=int(vlm.d_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_action_expert_layers),
        dropout=float(cfg.action_dropout),
        ffn_multiplier=int(cfg.ffn_multiplier),
        pos_norm_denom=float(cfg.pos_norm_denom),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    action_model.load_state_dict(ckpt["model_trainable_state"]["action_expert"])
    vlm.load_state_dict(ckpt["model_trainable_state"]["vlm_lora"], strict=False)
    vlm.eval()
    action_model.eval()

    out_dir = _next_eval_dir(args.output_root)
    renders_all_dir = out_dir / "renders_all"
    renders_variant_dir = out_dir / "renders_variants"
    scenes_dir = out_dir / "scenes"
    renders_all_dir.mkdir(parents=True, exist_ok=True)
    renders_variant_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    scene_rows: List[Dict[str, Any]] = []
    all_l2: List[float] = []
    all_l2_to_dataset: List[float] = []
    all_success: List[bool] = []
    all_tokens: List[str] = []
    token_to_l2: Dict[str, List[float]] = defaultdict(list)
    token_to_l2_ds: Dict[str, List[float]] = defaultdict(list)
    token_to_success: Dict[str, List[bool]] = defaultdict(list)
    token_to_nl_spread: Dict[str, List[float]] = defaultdict(list)
    nl_spreads_all: List[float] = []

    for scene_key, rows in sorted(selected.items()):
        rows_sorted = sorted(rows, key=lambda s: int(s.instruction_variant_index))
        first = rows_sorted[0]
        shard_obj = _load_shard_json(str(first.shard_path))
        kp_map = dict(shard_obj.get("keypoints", {}) or {})
        cam = dict(shard_obj.get("camera", {}) or {})
        intr = dict(cam.get("intrinsics", {}) or {})
        w_from_c = np.asarray(cam.get("world_from_camera", np.eye(4)), dtype=np.float64)

        def _kp_uv(kp_id: str) -> List[float] | None:
            row = kp_map.get(str(kp_id), {})
            uv = row.get("uv_px")
            if isinstance(uv, (list, tuple)) and len(uv) == 2:
                return [float(uv[0]), float(uv[1])]
            return None

        scene_result: Dict[str, Any] = {
            "scene_id": str(first.scene_id),
            "shard_path": str(first.shard_path),
            "image_path": str(shard_obj.get("image", str(first.image_path))),
            "goal_sample_group_index": int(first.goal_sample_group_index),
            "movement_token": str(first.movement_token).strip().lower(),
            "constraint_type": str(first.constraint_type),
            "constraint_params": dict(first.constraint_params or {}),
            "reference_xyz_world": (
                [float(x) for x in first.reference_xyz_world.tolist()]
                if first.reference_xyz_world is not None
                else None
            ),
            "secondary_reference_xyz_world": (
                [float(x) for x in first.secondary_reference_xyz_world.tolist()]
                if first.secondary_reference_xyz_world is not None
                else None
            ),
            "all_keypoints": [
                {
                    **kp,
                    "uv_px": _kp_uv(str(kp.get("id", ""))),
                }
                for kp in list(first.keypoints)
            ],
            "tool_keypoint_id": str(first.tool_keypoint_id),
            "tool_xyz_world": [float(x) for x in first.keypoints[0]["position_xyz_world"]],
            "tool_uv_px": _kp_uv(str(first.tool_keypoint_id)),
            # Shard goal (center of goal region); same notion as batch `dataset_goal_xyz_world`.
            "dataset_goal_xyz_world": [float(x) for x in first.goal_xyz_world.tolist()],
            "goal_xyz_world": [float(x) for x in first.goal_xyz_world.tolist()            ],
            "variants": [],
        }
        kp_list_for_uv = scene_result["all_keypoints"]
        scene_result["inferred_table_top_z_m"] = float(_estimate_table_top_z_m(kp_list_for_uv))
        if scene_result["reference_xyz_world"] is not None:
            scene_result["reference_uv_px"] = _uv_or_fallback(
                scene_result["reference_xyz_world"], intr, w_from_c, kp_list_for_uv
            )
        else:
            scene_result["reference_uv_px"] = None
        if scene_result["secondary_reference_xyz_world"] is not None:
            scene_result["secondary_reference_uv_px"] = _uv_or_fallback(
                scene_result["secondary_reference_xyz_world"], intr, w_from_c, kp_list_for_uv
            )
        else:
            scene_result["secondary_reference_uv_px"] = None
        scene_result["goal_uv_px"] = _uv_or_fallback(
            scene_result["goal_xyz_world"], intr, w_from_c, kp_list_for_uv
        )
        dpth_path = _resolve_depth_npy_path(str(first.shard_path), shard_obj)
        depth_arr = _load_depth_map_npy(dpth_path) if dpth_path is not None else None
        scene_result["depth_occlusion_eps_m"] = float(args.depth_occlusion_eps)
        scene_result["no_depth_occlusion"] = bool(args.no_depth_occlusion)
        scene_result["goal_region_depth_map_loaded"] = depth_arr is not None
        scene_result["goal_region_depth_occlusion_applied"] = bool(
            depth_arr is not None and not bool(args.no_depth_occlusion)
        )
        if depth_arr is not None:
            scene_result["_depth_map_m"] = depth_arr
        scene_stub = (
            f"{scene_result['movement_token']}__{scene_result['scene_id']}__g{scene_result['goal_sample_group_index']}"
        ).replace("/", "_")
        pred_xyz_list: List[np.ndarray] = []
        for s in rows_sorted:
            pred = _compute_prediction_for_sample(
                cfg=cfg,
                region_cfg=region_cfg,
                vlm=vlm,
                action_model=action_model,
                device=device,
                xyz_mean=xyz_mean_t,
                xyz_std=xyz_std_t,
                norm_eps=float(norm_eps_f),
                sample=s,
                sample_index=int(sample_id_to_idx[id(s)]),
                dataset=dataset,
            )
            pred["predicted_uv_px"] = _uv_or_fallback(
                pred["predicted_xyz_world"], intr, w_from_c, kp_list_for_uv
            )
            pred["dataset_goal_uv_px"] = _uv_or_fallback(
                pred["dataset_goal_xyz_world"], intr, w_from_c, kp_list_for_uv
            )
            pred["training_goal_uv_px"] = _uv_or_fallback(
                pred["goal_xyz_world"], intr, w_from_c, kp_list_for_uv
            )
            seed_payload = f"{scene_stub}|iv{pred['instruction_variant_index']}".encode("utf-8")
            seed_i = int(hashlib.sha256(seed_payload).hexdigest()[:8], 16)
            plot = _build_goal_region_plot(
                pred["dataset_goal_xyz_world"],
                movement_token=str(scene_result["movement_token"]),
                constraint_type=str(scene_result["constraint_type"]),
                constraint_params=dict(scene_result["constraint_params"] or {}),
                region_cfg=region_cfg,
                intr=intr,
                w_from_c=w_from_c,
                keypoints=kp_list_for_uv,
                n_volume=int(args.goal_region_points),
                n_rim=96,
                n_meridian_z=28,
                n_table_ring=int(args.table_ring_samples),
                rng=np.random.default_rng(seed_i),
            )
            pred["goal_region_meta"] = _goal_region_meta_only(plot)
            pred["_goal_region_plot"] = plot
            scene_result["variants"].append(pred)
            pred_xyz_list.append(np.asarray(pred["predicted_xyz_world"], dtype=np.float64))
            l2 = float(pred["l2_error_m"])
            l2_ds = float(pred.get("l2_error_to_dataset_goal_m", l2))
            succ = bool(pred["success"])
            tok = str(scene_result["movement_token"])
            all_l2.append(l2)
            all_l2_to_dataset.append(l2_ds)
            all_success.append(succ)
            all_tokens.append(tok)
            token_to_l2[tok].append(l2)
            token_to_l2_ds[tok].append(l2_ds)
            token_to_success[tok].append(succ)

        if len(pred_xyz_list) == 4:
            arr = np.asarray(pred_xyz_list, dtype=np.float64).reshape(4, 3)
            centroid = arr.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(arr - centroid[None, :], axis=1)))
            scene_result["nl_variance_spread_m"] = float(spread)
            nl_spreads_all.append(float(spread))
            token_to_nl_spread[str(scene_result["movement_token"])].append(float(spread))
        else:
            scene_result["nl_variance_spread_m"] = None

        _render_scene_all(scene_result, renders_all_dir / f"{scene_stub}__all.png")
        for v in scene_result["variants"]:
            iv = int(v["instruction_variant_index"])
            _render_scene_variant(
                scene_result,
                v,
                renders_variant_dir / f"{scene_stub}__iv{iv}.png",
            )

        for v in scene_result["variants"]:
            v.pop("_goal_region_plot", None)
        scene_result.pop("_depth_map_m", None)
        (scenes_dir / f"{scene_stub}.json").write_text(
            json.dumps(scene_result, indent=2), encoding="utf-8"
        )
        scene_rows.append(scene_result)

    summary = {
        "checkpoint": str(args.checkpoint),
        "success_metric": "goal_region_contains_vs_dataset_goal_xyz_world",
        "max_scenes": int(args.max_scenes) if int(args.max_scenes) > 0 else None,
        "goal_region_volume_samples": int(args.goal_region_points),
        "n_scenes": int(len(scene_rows)),
        "n_samples": int(len(all_l2)),
        "movement_tokens": sorted(set(all_tokens)),
        "overall": {
            "success_rate": float(np.mean(np.asarray(all_success, dtype=np.float64))) if all_success else 0.0,
            "mean_l2_error_m": float(np.mean(np.asarray(all_l2, dtype=np.float64))) if all_l2 else 0.0,
            "l2_std_error_m": float(np.std(np.asarray(all_l2, dtype=np.float64))) if all_l2 else 0.0,
            "mean_l2_error_to_dataset_goal_m": float(np.mean(np.asarray(all_l2_to_dataset, dtype=np.float64)))
            if all_l2_to_dataset
            else 0.0,
            "l2_std_error_to_dataset_goal_m": float(np.std(np.asarray(all_l2_to_dataset, dtype=np.float64)))
            if all_l2_to_dataset
            else 0.0,
            "nl_variance_mean_m": float(np.mean(np.asarray(nl_spreads_all, dtype=np.float64))) if nl_spreads_all else 0.0,
            "nl_variance_std_m": float(np.std(np.asarray(nl_spreads_all, dtype=np.float64))) if nl_spreads_all else 0.0,
        },
        "per_token": {},
    }
    for tok in sorted(set(all_tokens)):
        l2_arr = np.asarray(token_to_l2[tok], dtype=np.float64)
        l2ds_arr = np.asarray(token_to_l2_ds[tok], dtype=np.float64)
        s_arr = np.asarray(token_to_success[tok], dtype=np.float64)
        nl_arr = np.asarray(token_to_nl_spread.get(tok, []), dtype=np.float64)
        summary["per_token"][tok] = {
            "n_samples": int(l2_arr.size),
            "success_rate": float(np.mean(s_arr)) if s_arr.size else 0.0,
            "mean_l2_error_m": float(np.mean(l2_arr)) if l2_arr.size else 0.0,
            "l2_std_error_m": float(np.std(l2_arr)) if l2_arr.size else 0.0,
            "mean_l2_error_to_dataset_goal_m": float(np.mean(l2ds_arr)) if l2ds_arr.size else 0.0,
            "l2_std_error_to_dataset_goal_m": float(np.std(l2ds_arr)) if l2ds_arr.size else 0.0,
            "nl_variance_mean_m": float(np.mean(nl_arr)) if nl_arr.size else 0.0,
            "nl_variance_std_m": float(np.std(nl_arr)) if nl_arr.size else 0.0,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **summary["overall"]}, indent=2))


if __name__ == "__main__":
    main()
