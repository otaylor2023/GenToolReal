"""Procedurally generate reactive closed-loop stroke_sweep shards (dataset_0011).

Each scene is rolled out densely (brush + object motion), then windowed into many
(current state -> next 6 dense waypoints) datapoints. Object tracks the brush
contact during the sweep phase; disturbances relabel targets via
``plan_next_dense_steps``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from generative_str_pipeline.build_dataset_0009_brush_sweep_diverse import (
    DESTINATIONS_SWEEP,
    MATERIALS_SWEEP,
    PATH_STRAIGHT,
    TEMPLATES_SWEEP,
    TOOLS_SWEEP,
    _sample_path_shape,
)
from generative_str_pipeline.build_dataset_0008_brush_procedural import (
    TABLE_LABEL,
    TOOL_LABEL,
    _datapoint_rng,
    _orthonormal_basis_in_plane,
    _project_surface_dir,
    _safe_format,
    _sample_near_up_normal,
    _waypoints_from_contacts,
)
from generative_str_pipeline.build_dataset_0010_brush_sweep_sim import (
    SweepGenConfig,
    _build_sweep_contacts,
    _sample_material_destination,
    _sample_tool_pose_sweep,
    gen_stroke_sweep_from_inputs,
)
from generative_str_pipeline.sim_workspace import (
    BRUSH_BODY_FRONT_M,
    GOAL_REGION_RADIUS_M,
    WORKSPACE_X_EXTENT_M,
    WORKSPACE_Y_EXTENT_M,
    clip_xy_rect,
    sample_xy_rect,
)

DATASET_ID = "dataset_0011_brush_sweep_reactive"
OBJECT_RADIUS_M = 0.02
NUM_OUTPUT_WAYPOINTS = 6


@dataclass
class ReactiveGenConfig(SweepGenConfig):
    """Reactive closed-loop dataset generator."""

    dense_step_spacing_m: float = 0.05
    window_stride: int = 2
    max_windows_per_scene: int = 24
    # Closed-loop receding-horizon execution.
    num_output_waypoints: int = 15
    max_generations: int = 30
    executed_chunk: int = 5
    goal_region_radius_m: float = GOAL_REGION_RADIUS_M
    contact_xy_tol_m: float = 0.03
    contact_z_tol_m: float = 0.035
    # Disturbance probabilities (applied per generation).
    prob_jitter: float = 0.10
    prob_lateral_knock: float = 0.12
    prob_teleport: float = 0.02
    jitter_xy_m: float = 0.025
    knock_lateral_m: float = 0.04
    scenes_per_shard: int = 150
    # Start the sweep further back: the brush descends this far behind the object
    # (along the sweep direction) before pushing through it. Larger than the
    # shared default (0.03-0.10 m) so the approach/contact begins clearly further
    # behind the ball.
    behind_offset_m_range: tuple[float, float] = (0.12, 0.18)


def _sweep_unit(
    material_xyz: np.ndarray, destination_xyz: np.ndarray, surface_normal: np.ndarray
) -> np.ndarray:
    sweep_vec = destination_xyz - material_xyz
    sweep_vec = sweep_vec - np.dot(sweep_vec, surface_normal) * surface_normal
    n = float(np.linalg.norm(sweep_vec))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (sweep_vec / n).astype(np.float32)


def _densify_contacts(
    contacts: list[np.ndarray],
    *,
    step_spacing_m: float,
) -> list[np.ndarray]:
    """Linearly densify a polyline of 3D contact points."""
    if len(contacts) < 2:
        return [np.asarray(c, dtype=np.float32).reshape(3) for c in contacts]
    dense: list[np.ndarray] = []
    for i in range(len(contacts) - 1):
        a = np.asarray(contacts[i], dtype=np.float64).reshape(3)
        b = np.asarray(contacts[i + 1], dtype=np.float64).reshape(3)
        seg_len = float(np.linalg.norm(b - a))
        n_steps = max(1, int(np.round(seg_len / max(step_spacing_m, 1e-4))))
        for j in range(n_steps):
            t = float(j) / float(n_steps)
            dense.append(((1.0 - t) * a + t * b).astype(np.float32))
    dense.append(np.asarray(contacts[-1], dtype=np.float32).reshape(3))
    return dense


def _contacts_to_waypoints(
    contacts: list[np.ndarray],
    surface_normal: np.ndarray,
    surface_dir: np.ndarray,
) -> np.ndarray:
    n = len(contacts)
    wp_contacts = np.stack(contacts, axis=0).astype(np.float32)
    wp = np.zeros((n, 9), dtype=np.float32)
    sn = np.asarray(surface_normal, dtype=np.float32).reshape(3)
    sn = sn / max(float(np.linalg.norm(sn)), 1e-9)
    wp_normal = (-sn).astype(np.float32)

    # Forward direction (overall generation sweep dir), projected onto surface.
    default_fwd = np.asarray(surface_dir, dtype=np.float32).reshape(3)
    default_fwd = default_fwd - float(np.dot(default_fwd, sn)) * sn
    dn = float(np.linalg.norm(default_fwd))
    default_fwd = (
        default_fwd / dn if dn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    # The tool surface_dir tracks the local path tangent (motion axis), but since
    # the brush edge is symmetric the axis can point either way. We pick the sign
    # (+tangent or -tangent) closest to the previous waypoint's heading so the
    # brush never flips ~180 deg / rotates sharply mid-trajectory near the object
    # (which used to knock it off). Degenerate horizontal tangents (pure vertical
    # descent/retraction) carry forward the last valid heading.
    prev_dir = default_fwd.copy()
    for i in range(n):
        if n > 1 and i < n - 1:
            tang = wp_contacts[i + 1] - wp_contacts[i]
        elif n > 1:
            tang = wp_contacts[i] - wp_contacts[i - 1]
        else:
            tang = default_fwd
        tang = tang - float(np.dot(tang, sn)) * sn
        tn = float(np.linalg.norm(tang))
        if tn > 1e-5:
            axis = (tang / tn).astype(np.float32)
            # Choose the sign that requires the least rotation from prev_dir.
            if float(np.dot(axis, prev_dir)) < 0.0:
                axis = -axis
            prev_dir = axis
        else:
            axis = prev_dir
        wp[i, 0:3] = wp_contacts[i]
        wp[i, 3:6] = wp_normal
        wp[i, 6:9] = axis.astype(np.float32)
    return wp


def _object_from_brush(
    brush_contact: np.ndarray,
    sweep_unit: np.ndarray,
    table_z: float,
) -> np.ndarray:
    """Analytic push: object center sits just ahead of the brush contact."""
    offset = float(BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M)
    xy = brush_contact[:2] + sweep_unit[:2] * offset
    return np.array([xy[0], xy[1], table_z], dtype=np.float32)


def build_dense_rollout(
    rng: np.random.Generator,
    cfg: ReactiveGenConfig,
    *,
    tool_home: np.ndarray,
    tool_normal: np.ndarray,
    tool_surface_dir: np.ndarray,
    material_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    coarse_contacts: np.ndarray,
    surface_normal: np.ndarray,
    surface_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense analytic rollout: brush_steps [K,9], object_xyz [K,3], in_contact [K]."""
    table_z = float(cfg.table_xyz_world[2])
    sweep_unit = _sweep_unit(material_xyz, destination_xyz, surface_normal)

    keyframes = [tool_home.reshape(3)] + [
        coarse_contacts[i].reshape(3) for i in range(coarse_contacts.shape[0])
    ]
    dense_contacts = _densify_contacts(
        keyframes, step_spacing_m=float(cfg.dense_step_spacing_m)
    )
    brush_steps = _contacts_to_waypoints(dense_contacts, surface_normal, surface_dir)

    # Object tracks once the brush reaches the material-contact keyframe (wp3).
    material_kf = coarse_contacts[3].reshape(3)
    track_from_idx = len(dense_contacts)
    for i, c in enumerate(dense_contacts):
        if float(np.linalg.norm(np.asarray(c) - material_kf)) < 1e-4:
            track_from_idx = i
            break
        if float(np.linalg.norm(np.asarray(c)[:2] - material_kf[:2])) < 0.015:
            track_from_idx = i
            break
    if track_from_idx >= len(dense_contacts):
        track_from_idx = max(0, int(0.55 * len(dense_contacts)))

    k = len(dense_contacts)
    object_xyz = np.zeros((k, 3), dtype=np.float32)
    in_contact = np.zeros(k, dtype=bool)
    init_obj = np.asarray(material_xyz, dtype=np.float32).reshape(3).copy()
    init_obj[2] = table_z + float(init_obj[2] - table_z) * 0.0  # keep sampled z
    for i in range(k):
        if i < track_from_idx:
            object_xyz[i] = init_obj
            in_contact[i] = False
        else:
            in_contact[i] = True
            object_xyz[i] = _object_from_brush(dense_contacts[i], sweep_unit, table_z)
            object_xyz[i, 2] = init_obj[2]

    return brush_steps, object_xyz, in_contact


def _interpolate_contacts(
    start: np.ndarray,
    end: np.ndarray,
    *,
    n: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    a = np.asarray(start, dtype=np.float64).reshape(3)
    b = np.asarray(end, dtype=np.float64).reshape(3)
    for i in range(1, n + 1):
        t = float(i) / float(n)
        out.append(((1.0 - t) * a + t * b).astype(np.float32))
    return out


def plan_next_dense_steps(
    rng: np.random.Generator,
    cfg: ReactiveGenConfig,
    *,
    brush_contact: np.ndarray,
    brush_normal: np.ndarray,
    brush_surface_dir: np.ndarray,
    object_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    in_contact: bool,
    surface_normal: np.ndarray | None = None,
    num_steps: int = NUM_OUTPUT_WAYPOINTS,
) -> np.ndarray:
    """Plan the next ``num_steps`` dense brush waypoints from the current state."""
    table_z = float(cfg.table_xyz_world[2])
    surface_normal = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if surface_normal is None
        else np.asarray(surface_normal, dtype=np.float32).reshape(3)
    )
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3)
    dest = np.asarray(destination_xyz, dtype=np.float32).reshape(3)
    brush = np.asarray(brush_contact, dtype=np.float32).reshape(3)
    surface_dir = _sweep_unit(obj, dest, surface_normal)

    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    air_h = float(rng.uniform(*cfg.approach_air_height_m_range))
    contact_offset = surface_normal * contact_h
    air_offset = surface_normal * air_h

    behind_offset_m = float(rng.uniform(*cfg.behind_offset_m_range))
    behind_xyz = (obj - surface_dir * behind_offset_m).astype(np.float32)

    # The object rides ``push_offset`` ahead of the brush contact, so to land the
    # object on the destination the brush must stop short by that amount.
    push_offset = float(BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M)
    brush_goal = (dest - surface_dir * push_offset + contact_offset).astype(np.float32)

    if not in_contact:
        # Approach: descend behind the object and push through it. Only a brief
        # air hop when still far -- the descent must land within the executed
        # chunk, so we never "go up and over" (which would make the brush hover
        # since only the first couple of steps are executed each generation).
        dist_to_obj = float(np.linalg.norm(brush[:2] - obj[:2]))
        behind_contact = (behind_xyz + contact_offset).astype(np.float32)
        material_contact = (obj + contact_offset).astype(np.float32)
        if dist_to_obj > 0.18:
            behind_air = (behind_xyz + air_offset).astype(np.float32)
            keyframes = [
                brush,
                behind_air,
                behind_contact,
                material_contact,
                brush_goal,
            ]
        else:
            keyframes = [
                brush,
                behind_contact,
                material_contact,
                brush_goal,
            ]
    else:
        # Sweep: current -> along sweep, stopping short so the object lands on dest
        mid = (0.5 * (obj + contact_offset) + 0.5 * brush_goal).astype(np.float32)
        keyframes = [brush, (obj + contact_offset).astype(np.float32), mid, brush_goal]

    dense: list[np.ndarray] = []
    for i in range(len(keyframes) - 1):
        seg = _interpolate_contacts(
            keyframes[i],
            keyframes[i + 1],
            n=max(1, int(np.round(
                float(np.linalg.norm(keyframes[i + 1] - keyframes[i]))
                / max(cfg.dense_step_spacing_m, 1e-4)
            ))),
        )
        dense.extend(seg)

    if not dense:
        dense = [brush.copy()]
    # If the planned path (delivery to the destination) fits within the horizon,
    # the remaining steps lift the brush straight up (retract after delivery)
    # rather than dwelling on the destination.
    if len(dense) < num_steps:
        base = np.asarray(dense[-1], dtype=np.float32).copy()
        max_lift_z = float(base[2] + 0.15)
        j = 1
        while len(dense) < num_steps:
            lifted = base.copy()
            lifted[2] = float(
                min(base[2] + j * float(cfg.dense_step_spacing_m), max_lift_z)
            )
            dense.append(lifted)
            j += 1
    wp = _contacts_to_waypoints(dense[:num_steps], surface_normal, surface_dir)
    return wp.reshape(num_steps, 9)


def _pad_target_waypoints(target: np.ndarray, num_steps: int = NUM_OUTPUT_WAYPOINTS) -> np.ndarray:
    """Ensure exactly ``num_steps`` rows, repeating the last pose if needed."""
    t = np.asarray(target, dtype=np.float32).reshape(-1, 9)
    if t.shape[0] >= num_steps:
        return t[:num_steps].copy()
    out = np.zeros((num_steps, 9), dtype=np.float32)
    out[: t.shape[0]] = t
    for i in range(t.shape[0], num_steps):
        out[i] = t[-1]
    return out


def _apply_disturbance(
    rng: np.random.Generator,
    cfg: ReactiveGenConfig,
    *,
    object_xyz: np.ndarray,
    in_contact: bool,
    sweep_unit: np.ndarray,
    avoid_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return perturbed object xyz (may equal input if no disturbance)."""
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3).copy()
    u = float(rng.random())
    if u < cfg.prob_jitter and not in_contact:
        obj[0] += float(rng.uniform(-cfg.jitter_xy_m, cfg.jitter_xy_m))
        obj[1] += float(rng.uniform(-cfg.jitter_xy_m, cfg.jitter_xy_m))
        obj[:2] = clip_xy_rect(
            obj[:2],
            x_extent=cfg.workspace_x_extent_m,
            y_extent=cfg.workspace_y_extent_m,
        )
    elif u < cfg.prob_jitter + cfg.prob_lateral_knock and in_contact:
        perp = np.array([-sweep_unit[1], sweep_unit[0], 0.0], dtype=np.float32)
        obj += perp * float(rng.uniform(-cfg.knock_lateral_m, cfg.knock_lateral_m))
        obj[:2] = clip_xy_rect(
            obj[:2],
            x_extent=cfg.workspace_x_extent_m,
            y_extent=cfg.workspace_y_extent_m,
        )
    elif u < cfg.prob_jitter + cfg.prob_lateral_knock + cfg.prob_teleport:
        for _ in range(48):
            xy = sample_xy_rect(
                rng,
                x_extent=cfg.workspace_x_extent_m,
                y_extent=cfg.workspace_y_extent_m,
            )
            if avoid_xy is None or float(np.linalg.norm(xy - avoid_xy)) >= cfg.min_tool_material_sep_m:
                obj[0], obj[1] = float(xy[0]), float(xy[1])
                break
    return obj


def _execute_chunk(
    cfg: ReactiveGenConfig,
    plan: np.ndarray,
    *,
    brush_xyz: np.ndarray,
    object_xyz: np.ndarray,
    in_contact: bool,
    destination_xyz: np.ndarray,
    surface_normal: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Execute the first ``chunk`` planned steps; move the object only from those.

    Returns (new_brush_step[9], new_object_xyz[3], new_in_contact,
    object_trace[n_exec, 3]). Object is pushed analytically by the brush only
    while in contact during the executed steps -- not from the full plan.
    """
    table_z = float(cfg.table_xyz_world[2])
    dest = np.asarray(destination_xyz, dtype=np.float32).reshape(3)
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3).copy()
    contact = bool(in_contact)
    n_exec = max(1, min(int(chunk), int(plan.shape[0])))
    contact_xy_radius = float(BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M + cfg.contact_xy_tol_m)
    # Fixed sweep direction for this chunk (per-step recompute goes noisy as the
    # object nears the destination, causing it to orbit instead of settling).
    sweep_unit = _sweep_unit(obj, dest, surface_normal)
    obj_trace: list[np.ndarray] = []

    for i in range(n_exec):
        bc = plan[i, 0:3].astype(np.float32)
        if not contact:
            near_xy = float(np.linalg.norm(bc[:2] - obj[:2])) <= contact_xy_radius
            # Contact is relative to the OBJECT height (the object may be sampled
            # floating above the table), not the table surface -- otherwise the
            # brush sweeping at the object's height never registers contact.
            low = float(bc[2]) <= float(obj[2]) + float(cfg.contact_z_tol_m)
            if near_xy and low:
                contact = True
        if contact:
            pushed = _object_from_brush(bc, sweep_unit, table_z)
            pushed[2] = obj[2]
            # Clamp so the object settles on the destination instead of orbiting
            # past it once the brush has pushed it that far.
            if float(np.dot(dest[:2] - pushed[:2], sweep_unit[:2])) < 0.0:
                pushed[0], pushed[1] = float(dest[0]), float(dest[1])
            obj = pushed.astype(np.float32)
        obj_trace.append(obj.copy())

    new_brush = plan[n_exec - 1].astype(np.float32)
    return new_brush, obj, contact, np.stack(obj_trace, axis=0).astype(np.float32)


def scene_to_datapoints(
    rng: np.random.Generator,
    cfg: ReactiveGenConfig,
    *,
    shard_id: str,
    scene_index: int,
    base_datapoint_index: int,
) -> list[dict[str, Any]]:
    """One sampled scene -> a closed-loop reactive rollout of generations.

    Each generation: plan ``num_output_waypoints`` dense steps from the current
    (brush, object) state, record it, execute the first ``executed_chunk`` steps,
    and update the object only from those executed steps. The rollout stops as
    soon as the executed trajectory delivers the object into the goal region (the
    plan tail lifts the brush straight up after delivery).
    """
    table_xyz = cfg.sample_table_xyz(rng)
    dp_cfg = replace(cfg, table_xyz_world=table_xyz)

    tool_home, tool_normal, tool_surface_dir = _sample_tool_pose_sweep(rng, dp_cfg)
    material_xyz, destination_xyz = _sample_material_destination(
        rng,
        dp_cfg,
        avoid_xy=tool_home[:2],
        tool_heading_xy=tool_surface_dir[:2],
    )

    body = gen_stroke_sweep_from_inputs(
        rng,
        dp_cfg,
        (tool_home, tool_normal, tool_surface_dir),
        material_xyz=material_xyz,
        destination_xyz=destination_xyz,
    )
    surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    path_shape = body.get("path_shape", PATH_STRAIGHT)

    brush = np.asarray(tool_home, dtype=np.float32).reshape(3).copy()
    brush_normal = np.asarray(tool_normal, dtype=np.float32).reshape(3).copy()
    brush_surface_dir = np.asarray(tool_surface_dir, dtype=np.float32).reshape(3).copy()
    obj = np.asarray(material_xyz, dtype=np.float32).reshape(3).copy()
    in_contact = False

    chunk = max(1, int(cfg.executed_chunk))
    n_wp = int(cfg.num_output_waypoints)
    datapoints: list[dict[str, Any]] = []

    for gen in range(int(cfg.max_generations)):
        target = plan_next_dense_steps(
            rng,
            dp_cfg,
            brush_contact=brush,
            brush_normal=brush_normal,
            brush_surface_dir=brush_surface_dir,
            object_xyz=obj,
            destination_xyz=destination_xyz,
            in_contact=in_contact,
            surface_normal=surface_normal,
            num_steps=n_wp,
        )
        target = _pad_target_waypoints(target, num_steps=n_wp)

        # Execute the first `chunk` steps; the object moves only from those steps.
        new_brush, new_obj, new_contact, material_trace = _execute_chunk(
            dp_cfg,
            target,
            brush_xyz=brush,
            object_xyz=obj,
            in_contact=in_contact,
            destination_xyz=destination_xyz,
            surface_normal=surface_normal,
            chunk=chunk,
        )

        # Stop once the executed trajectory delivers the object to the goal. The
        # plan's tail already lifts the brush straight up after delivery.
        reached_goal = (
            float(np.linalg.norm(new_obj[:2] - destination_xyz[:2]))
            <= cfg.goal_region_radius_m
        )

        dp_idx = int(base_datapoint_index + gen)
        datapoints.append(
            {
                "datapoint_id": f"{shard_id}_{dp_idx:06d}",
                "datapoint_index": dp_idx,
                "scene_index": int(scene_index),
                "window_index": int(gen),
                "rollout_step": int(gen * chunk),
                "in_contact": bool(in_contact),
                "reached_goal": bool(reached_goal),
                "movement_token": "stroke_sweep",
                "path_shape": path_shape,
                "instruction": body["instruction"],
                "tool_label": body.get("tool_label", TOOL_LABEL),
                "tool_contact_xyz_world": brush.tolist(),
                "tool_current_normal": brush_normal.tolist(),
                "tool_current_surface_dir": brush_surface_dir.tolist(),
                "has_material": True,
                "material_label": body["material_label"],
                "material_xyz_world": obj.tolist(),
                # Object position after executing this generation's chunk; lets
                # the rollout viz move the ball mid-trajectory based on the
                # same per-step analytic motion used by the closed-loop update.
                "material_xyz_after_world": new_obj.tolist(),
                "material_xyz_executed_world": material_trace.tolist(),
                "has_destination": True,
                "destination_label": body["destination_label"],
                "destination_xyz_world": destination_xyz.tolist(),
                "table_label": TABLE_LABEL,
                "table_xyz_world": list(table_xyz),
                "waypoints": target.reshape(n_wp, 9).tolist(),
            }
        )

        if reached_goal:
            break

        brush = new_brush[0:3].copy()
        brush_normal = new_brush[3:6].copy()
        brush_surface_dir = new_brush[6:9].copy()
        obj = new_obj
        in_contact = new_contact

        # Random disturbance after execution (reactive perturbation).
        sweep_unit = _sweep_unit(obj, destination_xyz, surface_normal)
        disturbed = _apply_disturbance(
            rng,
            dp_cfg,
            object_xyz=obj,
            in_contact=in_contact,
            sweep_unit=sweep_unit,
            avoid_xy=brush[:2],
        )
        if float(np.linalg.norm(disturbed - obj)) > 1e-5:
            obj = disturbed
            contact_xy_radius = float(
                BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M + cfg.contact_xy_tol_m
            )
            if float(np.linalg.norm(brush[:2] - obj[:2])) > contact_xy_radius:
                in_contact = False

    return datapoints


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    cfg: ReactiveGenConfig,
) -> dict[str, Any]:
    shard_id = f"brush_sweep_reactive_{shard_idx:04d}"
    datapoints: list[dict[str, Any]] = []
    dp_idx = 0
    for scene_i in range(int(cfg.scenes_per_shard)):
        rng = _datapoint_rng(seed, shard_idx, scene_i)
        scene_dps = scene_to_datapoints(
            rng,
            cfg,
            shard_id=shard_id,
            scene_index=scene_i,
            base_datapoint_index=dp_idx,
        )
        datapoints.extend(scene_dps)
        dp_idx += len(scene_dps)

    return {
        "dataset_id": DATASET_ID,
        "shard_id": shard_id,
        "scene_id": shard_id,
        "generator": "brush_sweep_reactive_v1",
        "seed": int(seed),
        "num_datapoints": len(datapoints),
        "scenes_per_shard": int(cfg.scenes_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reactive closed-loop stroke_sweep shards (dataset_0011)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0011_brush_sweep_reactive/shards",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--start_shard", type=int, default=0, help="Resume build from this shard index")
    parser.add_argument("--scenes_per_shard", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReactiveGenConfig(scenes_per_shard=int(args.scenes_per_shard))
    summary: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "num_shards": int(args.num_shards),
        "scenes_per_shard": int(args.scenes_per_shard),
        "seed": int(args.seed),
        "dense_step_spacing_m": cfg.dense_step_spacing_m,
        "window_stride": cfg.window_stride,
        "shard_paths": [],
    }

    for shard_idx in range(int(args.start_shard), int(args.num_shards)):
        shard = build_shard(shard_idx=shard_idx, seed=int(args.seed), cfg=cfg)
        out_path = out_dir / f"{shard['shard_id']}_shard.json"
        out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        summary["shard_paths"].append(str(out_path))
        print(
            f"Wrote {out_path} ({shard['num_datapoints']} datapoints, "
            f"{shard['scenes_per_shard']} scenes)"
        )

    summary_path = out_dir.parent / f"{DATASET_ID}_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
