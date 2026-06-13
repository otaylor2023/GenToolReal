"""Procedurally generate stroke_sweep shards for dataset_0010 (sim-aligned).

- Workspace clipped to SimToolReal narrow-table extents (0.475 x 0.4 m).
- Brush starts 3-10 cm behind the material, then moves in to sweep.
- Inputs sampled first: tool home, ball (material), goal (destination), then GT waypoints.
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
    PATH_ARC,
    PATH_STRAIGHT,
    PATH_TWO_SEGMENT,
    TEMPLATES_SWEEP,
    TOOLS_SWEEP,
    _arc_midpoint,
    _sample_path_shape,
)
from generative_str_pipeline.build_dataset_0008_brush_procedural import (
    TABLE_LABEL,
    TOOL_LABEL,
    _datapoint_rng,
    _jitter_waypoint_z,
    _orthonormal_basis_in_plane,
    _project_surface_dir,
    _safe_format,
    _sample_near_up_normal,
    _waypoints_from_contacts,
)
from generative_str_pipeline.sim_workspace import (
    BEHIND_OFFSET_M_RANGE,
    TABLE_XY_JITTER_M,
    TABLE_Z_RANGE,
    WORKSPACE_X_EXTENT_M,
    WORKSPACE_Y_EXTENT_M,
    ball_clear_of_brush,
    clip_waypoints_xy,
    clip_xy_rect,
    dest_extent_for_goal_region,
    sample_xy_rect,
)

DATASET_ID = "dataset_0010_brush_sweep_sim"


@dataclass
class SweepGenConfig:
    """Sim-aligned stroke_sweep generator (narrow table workspace)."""

    table_xyz_world: tuple[float, float, float] = (0.0, 0.0, 0.53)
    table_z_range: tuple[float, float] = TABLE_Z_RANGE
    table_xy_jitter_m: float = TABLE_XY_JITTER_M
    workspace_x_extent_m: float = WORKSPACE_X_EXTENT_M
    workspace_y_extent_m: float = WORKSPACE_Y_EXTENT_M
    approach_height_m_range: tuple[float, float] = (0.04, 0.28)
    # Elevated approach height for the first two waypoints (in the air near /
    # behind the ball before lowering to contact).
    approach_air_height_m_range: tuple[float, float] = (0.04, 0.10)
    contact_offset_m_range: tuple[float, float] = (0.001, 0.022)
    sweep_dist_m_range: tuple[float, float] = (0.05, 0.22)
    # Tool home rests ON the table (small clearance above the surface), since the
    # VLA conditions on where the brush actually sits before being picked up.
    tool_home_z_above_table_m_range: tuple[float, float] = (0.0, 0.012)
    # Keep the ball away from the tool home so it is not disturbed on pick-up.
    min_tool_material_sep_m: float = 0.14
    surface_tilt_max_deg: float = 12.0
    table_clearance_m: float = 0.004
    table_lift_jitter_m_range: tuple[float, float] = (0.0, 0.06)
    waypoint_z_jitter_m: float = 0.02
    behind_offset_m_range: tuple[float, float] = BEHIND_OFFSET_M_RANGE
    mid_frac_range: tuple[float, float] = (0.30, 0.70)
    arc_lateral_frac_range: tuple[float, float] = (0.05, 0.25)
    path_shape_probs: tuple[float, float, float] = (0.55, 0.30, 0.15)
    two_segment_lift_frac_range: tuple[float, float] = (0.35, 0.55)
    two_segment_second_frac_range: tuple[float, float] = (0.65, 0.90)
    min_sweep_dist_m: float = 0.04

    def sample_table_xyz(self, rng: np.random.Generator) -> tuple[float, float, float]:
        xy = rng.uniform(-self.table_xy_jitter_m, self.table_xy_jitter_m, size=2)
        z = float(rng.uniform(*self.table_z_range))
        return (float(xy[0]), float(xy[1]), z)

    def clip_xy(self, xy: np.ndarray) -> np.ndarray:
        return clip_xy_rect(
            xy,
            x_extent=self.workspace_x_extent_m,
            y_extent=self.workspace_y_extent_m,
        )

    def clip_dest_xy(self, xy: np.ndarray) -> np.ndarray:
        """Clip the destination so the goal region (blue patch) stays on table."""
        dx, dy = dest_extent_for_goal_region(
            self.workspace_x_extent_m, self.workspace_y_extent_m
        )
        return clip_xy_rect(xy, x_extent=dx, y_extent=dy)


def _sample_tool_pose_sweep(
    rng: np.random.Generator, cfg: SweepGenConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_z = float(cfg.table_xyz_world[2])
    xy = sample_xy_rect(
        rng,
        x_extent=cfg.workspace_x_extent_m,
        y_extent=cfg.workspace_y_extent_m,
    )
    z_above = float(rng.uniform(*cfg.tool_home_z_above_table_m_range))
    contact = np.array([xy[0], xy[1], table_z + z_above], dtype=np.float32)
    normal = _sample_near_up_normal(rng, cfg.surface_tilt_max_deg)
    surf = _project_surface_dir(
        contact + np.array([0.05, 0.0, 0.0], dtype=np.float32),
        contact,
        normal,
    )
    if float(np.linalg.norm(surf)) < 1e-6:
        u, _ = _orthonormal_basis_in_plane(normal)
        surf = u
    return contact, normal, surf


def _sample_material_destination(
    rng: np.random.Generator,
    cfg: SweepGenConfig,
    *,
    avoid_xy: np.ndarray | None = None,
    tool_heading_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ball and goal xy on the table, then z above table surface.

    If ``avoid_xy`` (the tool home contact xy) and ``tool_heading_xy`` (the brush
    in-plane heading) are given, the ball is kept fully outside the brush's
    oriented footprint so picking up the tool does not disturb it.
    """
    table_z = float(cfg.table_xyz_world[2])
    material_xy = sample_xy_rect(
        rng,
        x_extent=cfg.workspace_x_extent_m,
        y_extent=cfg.workspace_y_extent_m,
    )
    if avoid_xy is not None:
        avoid = np.asarray(avoid_xy, dtype=np.float32).reshape(2)
        for _ in range(96):
            far_enough = (
                float(np.linalg.norm(material_xy - avoid)) >= cfg.min_tool_material_sep_m
            )
            clear = (
                tool_heading_xy is None
                or ball_clear_of_brush(material_xy, avoid, tool_heading_xy)
            )
            if far_enough and clear:
                break
            material_xy = sample_xy_rect(
                rng,
                x_extent=cfg.workspace_x_extent_m,
                y_extent=cfg.workspace_y_extent_m,
            )
    for _ in range(48):
        dist = float(rng.uniform(*cfg.sweep_dist_m_range))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        destination_xy = material_xy + dist * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32
        )
        destination_xy = cfg.clip_dest_xy(destination_xy)
        if float(np.linalg.norm(destination_xy - material_xy)) >= cfg.min_sweep_dist_m:
            break

    material_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    destination_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    material_xyz = np.array(
        [material_xy[0], material_xy[1], material_z], dtype=np.float32
    )
    destination_xyz = np.array(
        [destination_xy[0], destination_xy[1], destination_z], dtype=np.float32
    )
    return material_xyz, destination_xyz


def _build_sweep_contacts(
    rng: np.random.Generator,
    cfg: SweepGenConfig,
    *,
    material_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    surface_normal: np.ndarray,
    surface_dir: np.ndarray,
    path_shape: str,
) -> np.ndarray:
    table_z = float(cfg.table_xyz_world[2])
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    air_h = float(rng.uniform(*cfg.approach_air_height_m_range))
    contact_offset = surface_normal * contact_h
    air_offset = surface_normal * air_h

    sweep_vec = destination_xyz - material_xyz
    sweep_vec = sweep_vec - np.dot(sweep_vec, surface_normal) * surface_normal
    sweep_norm = float(np.linalg.norm(sweep_vec))
    sweep_unit = (
        (sweep_vec / sweep_norm).astype(np.float32)
        if sweep_norm > 1e-6
        else surface_dir
    )

    behind_offset_m = float(rng.uniform(*cfg.behind_offset_m_range))
    behind_xyz = (material_xyz - sweep_unit * behind_offset_m).astype(np.float32)

    # Approach (shared across path shapes): from the picked-up tool, move to a
    # pose in the air near the ball, then in the air behind the ball, then lower
    # to contact behind the ball ready to sweep.
    near_air_xyz = (material_xyz + air_offset).astype(np.float32)
    behind_air_xyz = (behind_xyz + air_offset).astype(np.float32)
    behind_contact_xyz = (behind_xyz + contact_offset).astype(np.float32)

    if path_shape == PATH_ARC:
        midpoint_xyz = _arc_midpoint(
            material_xyz, destination_xyz, surface_normal, rng, cfg  # type: ignore[arg-type]
        )
        mid_contact_xyz = 0.5 * midpoint_xyz + 0.5 * destination_xyz + contact_offset
    elif path_shape == PATH_TWO_SEGMENT:
        second_frac = float(rng.uniform(*cfg.two_segment_second_frac_range))
        mid_contact_xyz = (
            (1.0 - second_frac) * material_xyz + second_frac * destination_xyz
        ) + contact_offset
    else:  # straight
        mid_frac = float(rng.uniform(*cfg.mid_frac_range))
        mid_contact_xyz = (
            (1.0 - mid_frac) * material_xyz + mid_frac * destination_xyz
        ) + contact_offset

    wp_contacts = np.stack(
        [
            near_air_xyz,
            behind_air_xyz,
            behind_contact_xyz,
            (material_xyz + contact_offset).astype(np.float32),
            mid_contact_xyz.astype(np.float32),
            (destination_xyz + contact_offset).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    # Jitter only the on-table contact waypoints (2..5); keep the elevated
    # approach (0,1) clean so they stay in the air at the intended height.
    _jitter_waypoint_z(
        wp_contacts[2:],
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    clip_waypoints_xy(
        wp_contacts,
        x_extent=cfg.workspace_x_extent_m,
        y_extent=cfg.workspace_y_extent_m,
    )
    return wp_contacts


def gen_stroke_sweep_from_inputs(
    rng: np.random.Generator,
    cfg: SweepGenConfig,
    tool_home_pose: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    material_xyz: np.ndarray,
    destination_xyz: np.ndarray,
) -> dict[str, Any]:
    """Build instruction + GT waypoints from pre-sampled tool/material/destination."""
    material_word = str(rng.choice(MATERIALS_SWEEP))
    destination_word = str(rng.choice(DESTINATIONS_SWEEP))

    surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface_dir = _project_surface_dir(destination_xyz, material_xyz, surface_normal)

    path_shape = _sample_path_shape(rng, cfg)  # type: ignore[arg-type]
    wp_contacts = _build_sweep_contacts(
        rng,
        cfg,
        material_xyz=material_xyz,
        destination_xyz=destination_xyz,
        surface_normal=surface_normal,
        surface_dir=surface_dir,
        path_shape=path_shape,
    )
    waypoints = _waypoints_from_contacts(
        wp_contacts, surface_normal, surface_dir=surface_dir
    )
    _ = tool_home_pose  # wp0 is now the elevated near-ball approach, not an intermediate

    template = str(rng.choice(TEMPLATES_SWEEP))
    tool_word = str(rng.choice(TOOLS_SWEEP)) if "{tool}" in template else TOOL_LABEL
    instruction = _safe_format(
        template,
        material=material_word,
        destination=destination_word,
        tool=tool_word,
    )

    return {
        "movement_token": "stroke_sweep",
        "instruction": instruction,
        "tool_label": tool_word,
        "path_shape": path_shape,
        "has_material": True,
        "material_label": material_word,
        "material_xyz_world": material_xyz.tolist(),
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": destination_xyz.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


def build_datapoint(
    rng: np.random.Generator,
    cfg: SweepGenConfig,
    *,
    shard_id: str,
    datapoint_index: int,
) -> dict[str, Any]:
    table_xyz = cfg.sample_table_xyz(rng)
    dp_cfg = replace(cfg, table_xyz_world=table_xyz)

    # Inputs first: tool home (on table), ball (away from tool), goal — then
    # synthesize the sweep waypoints from those inputs.
    tool_contact, tool_normal, tool_surface_dir = _sample_tool_pose_sweep(rng, dp_cfg)
    material_xyz, destination_xyz = _sample_material_destination(
        rng,
        dp_cfg,
        avoid_xy=tool_contact[:2],
        tool_heading_xy=tool_surface_dir[:2],
    )
    body = gen_stroke_sweep_from_inputs(
        rng,
        dp_cfg,
        (tool_contact, tool_normal, tool_surface_dir),
        material_xyz=material_xyz,
        destination_xyz=destination_xyz,
    )

    datapoint_id = f"{shard_id}_{datapoint_index:06d}"
    return {
        "datapoint_id": datapoint_id,
        "datapoint_index": int(datapoint_index),
        "movement_token": "stroke_sweep",
        "path_shape": body.get("path_shape", PATH_STRAIGHT),
        "instruction": body["instruction"],
        "tool_label": body.get("tool_label", TOOL_LABEL),
        "tool_contact_xyz_world": tool_contact.tolist(),
        "tool_current_normal": tool_normal.tolist(),
        "tool_current_surface_dir": tool_surface_dir.tolist(),
        "has_material": True,
        "material_label": body["material_label"],
        "material_xyz_world": body["material_xyz_world"],
        "has_destination": True,
        "destination_label": body["destination_label"],
        "destination_xyz_world": body["destination_xyz_world"],
        "table_label": TABLE_LABEL,
        "table_xyz_world": list(table_xyz),
        "waypoints": body["waypoints"],
    }


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    datapoints_per_shard: int,
    cfg: SweepGenConfig,
) -> dict[str, Any]:
    shard_id = f"brush_sweep_sim_{shard_idx:04d}"
    datapoints: list[dict[str, Any]] = []
    for dp_idx in range(datapoints_per_shard):
        rng = _datapoint_rng(seed, shard_idx, dp_idx)
        datapoints.append(
            build_datapoint(
                rng,
                cfg,
                shard_id=shard_id,
                datapoint_index=dp_idx,
            )
        )
    return {
        "dataset_id": DATASET_ID,
        "shard_id": shard_id,
        "scene_id": shard_id,
        "generator": "brush_sweep_sim_v1",
        "seed": int(seed),
        "num_datapoints": int(datapoints_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sim-aligned stroke_sweep shards (dataset_0010)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0010_brush_sweep_sim/shards",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--datapoints_per_shard", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SweepGenConfig()
    summary: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "num_shards": int(args.num_shards),
        "datapoints_per_shard": int(args.datapoints_per_shard),
        "seed": int(args.seed),
        "workspace_x_extent_m": cfg.workspace_x_extent_m,
        "workspace_y_extent_m": cfg.workspace_y_extent_m,
        "behind_offset_m_range": list(cfg.behind_offset_m_range),
        "shard_paths": [],
    }

    for shard_idx in range(int(args.num_shards)):
        shard = build_shard(
            shard_idx=shard_idx,
            seed=int(args.seed),
            datapoints_per_shard=int(args.datapoints_per_shard),
            cfg=cfg,
        )
        out_path = out_dir / f"{shard['shard_id']}_shard.json"
        out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        summary["shard_paths"].append(str(out_path))
        print(f"Wrote {out_path} ({shard['num_datapoints']} datapoints)")

    summary_path = out_dir.parent / f"{DATASET_ID}_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
