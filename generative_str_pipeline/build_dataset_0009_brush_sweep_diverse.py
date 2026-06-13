"""Procedurally generate stroke_sweep-only brush trajectory shards for dataset_0009.

Expanded diversity vs dataset_0008: more templates/materials/destinations/tools,
wider spatial ranges, curved and multi-segment sweep paths, per-datapoint table height.
Schema omits material_normal, destination_normal, and table_normal (xyz only for refs).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import (
    DESTINATIONS_STROKE_SWEEP,
    MATERIALS_STROKE_SWEEP,
    TABLE_LABEL,
    TEMPLATES_BRUSH_STROKE_SWEEP,
    TOOL_LABEL,
    TOOLS_STROKE_SWEEP,
    _apply_intermediate_wp0,
    _datapoint_rng,
    _jitter_waypoint_z,
    _orthonormal_basis_in_plane,
    _project_surface_dir,
    _safe_format,
    _sample_near_up_normal,
    _sample_xy,
    _waypoints_from_contacts,
)

# Extended vocabulary (imported 0008 lists + additions)
EXTRA_TEMPLATES_STROKE_SWEEP = [
    "Sweep {material} over to {destination}",
    "Brush {material} across into {destination}",
    "Push {material} across the table into {destination}",
    "Guide {material} into {destination}",
    "Sweep from the pile into {destination}",
    "Clear {material} off into {destination}",
    "Scoop {material} toward {destination}",
    "Rake {material} into {destination}",
    "Shovel {material} into {destination}",
    "Corral {material} into {destination}",
    "Sweep the mess into {destination}",
    "Brush the spill into {destination}",
    "Move the pile into {destination}",
    "Sweep the scatter into {destination}",
    "Use {tool} to gather {material} into {destination}",
    "With {tool}, corral {material} into {destination}",
    "Carefully sweep {material} across into {destination}",
    "Quickly brush {material} into {destination}",
    "Gently push {material} toward {destination}",
    "Firmly sweep {material} into {destination}",
    "Sweep {material} along the table into {destination}",
    "Brush {material} from the center into {destination}",
    "Push {material} from the middle into {destination}",
    "Sweep {material} from the left into {destination}",
    "Sweep {material} from the right into {destination}",
    "Clear {material} from the surface into {destination}",
    "Gather {material} from the table into {destination}",
    "Collect {material} from the surface into {destination}",
    "Sweep up {material} and put it in {destination}",
    "Brush {material} up into {destination}",
]

EXTRA_MATERIALS_STROKE_SWEEP = [
    "the marbles", "the pebbles", "the nuts", "the bolts", "the washers",
    "the buttons", "the tokens", "the caps", "the tabs", "the rings",
    "the sprinkles", "the cat litter",
    "the coffee grounds", "the tea leaves", "the peppercorns", "the lentils",
    "the beans", "the chickpeas", "the popcorn", "the packing peanuts",
    "the foam beads", "the rubber chips", "the plastic bits", "the shreds",
    "the filings", "the granules", "the crystals", "the chunks",
    "the loose beads", "the scattered coins", "the pile of sand",
    "the heap of crumbs", "the mound of dirt", "the dust pile",
    "the grain spill", "the seed spill", "the powder spill",
    "some pebbles", "some marbles", "some lentils", "some beans",
    "the fine gravel", "the coarse salt", "the loose soil",
]

EXTRA_DESTINATIONS_STROKE_SWEEP = [
    "the chute", "the funnel", "the hopper", "the receptacle", "the holder",
    "the rack", "the slot", "the opening", "the pocket", "the well",
    "the catch tray", "the collection tray", "the waste tray", "the scrap bin",
    "the recycling bin", "the waste basket", "the liner", "the mat",
    "the far corner", "the near corner", "the back corner", "the front corner",
    "the left side", "the right side", "the center tray", "the side tray",
    "the shallow bowl", "the deep bowl", "the wide tray", "the narrow tray",
    "the silicone mat", "the cutting board edge", "the prep bowl",
    "the mixing bowl", "the serving bowl", "the colander",
]

EXTRA_TOOLS_STROKE_SWEEP = [
    "the hand broom", "the counter brush", "the pastry brush",
    "the deck brush", "the shop brush", "the utility brush",
    "the dustpan brush", "the table brush", "the sweep brush",
    "the angled brush", "the chip brush",
    "the whisk broom", "the corn broom", "the straw broom",
    "the silicone brush", "the rubber brush",
]

TEMPLATES_SWEEP = TEMPLATES_BRUSH_STROKE_SWEEP + EXTRA_TEMPLATES_STROKE_SWEEP
MATERIALS_SWEEP = MATERIALS_STROKE_SWEEP + EXTRA_MATERIALS_STROKE_SWEEP
DESTINATIONS_SWEEP = DESTINATIONS_STROKE_SWEEP + EXTRA_DESTINATIONS_STROKE_SWEEP
TOOLS_SWEEP = list(dict.fromkeys(TOOLS_STROKE_SWEEP + EXTRA_TOOLS_STROKE_SWEEP))

PATH_STRAIGHT = "straight"
PATH_ARC = "arc"
PATH_TWO_SEGMENT = "two_segment"


@dataclass
class SweepGenConfig:
    """Broader ranges than BrushGenConfig for diverse stroke_sweep data."""

    table_xyz_world: tuple[float, float, float] = (0.0, 0.0, 0.53)
    table_z_range: tuple[float, float] = (0.45, 0.60)
    table_xy_jitter_m: float = 0.03
    table_extent_m: float = 0.32
    approach_height_m_range: tuple[float, float] = (0.04, 0.28)
    contact_offset_m_range: tuple[float, float] = (0.001, 0.022)
    sweep_dist_m_range: tuple[float, float] = (0.05, 0.28)
    tool_home_z_above_table_m_range: tuple[float, float] = (0.08, 0.32)
    tool_start_xy_extent_m: float = 0.28
    surface_tilt_max_deg: float = 15.0
    table_clearance_m: float = 0.004
    table_lift_jitter_m_range: tuple[float, float] = (0.0, 0.08)
    waypoint_z_jitter_m: float = 0.025
    behind_offset_m_range: tuple[float, float] = (0.015, 0.10)
    mid_frac_range: tuple[float, float] = (0.30, 0.70)
    arc_lateral_frac_range: tuple[float, float] = (0.05, 0.30)
    path_shape_probs: tuple[float, float, float] = (0.50, 0.30, 0.20)
    two_segment_lift_frac_range: tuple[float, float] = (0.35, 0.55)
    two_segment_second_frac_range: tuple[float, float] = (0.65, 0.90)

    def sample_table_xyz(self, rng: np.random.Generator) -> tuple[float, float, float]:
        xy = rng.uniform(-self.table_xy_jitter_m, self.table_xy_jitter_m, size=2)
        z = float(rng.uniform(*self.table_z_range))
        return (float(xy[0]), float(xy[1]), z)


def _sample_path_shape(rng: np.random.Generator, cfg: SweepGenConfig) -> str:
    p_straight, p_arc, p_two = cfg.path_shape_probs
    total = p_straight + p_arc + p_two
    r = float(rng.random()) * total
    if r < p_straight:
        return PATH_STRAIGHT
    if r < p_straight + p_arc:
        return PATH_ARC
    return PATH_TWO_SEGMENT


def _arc_midpoint(
    material_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    surface_normal: np.ndarray,
    rng: np.random.Generator,
    cfg: SweepGenConfig,
) -> np.ndarray:
    """Lateral-bowed midpoint between material and destination."""
    sweep_vec = destination_xyz - material_xyz
    sweep_vec = sweep_vec - np.dot(sweep_vec, surface_normal) * surface_normal
    sweep_norm = float(np.linalg.norm(sweep_vec))
    if sweep_norm < 1e-6:
        return 0.5 * (material_xyz + destination_xyz)
    sweep_unit = (sweep_vec / sweep_norm).astype(np.float32)
    u, v = _orthonormal_basis_in_plane(surface_normal)
    lateral = np.cross(surface_normal, sweep_unit).astype(np.float32)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm < 1e-6:
        lateral = v
    else:
        lateral = lateral / lat_norm
    mid_frac = float(rng.uniform(*cfg.mid_frac_range))
    base_mid = (1.0 - mid_frac) * material_xyz + mid_frac * destination_xyz
    bow_frac = float(rng.uniform(*cfg.arc_lateral_frac_range))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    offset = sign * bow_frac * sweep_norm * lateral
    return (base_mid + offset).astype(np.float32)


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
    approach_h = float(rng.uniform(*cfg.approach_height_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    approach_offset = surface_normal * approach_h
    lift_offset = surface_normal * lift_h
    contact_offset = surface_normal * contact_h

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
    first_xyz = (behind_xyz + contact_offset).astype(np.float32)

    if path_shape == PATH_STRAIGHT:
        mid_frac = float(rng.uniform(*cfg.mid_frac_range))
        midpoint_xyz = (1.0 - mid_frac) * material_xyz + mid_frac * destination_xyz
        wp_contacts = np.stack(
            [
                first_xyz,
                first_xyz,
                material_xyz + contact_offset,
                midpoint_xyz + contact_offset,
                destination_xyz + contact_offset,
                destination_xyz + lift_offset,
            ],
            axis=0,
        )
    elif path_shape == PATH_ARC:
        midpoint_xyz = _arc_midpoint(
            material_xyz, destination_xyz, surface_normal, rng, cfg
        )
        q1 = 0.25 * material_xyz + 0.75 * midpoint_xyz
        q2 = 0.75 * midpoint_xyz + 0.25 * destination_xyz
        wp_contacts = np.stack(
            [
                first_xyz,
                first_xyz,
                material_xyz + contact_offset,
                q1 + contact_offset,
                q2 + contact_offset,
                destination_xyz + lift_offset,
            ],
            axis=0,
        )
    else:
        lift_frac = float(rng.uniform(*cfg.two_segment_lift_frac_range))
        second_frac = float(rng.uniform(*cfg.two_segment_second_frac_range))
        lift_xyz = (
            (1.0 - lift_frac) * material_xyz + lift_frac * destination_xyz + lift_offset
        ).astype(np.float32)
        partial_xyz = (
            (1.0 - second_frac) * material_xyz + second_frac * destination_xyz
        ).astype(np.float32)
        wp_contacts = np.stack(
            [
                first_xyz,
                first_xyz,
                material_xyz + contact_offset,
                lift_xyz,
                partial_xyz + contact_offset,
                destination_xyz + lift_offset,
            ],
            axis=0,
        )

    wp_contacts = wp_contacts.astype(np.float32)
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    return wp_contacts


def _sample_tool_pose_sweep(
    rng: np.random.Generator, cfg: SweepGenConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wider tool home sampling using table_extent + optional xy extent."""
    table_z = float(cfg.table_xyz_world[2])
    extent = max(cfg.table_extent_m, cfg.tool_start_xy_extent_m)
    xy = _sample_xy(rng, extent)
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


def gen_stroke_sweep_diverse(
    rng: np.random.Generator,
    cfg: SweepGenConfig,
    tool_home_pose: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    material_word = str(rng.choice(MATERIALS_SWEEP))
    destination_word = str(rng.choice(DESTINATIONS_SWEEP))

    material_xy = _sample_xy(rng, cfg.table_extent_m)
    dist = float(rng.uniform(*cfg.sweep_dist_m_range))
    theta = float(rng.uniform(0.0, 2 * np.pi))
    destination_xy = material_xy + dist * np.array(
        [np.cos(theta), np.sin(theta)], dtype=np.float32
    )
    destination_xy = np.clip(
        destination_xy, -cfg.table_extent_m, cfg.table_extent_m
    )

    material_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    destination_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    material_xyz = np.array(
        [material_xy[0], material_xy[1], material_z], dtype=np.float32
    )
    destination_xyz = np.array(
        [destination_xy[0], destination_xy[1], destination_z], dtype=np.float32
    )

    surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface_dir = _project_surface_dir(destination_xyz, material_xyz, surface_normal)

    path_shape = _sample_path_shape(rng, cfg)
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
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[1],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_SWEEP))
    if "{tool}" in template:
        tool_word = str(rng.choice(TOOLS_SWEEP))
    else:
        tool_word = TOOL_LABEL
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

    tool_contact, tool_normal, tool_surface_dir = _sample_tool_pose_sweep(rng, dp_cfg)
    body = gen_stroke_sweep_diverse(
        rng,
        dp_cfg,
        (tool_contact, tool_normal, tool_surface_dir),
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
    shard_id = f"brush_sweep_diverse_{shard_idx:04d}"
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
        "dataset_id": "dataset_0009_brush_sweep_diverse",
        "shard_id": shard_id,
        "scene_id": shard_id,
        "generator": "brush_sweep_diverse_v1",
        "seed": int(seed),
        "num_datapoints": int(datapoints_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stroke_sweep-only diverse brush trajectory shards (dataset_0009)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0009_brush_sweep_diverse/shards",
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
        "dataset_id": "dataset_0009_brush_sweep_diverse",
        "num_shards": int(args.num_shards),
        "datapoints_per_shard": int(args.datapoints_per_shard),
        "seed": int(args.seed),
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

    summary_path = out_dir.parent / "dataset_0009_brush_sweep_diverse_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
