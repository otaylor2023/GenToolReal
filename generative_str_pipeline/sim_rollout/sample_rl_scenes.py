"""Sample random RL training scenes (keypoints + prompt) for VLA GRPO rollouts.

Samples brush home, ball (material), and goal (destination) on the sim table FIRST.
Waypoints are left zero — the VLA predicts them at rollout time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import (
    DESTINATIONS_STROKE_SWEEP,
    MATERIALS_STROKE_SWEEP,
    TEMPLATES_BRUSH_STROKE_SWEEP,
    TOOL_LABEL,
    _orthonormal_basis_in_plane,
    _project_surface_dir,
    _safe_format,
    _sample_near_up_normal,
)
from generative_str_pipeline.build_dataset_0010_brush_sweep_sim import (
    SweepGenConfig,
)
from generative_str_pipeline.sim_workspace import (
    BRUSH_STAGE_X_RANGE_M,
    BRUSH_STAGE_Y_RANGE_M,
    GOAL_REGION_MARGIN_M,
    GOAL_REGION_RADIUS_M,
    OBJ_REGION_X_RANGE_M,
    OBJ_REGION_Y_RANGE_M,
    TABLE_HALF_X_M,
    TABLE_HALF_Y_M,
    ball_clear_of_brush,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RLSceneSampleConfig:
    table_xyz_world: tuple[float, float, float] = (0.0, 0.0, 0.53)
    keypoint_xy_jitter_m: float = 0.02
    keypoint_z_jitter_m: float = 0.015
    material_radius_m: float = 0.02


def _jitter_xyz(xyz: np.ndarray, rng: np.random.Generator, cfg: RLSceneSampleConfig) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float32).reshape(3).copy()
    out[0] += float(rng.uniform(-cfg.keypoint_xy_jitter_m, cfg.keypoint_xy_jitter_m))
    out[1] += float(rng.uniform(-cfg.keypoint_xy_jitter_m, cfg.keypoint_xy_jitter_m))
    out[2] += float(rng.uniform(-cfg.keypoint_z_jitter_m, cfg.keypoint_z_jitter_m))
    out[2] = max(float(cfg.table_xyz_world[2]), float(out[2]))
    return out


# Goal-region (blue patch) must stay fully on the *original* table, and the goal
# stays in the same -x quadrant as the ball (opposite the brush).
def _obj_dest_bounds() -> tuple[float, float, float, float]:
    gx = TABLE_HALF_X_M - GOAL_REGION_RADIUS_M - GOAL_REGION_MARGIN_M
    gy = TABLE_HALF_Y_M - GOAL_REGION_RADIUS_M - GOAL_REGION_MARGIN_M
    x_lo = max(float(OBJ_REGION_X_RANGE_M[0]), -gx)
    x_hi = min(float(OBJ_REGION_X_RANGE_M[1]), gx)
    y_lo = max(float(OBJ_REGION_Y_RANGE_M[0]), -gy)
    y_hi = min(float(OBJ_REGION_Y_RANGE_M[1]), gy)
    return x_lo, x_hi, y_lo, y_hi


def _clip_obj_dest_xy(xy: np.ndarray) -> np.ndarray:
    x_lo, x_hi, y_lo, y_hi = _obj_dest_bounds()
    out = np.asarray(xy, dtype=np.float32).reshape(2).copy()
    out[0] = float(np.clip(out[0], x_lo, x_hi))
    out[1] = float(np.clip(out[1], y_lo, y_hi))
    return out


def _sample_brush_staging_pose(
    rng: np.random.Generator, sweep_cfg: SweepGenConfig, table_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brush home: head on the +x table extension, handle over the original region.

    The contact (front edge of the head) is sampled on the +x staging band; the
    +x heading means the handle extends back (-x) into the original table area.
    """
    x = float(rng.uniform(*BRUSH_STAGE_X_RANGE_M))
    y = float(rng.uniform(*BRUSH_STAGE_Y_RANGE_M))
    contact = np.array([x, y, table_z], dtype=np.float32)
    normal = _sample_near_up_normal(rng, sweep_cfg.surface_tilt_max_deg)
    # Heading +x: the brush handle extends back (-x) into the original table
    # region while the head (contact) sits on the +x extension strip.
    surf = _project_surface_dir(
        contact - np.array([0.05, 0.0, 0.0], dtype=np.float32), contact, normal
    )
    if float(np.linalg.norm(surf)) < 1e-6:
        u, _ = _orthonormal_basis_in_plane(normal)
        surf = u
    return contact, normal, surf


def _sample_objects_opposite_quadrant(
    rng: np.random.Generator,
    sweep_cfg: SweepGenConfig,
    table_z: float,
    *,
    avoid_xy: np.ndarray,
    tool_heading_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ball + goal in the original -x region (a different quadrant than the brush)."""

    def _samp_xy() -> np.ndarray:
        return np.array(
            [
                float(rng.uniform(*OBJ_REGION_X_RANGE_M)),
                float(rng.uniform(*OBJ_REGION_Y_RANGE_M)),
            ],
            dtype=np.float32,
        )

    avoid = np.asarray(avoid_xy, dtype=np.float32).reshape(2)
    material_xy = _samp_xy()
    for _ in range(128):
        far = float(np.linalg.norm(material_xy - avoid)) >= sweep_cfg.min_tool_material_sep_m
        clear = ball_clear_of_brush(material_xy, avoid, tool_heading_xy)
        if far and clear:
            break
        material_xy = _samp_xy()

    dest_xy = material_xy.copy()
    for _ in range(48):
        dist = float(rng.uniform(*sweep_cfg.sweep_dist_m_range))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        cand = material_xy + dist * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32
        )
        cand = _clip_obj_dest_xy(cand)
        if float(np.linalg.norm(cand - material_xy)) >= sweep_cfg.min_sweep_dist_m:
            dest_xy = cand
            break
        dest_xy = cand

    material_z = table_z + float(rng.uniform(*sweep_cfg.table_lift_jitter_m_range))
    dest_z = table_z + float(rng.uniform(*sweep_cfg.table_lift_jitter_m_range))
    material_xyz = np.array([material_xy[0], material_xy[1], material_z], dtype=np.float32)
    destination_xyz = np.array([dest_xy[0], dest_xy[1], dest_z], dtype=np.float32)
    return material_xyz, destination_xyz


def sample_rl_scene(
    rng: np.random.Generator,
    *,
    scene_cfg: RLSceneSampleConfig | None = None,
    sweep_cfg: SweepGenConfig | None = None,
) -> Dict[str, Any]:
    """One scene dict compatible with WaypointTrajectoryDataset fields."""
    scene_cfg = scene_cfg or RLSceneSampleConfig()
    table_z = float(scene_cfg.table_xyz_world[2])
    # Sim-only overrides: start the cube further from the goal (longer sweep) and
    # keep the brush spawn well clear of the cube. Dataset defaults are unchanged.
    sweep_cfg = sweep_cfg or SweepGenConfig(
        table_xyz_world=scene_cfg.table_xyz_world,
        sweep_dist_m_range=(0.16, 0.32),
        min_sweep_dist_m=0.14,
        min_tool_material_sep_m=0.24,
    )

    # Sim-only wide-table layout: brush staged on the +x extension (head on the
    # added strip, handle over the original region); ball + goal in the original
    # -x region (a different quadrant). The dataset itself is unchanged.
    tool_contact, tool_normal, tool_surface_dir = _sample_brush_staging_pose(
        rng, sweep_cfg, table_z
    )
    material_xyz, destination_xyz = _sample_objects_opposite_quadrant(
        rng,
        sweep_cfg,
        table_z,
        avoid_xy=tool_contact[:2],
        tool_heading_xy=tool_surface_dir[:2],
    )

    tool_contact = _jitter_xyz(tool_contact, rng, scene_cfg)
    tool_contact[0] = float(
        np.clip(tool_contact[0], BRUSH_STAGE_X_RANGE_M[0], BRUSH_STAGE_X_RANGE_M[1])
    )
    tool_contact[1] = float(
        np.clip(tool_contact[1], BRUSH_STAGE_Y_RANGE_M[0], BRUSH_STAGE_Y_RANGE_M[1])
    )
    mat = _jitter_xyz(material_xyz, rng, scene_cfg)
    dest = _jitter_xyz(destination_xyz, rng, scene_cfg)
    # Re-clip the (post-jitter) destination so the goal-region patch, which is
    # used for both the reward target and the sim marker, stays fully on the
    # original table and in the ball's quadrant.
    dest[:2] = _clip_obj_dest_xy(dest[:2])
    # Enforce a minimum cube->goal gap after jitter so the cube never starts
    # essentially on the goal (the jitter can otherwise erode the sampled sweep).
    min_gap = float(sweep_cfg.min_sweep_dist_m)
    d = dest[:2] - mat[:2]
    gap = float(np.linalg.norm(d))
    if gap < min_gap:
        if gap < 1e-6:
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            dirxy = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        else:
            dirxy = (d / gap).astype(np.float32)
        dest[:2] = _clip_obj_dest_xy(mat[:2] + dirxy * min_gap)

    material_word = str(rng.choice(MATERIALS_STROKE_SWEEP))
    destination_word = str(rng.choice(DESTINATIONS_STROKE_SWEEP))
    template = str(rng.choice(TEMPLATES_BRUSH_STROKE_SWEEP))
    instruction = _safe_format(
        template,
        material=material_word,
        destination=destination_word,
        tool=TOOL_LABEL,
    )

    return {
        "movement_token": "stroke_sweep",
        "instruction": instruction,
        "tool_label": TOOL_LABEL,
        "tool_contact_xyz_world": tool_contact.tolist(),
        "tool_current_normal": tool_normal.tolist(),
        "tool_current_surface_dir": tool_surface_dir.tolist(),
        "material_label": material_word,
        "material_xyz_world": mat.tolist(),
        "has_material": True,
        "destination_label": destination_word,
        "destination_xyz_world": dest.tolist(),
        "has_destination": True,
        "table_label": "table surface center",
        "table_xyz_world": list(scene_cfg.table_xyz_world),
        "table_normal": [0.0, 0.0, 1.0],
        "waypoints": np.zeros((6, 9), dtype=np.float32).tolist(),
    }


def sample_rl_scenes(
    num_scenes: int,
    *,
    seed: int = 0,
    scene_cfg: RLSceneSampleConfig | None = None,
    table_z: float | None = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    if table_z is not None:
        scene_cfg = scene_cfg or RLSceneSampleConfig()
        scene_cfg = RLSceneSampleConfig(
            table_xyz_world=(scene_cfg.table_xyz_world[0], scene_cfg.table_xyz_world[1], float(table_z)),
            keypoint_xy_jitter_m=scene_cfg.keypoint_xy_jitter_m,
            keypoint_z_jitter_m=scene_cfg.keypoint_z_jitter_m,
            material_radius_m=scene_cfg.material_radius_m,
        )
    return [sample_rl_scene(rng, scene_cfg=scene_cfg) for _ in range(int(num_scenes))]


def scenes_to_mini_shard(scenes: List[Dict[str, Any]], scene_id: str = "rl_batch") -> Dict[str, Any]:
    out = []
    for i, s in enumerate(scenes):
        dp = dict(s)
        dp["datapoint_index"] = int(i)
        out.append(dp)
    return {"scene_id": scene_id, "datapoints": out}


def main() -> None:
    p = argparse.ArgumentParser(description="Sample RL scenes to a mini-shard JSON.")
    p.add_argument("--num_scenes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()
    scenes = sample_rl_scenes(args.num_scenes, seed=args.seed)
    shard = scenes_to_mini_shard(scenes)
    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shard, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(scenes)} scenes)")


if __name__ == "__main__":
    main()
