"""Render VLA-predicted trajectory images for RL rollouts (subprocess-safe).

Supports single-plan ``waypoints`` or multi-generation closed-loop ``generations``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_trajectory.dataset import WaypointTrajectorySample
from generative_str_pipeline.visualize_brush_trajectories import (
    render_datapoint,
    render_rollout_panel,
)


def _arr(v, n=3):
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).reshape(n)


def _scene_to_sample(scene: dict, waypoints: np.ndarray, idx: int) -> WaypointTrajectorySample:
    return WaypointTrajectorySample(
        scene_id="rl_grpo",
        shard_path="",
        datapoint_index=int(idx),
        movement_token=str(scene.get("movement_token", "stroke_sweep")),
        instruction=str(scene.get("instruction", "")),
        tool_label=str(scene.get("tool_label", "the brush")),
        tool_contact_xyz_world=_arr(scene["tool_contact_xyz_world"]),
        tool_current_normal=_arr(scene["tool_current_normal"]),
        tool_current_surface_dir=_arr(scene["tool_current_surface_dir"]),
        material_label=scene.get("material_label"),
        material_xyz_world=_arr(scene.get("material_xyz_world")),
        material_normal=None,
        has_material=True,
        destination_label=scene.get("destination_label"),
        destination_xyz_world=_arr(scene.get("destination_xyz_world")),
        destination_normal=None,
        has_destination=True,
        table_label=str(scene.get("table_label", "table surface center")),
        table_xyz_world=_arr(scene.get("table_xyz_world", [0.0, 0.0, 0.53])),
        table_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        waypoints=np.asarray(waypoints, dtype=np.float32).reshape(-1, 9),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Render RL rollout trajectory images.")
    p.add_argument("--input", type=str, required=True)
    args = p.parse_args()

    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for i, item in enumerate(spec["items"]):
        generations = item.get("generations")
        if generations:
            sample = _scene_to_sample(
                item["scene"],
                np.asarray(item.get("waypoints", np.zeros((1, 9))), dtype=np.float32),
                i,
            )
            img = render_rollout_panel(
                sample,
                sample.movement_token,
                generations,
                draw_destination_surface=False,
            )
        else:
            sample = _scene_to_sample(
                item["scene"], np.asarray(item["waypoints"], dtype=np.float32), i
            )
            img = render_datapoint(
                sample, sample.movement_token, draw_destination_surface=False
            )
        out_path = Path(item["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
