"""Author a sim scene (brush keypoints + material + goal region + prompt) and
emit it as a one-datapoint *mini-shard* JSON.

This is the "close the loop" entry point: instead of pulling a datapoint from a
procedural dataset shard, you place a brush, a material pile, and a goal region
in the (sim/world) scene, take their keypoints, and write them in the exact
schema ``load_waypoint_samples`` expects. The resulting file can be fed straight
to ``predict_waypoints --shard_path <scene> --datapoint_index 0`` to have the
VLA generate a sweep trajectory for that scene.

Frame: everything is expressed in sim/world coordinates. The action_trajectory
VLA (dataset_0009) was normalized with the table at z=0.53, i.e. the same frame
as the IsaacGym SimToolReal table, so authored coordinates are in-distribution.

Tool keypoints follow the training convention for a brush in sweeping contact:
``tool_current_normal`` ~ +z (bristle face normal points up off the table) and
``tool_current_surface_dir`` is the in-plane sweep direction. The brush is held
slightly above the table surface (contact z ~ table + ~0.10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _unit(v: List[float]) -> List[float]:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError(f"near-zero vector: {v}")
    return (a / n).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Author a sim scene as a mini-shard JSON.")
    p.add_argument(
        "--instruction",
        type=str,
        default="Sweep the pile of crumbs into the dustpan",
    )
    p.add_argument("--scene_id", type=str, default="authored_scene_0")

    # Brush (tool) keypoints -- contact a bit above the table, normal up.
    p.add_argument("--tool_label", type=str, default="the brush")
    p.add_argument("--tool_contact", type=float, nargs=3, default=[0.12, 0.18, 0.63])
    p.add_argument("--tool_normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--tool_surface_dir", type=float, nargs=3, default=[-1.0, 0.0, 0.0])

    # Material pile (gets swept).
    p.add_argument("--material_label", type=str, default="the pile of crumbs")
    p.add_argument("--material_xyz", type=float, nargs=3, default=[0.0, 0.05, 0.53])

    # Goal region / destination (swept into).
    p.add_argument("--destination_label", type=str, default="the dustpan")
    p.add_argument("--destination_xyz", type=float, nargs=3, default=[-0.24, -0.02, 0.53])

    # Table reference (sim surface center).
    p.add_argument("--table_label", type=str, default="table surface center")
    p.add_argument("--table_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.53])

    p.add_argument(
        "--output",
        type=str,
        default="training/verification/sim_rollout/scenes/authored_scene_0_shard.json",
    )
    args = p.parse_args()

    datapoint = {
        "datapoint_index": 0,
        "movement_token": "stroke_sweep",
        "instruction": args.instruction,
        "tool_label": args.tool_label,
        "tool_contact_xyz_world": [float(x) for x in args.tool_contact],
        "tool_current_normal": _unit(args.tool_normal),
        "tool_current_surface_dir": _unit(args.tool_surface_dir),
        "material_label": args.material_label,
        "material_xyz_world": [float(x) for x in args.material_xyz],
        "has_material": True,
        "destination_label": args.destination_label,
        "destination_xyz_world": [float(x) for x in args.destination_xyz],
        "has_destination": True,
        "table_label": args.table_label,
        "table_xyz_world": [float(x) for x in args.table_xyz],
        "table_normal": [0.0, 0.0, 1.0],
        # Ground-truth waypoints are unused at inference; emit zeros to satisfy
        # the dataset loader's [6, 9] shape requirement.
        "waypoints": np.zeros((6, 9), dtype=float).tolist(),
    }

    shard = {"scene_id": args.scene_id, "datapoints": [datapoint]}

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
    print(f"Wrote scene mini-shard: {out_path}")
    print(f"  instruction : {args.instruction}")
    print(f"  tool_contact: {datapoint['tool_contact_xyz_world']}")
    print(f"  material    : {datapoint['material_xyz_world']}")
    print(f"  destination : {datapoint['destination_xyz_world']}")


if __name__ == "__main__":
    main()
