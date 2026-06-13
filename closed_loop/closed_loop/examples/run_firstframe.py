#!/usr/bin/env python3
"""Open-loop first-frame demo: predict 15 object root poses, save JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from closed_loop.frames import contact_frame_direct, robot_to_model_xyz
from closed_loop.inference import BrushPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-frame", default="blue_brush")
    parser.add_argument("--y-shift", type=float, default=0.8)
    parser.add_argument("--tool-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--tool-quat-xyzw", type=float, nargs=4, required=True)
    parser.add_argument("--material-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--destination-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--instruction", default="Sweep the cube to the goal with the brush")
    parser.add_argument("--out-json", type=Path, default=Path("brush_object_goal_poses.json"))
    args = parser.parse_args()

    shift = np.array([0.0, args.y_shift, 0.0], dtype=np.float64)
    contact, normal, sd = contact_frame_direct(
        robot_to_model_xyz(np.asarray(args.tool_xyz), shift),
        np.asarray(args.tool_quat_xyzw),
    )
    material = robot_to_model_xyz(np.asarray(args.material_xyz), shift).astype(np.float32)
    destination = robot_to_model_xyz(np.asarray(args.destination_xyz), shift).astype(np.float32)

    from closed_loop.scene import SceneState

    policy = BrushPolicy(device=args.device, control_frame=args.control_frame, instruction=args.instruction)
    scene = SceneState(
        instruction=args.instruction,
        tool_label=policy.tool_label,
        tool_contact_xyz_world=contact,
        tool_current_normal=normal,
        tool_current_surface_dir=sd,
        material_xyz_world=material,
        destination_xyz_world=destination,
        table_xyz_world=np.array([0.0, 0.0, policy.table_z], dtype=np.float32),
    )
    waypoints = policy.predict_waypoints(scene)
    poses = policy.waypoints_to_object_poses_robot(waypoints, shift)

    out = {
        "description": "15 open-loop brush object root poses (robot frame)",
        "frame": "robot",
        "y_shift_removed": float(args.y_shift),
        "control_frame": str(policy.control_frame_path),
        "poses": [
            {
                "index": i,
                "position": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
                "orientation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            }
            for i, (xyz, quat) in enumerate(poses)
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json} ({len(poses)} poses)")


if __name__ == "__main__":
    main()
