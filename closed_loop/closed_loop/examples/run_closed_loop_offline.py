#!/usr/bin/env python3
"""Offline closed-loop test with self-contained analytic cube push."""

from __future__ import annotations

import argparse

import numpy as np

from closed_loop.analytic_push import GOAL_REGION_RADIUS_M, execute_chunk
from closed_loop.frames import contact_frame_direct, robot_to_model_xyz
from closed_loop.inference import BrushPolicy
from closed_loop.scene import SceneState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-frame", default="blue_brush")
    parser.add_argument("--y-shift", type=float, default=0.8)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--max-replans", type=int, default=30)
    # First-frame robot poses from verification run
    parser.add_argument(
        "--tool-xyz",
        type=float,
        nargs=3,
        default=[-0.03649736995233821, -0.7696878540025665, 0.5481157988738621],
    )
    parser.add_argument(
        "--tool-quat-xyzw",
        type=float,
        nargs=4,
        default=[0.018379613854218015, -0.03682508820482355, -0.21215851474702213, 0.9763682027255979],
    )
    parser.add_argument(
        "--material-xyz",
        type=float,
        nargs=3,
        default=[-0.22108551764398665, -0.7643075076086873, 0.5488788733528523],
    )
    parser.add_argument(
        "--destination-xyz",
        type=float,
        nargs=3,
        default=[-0.36499680187682326, -0.8561414030677632, 0.5170111614487249],
    )
    args = parser.parse_args()

    shift = np.array([0.0, args.y_shift, 0.0], dtype=np.float64)
    policy = BrushPolicy(device=args.device, control_frame=args.control_frame)

    contact, normal, sd = contact_frame_direct(
        robot_to_model_xyz(np.asarray(args.tool_xyz), shift),
        np.asarray(args.tool_quat_xyzw),
    )
    obj = robot_to_model_xyz(np.asarray(args.material_xyz), shift).astype(np.float32)
    dest = robot_to_model_xyz(np.asarray(args.destination_xyz), shift).astype(np.float32)
    in_contact = False
    table_z = policy.table_z

    for gen in range(int(args.max_replans)):
        scene = SceneState(
            instruction=policy.instruction,
            tool_label=policy.tool_label,
            tool_contact_xyz_world=contact.copy(),
            tool_current_normal=normal.copy(),
            tool_current_surface_dir=sd.copy(),
            material_xyz_world=obj.copy(),
            destination_xyz_world=dest.copy(),
            table_xyz_world=np.array([0.0, 0.0, table_z], dtype=np.float32),
        )
        wps = policy.predict_waypoints(scene)
        new_brush, new_obj, new_contact = execute_chunk(
            wps,
            object_xyz=obj,
            in_contact=in_contact,
            destination_xyz=dest,
            table_z=table_z,
            chunk=int(args.chunk),
        )
        dist = float(np.linalg.norm(new_obj[:2] - dest[:2]))
        print(
            f"[gen {gen}] obj=({obj[0]:+.3f},{obj[1]:+.3f}) -> ({new_obj[0]:+.3f},{new_obj[1]:+.3f}) "
            f"contact={new_contact} dist_to_goal={dist:.3f}"
        )
        if dist <= GOAL_REGION_RADIUS_M:
            print(f"[done] delivered at gen {gen}")
            break
        contact = new_brush[0:3].copy()
        normal = new_brush[3:6].copy()
        sd = new_brush[6:9].copy()
        obj = new_obj
        in_contact = new_contact
    else:
        print("[warn] max replans reached without goal delivery")


if __name__ == "__main__":
    main()
