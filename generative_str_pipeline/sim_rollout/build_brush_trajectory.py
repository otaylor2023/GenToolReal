"""Build a dextoolbench brush-trajectory JSON from VLA waypoints.

Two input sources are supported:
  * a procedural shard datapoint (ground-truth waypoints) for pipeline bring-up
  * a predicted-waypoints JSON ({"waypoints": [[...9...] x 6], "tool_*": ...})

Output schema matches dextoolbench eval (``start_pose`` + ``goals``), where each
pose is ``[x, y, z, qx, qy, qz, qw]`` for the brush object in world space.

Densification: consecutive object poses are interpolated (lerp position, slerp
orientation) so the low-level policy sees small goal-to-goal steps like the
reference trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    flat_rest_object_pose,
    load_control_frame,
    matrix_from_quat_xyzw,
    quat_xyzw_from_matrix,
    waypoint_to_object_pose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BRUSH_OBJ = REPO_ROOT / "policy_exec/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj"


def _pose7_to_T(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat_xyzw(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def _T_to_pose7(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], quat_xyzw_from_matrix(T[:3, :3])])


def _align_transform(vla_start: np.ndarray, canonical_start: np.ndarray) -> np.ndarray:
    """Rigid SE(3) that maps the VLA start object pose onto a canonical sim grasp.

    Applying this to every pose preserves the relative sweep motion while
    guaranteeing the start pose is a valid in-hand grasp inside the workspace.
    """
    return _pose7_to_T(canonical_start) @ np.linalg.inv(_pose7_to_T(vla_start))


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * u
    sin0 = np.sin(theta0)
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    q = s0 * q0 + s1 * q1
    return q / max(np.linalg.norm(q), 1e-12)


def _densify(poses: np.ndarray, steps_per_segment: int) -> np.ndarray:
    """poses [N,7] -> densified [M,7] with lerp position + slerp orientation."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    if steps_per_segment <= 1 or poses.shape[0] < 2:
        return poses
    out: List[np.ndarray] = [poses[0]]
    for i in range(poses.shape[0] - 1):
        p0, p1 = poses[i], poses[i + 1]
        for k in range(1, steps_per_segment + 1):
            u = k / float(steps_per_segment)
            pos = (1.0 - u) * p0[:3] + u * p1[:3]
            quat = _slerp(p0[3:7], p1[3:7], u)
            out.append(np.concatenate([pos, quat]))
    return np.asarray(out, dtype=np.float64)


def _waypoints_from_shard(shard_path: Path, datapoint_index: int) -> Dict[str, Any]:
    shard = json.loads(Path(shard_path).read_text(encoding="utf-8"))
    dps = shard.get("datapoints", [])
    match = None
    for dp in dps:
        if int(dp.get("datapoint_index", -1)) == int(datapoint_index):
            match = dp
            break
    if match is None:
        raise ValueError(f"datapoint_index {datapoint_index} not found in {shard_path}")
    return match


def _start_pose_from_tool(dp: Dict[str, Any], T_oc: np.ndarray) -> np.ndarray:
    contact = np.asarray(dp["tool_contact_xyz_world"], dtype=np.float64)
    surface_dir = np.asarray(dp["tool_current_surface_dir"], dtype=np.float64)
    table_xyz = dp.get("table_xyz_world", [0.0, 0.0, contact[2]])
    table_z = float(table_xyz[2])
    xyz, quat = flat_rest_object_pose(
        contact,
        surface_dir,
        T_oc,
        table_z=table_z,
        mesh_path=str(BRUSH_OBJ),
    )
    return np.concatenate([xyz, quat])


def build_trajectory(
    *,
    waypoints: np.ndarray,
    T_oc: np.ndarray,
    start_pose: np.ndarray,
    steps_per_segment: int,
    canonical_start_pose: np.ndarray | None = None,
    align_mode: str = "se3",
    material_xyz: np.ndarray | None = None,
    destination_xyz: np.ndarray | None = None,
    marker_surface_z: float = 0.53,
    anchor_waypoint: int = 2,
) -> Dict[str, Any]:
    waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 9)
    n_wp = int(waypoints.shape[0])
    goal_poses = np.zeros((n_wp, 7), dtype=np.float64)
    for i in range(n_wp):
        xyz, quat = waypoint_to_object_pose(
            waypoints[i, 0:3], waypoints[i, 3:6], waypoints[i, 6:9], T_oc
        )
        goal_poses[i, :3] = xyz
        goal_poses[i, 3:7] = quat

    # Scene markers (material + destination region) follow the SAME alignment as
    # the brush goals so they stay spatially consistent with the swept path.
    def _align_point(p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=np.float64).reshape(3).copy()

    aligned = False
    align_shift = np.zeros(3)
    T_align = np.eye(4)
    if canonical_start_pose is not None:
        canonical_start_pose = np.asarray(canonical_start_pose, dtype=np.float64).reshape(7)
        if align_mode == "translation":
            # Both frames are z-up with the table at ~0.53; only shift x,y so the
            # anchor waypoint lands at the canonical start x,y, and seed start_pose
            # from the canonical (valid in-hand) grasp. anchor_waypoint=2 is the
            # touchdown contact; anchor_waypoint=0 is the elevated approach (brush
            # then starts further from the material).
            a = int(np.clip(anchor_waypoint, 0, n_wp - 1))
            align_shift[:2] = canonical_start_pose[:2] - goal_poses[a, :2]
            goal_poses[:, :3] += align_shift
            start_pose = canonical_start_pose

            def _align_point(p: np.ndarray) -> np.ndarray:  # noqa: F811
                q = np.asarray(p, dtype=np.float64).reshape(3) + align_shift
                q[2] = marker_surface_z
                return q
        else:
            T_align = _align_transform(start_pose, canonical_start_pose)
            start_pose = canonical_start_pose
            goal_poses = np.stack(
                [_T_to_pose7(T_align @ _pose7_to_T(goal_poses[i])) for i in range(n_wp)],
                axis=0,
            )

            def _align_point(p: np.ndarray) -> np.ndarray:  # noqa: F811
                hp = np.ones(4)
                hp[:3] = np.asarray(p, dtype=np.float64).reshape(3)
                q = (T_align @ hp)[:3]
                q[2] = marker_surface_z
                return q

        aligned = True

    markers: Dict[str, Any] = {}
    if material_xyz is not None:
        markers["material_xyz"] = _align_point(material_xyz).tolist()
    if destination_xyz is not None:
        markers["destination_xyz"] = _align_point(destination_xyz).tolist()
    if markers:
        markers["surface_z"] = float(marker_surface_z)

    # Prepend start pose so the first segment is also densified.
    full = np.concatenate([start_pose.reshape(1, 7), goal_poses], axis=0)
    dense = _densify(full, steps_per_segment)
    # goals exclude the start pose itself
    goals = dense[1:]
    out: Dict[str, Any] = {
        "start_pose": start_pose.tolist(),
        "goals": goals.tolist(),
        "_meta": {
            "source": "vla_waypoints",
            "num_waypoints": n_wp,
            "steps_per_segment": int(steps_per_segment),
            "num_goals": int(goals.shape[0]),
            "aligned_to_canonical_start": aligned,
        },
    }
    if markers:
        out["markers"] = markers
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dextoolbench brush-trajectory JSON from VLA waypoints."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard_path", type=str, help="Procedural shard JSON (GT waypoints)")
    src.add_argument(
        "--waypoints_json",
        type=str,
        help='Predicted waypoints JSON: {"waypoints": [[..9..] x6], "tool_*": [..]}',
    )
    parser.add_argument("--datapoint_index", type=int, default=0)
    parser.add_argument(
        "--control_frame",
        type=str,
        default="generative_str_pipeline/assets/object_control_points/blue_brush.json",
    )
    parser.add_argument("--steps_per_segment", type=int, default=6)
    parser.add_argument(
        "--align_start_to",
        type=str,
        default="",
        help="Reference trajectory JSON; align VLA trajectory into the sim workspace "
        "using its start_pose as the canonical anchor.",
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="se3",
        choices=["se3", "translation"],
        help="se3: full rigid map of VLA start->canonical; translation: only x,y shift "
        "so first contact lands at canonical x,y (keeps z-up orientations).",
    )
    parser.add_argument(
        "--marker_surface_z",
        type=float,
        default=0.53,
        help="Sim table surface z; material/destination markers are placed here.",
    )
    parser.add_argument(
        "--no_markers",
        action="store_true",
        help="Do not emit material/destination markers in the trajectory JSON.",
    )
    parser.add_argument(
        "--anchor_waypoint",
        type=int,
        default=2,
        help="Which waypoint maps to the canonical start (translation mode). "
        "2 = touchdown contact (brush starts on material); 0 = elevated approach "
        "(brush starts further from the material).",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    control_frame_path = Path(args.control_frame)
    if not control_frame_path.is_absolute():
        control_frame_path = REPO_ROOT / control_frame_path
    T_oc = load_control_frame(control_frame_path)

    if args.shard_path:
        shard_path = Path(args.shard_path)
        if not shard_path.is_absolute():
            shard_path = REPO_ROOT / shard_path
        dp = _waypoints_from_shard(shard_path, args.datapoint_index)
        waypoints = np.asarray(dp["waypoints"], dtype=np.float64).reshape(6, 9)
        start_pose = _start_pose_from_tool(dp, T_oc)
        source = dp
    else:
        wp_path = Path(args.waypoints_json)
        if not wp_path.is_absolute():
            wp_path = REPO_ROOT / wp_path
        data = json.loads(wp_path.read_text(encoding="utf-8"))
        waypoints = np.asarray(data["waypoints"], dtype=np.float64).reshape(6, 9)
        if "tool_contact_xyz_world" in data:
            start_pose = _start_pose_from_tool(data, T_oc)
        else:
            # fall back to first waypoint pose as start
            xyz, quat = waypoint_to_object_pose(
                waypoints[0, 0:3], waypoints[0, 3:6], waypoints[0, 6:9], T_oc
            )
            start_pose = np.concatenate([xyz, quat])
        source = data

    material_xyz = None
    destination_xyz = None
    if not args.no_markers:
        mat = source.get("material_xyz_world")
        dest = source.get("destination_xyz_world")
        if mat is not None:
            material_xyz = np.asarray(mat, dtype=np.float64)
        if dest is not None:
            destination_xyz = np.asarray(dest, dtype=np.float64)

    canonical_start_pose = None
    if args.align_start_to.strip():
        ref_path = Path(args.align_start_to)
        if not ref_path.is_absolute():
            ref_path = REPO_ROOT / ref_path
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        canonical_start_pose = np.asarray(ref["start_pose"], dtype=np.float64).reshape(7)

    traj = build_trajectory(
        waypoints=waypoints,
        T_oc=T_oc,
        start_pose=start_pose,
        steps_per_segment=int(args.steps_per_segment),
        canonical_start_pose=canonical_start_pose,
        align_mode=str(args.align_mode),
        material_xyz=material_xyz,
        destination_xyz=destination_xyz,
        marker_surface_z=float(args.marker_surface_z),
        anchor_waypoint=int(args.anchor_waypoint),
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traj, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (start + {traj['_meta']['num_goals']} goals)")


if __name__ == "__main__":
    main()
