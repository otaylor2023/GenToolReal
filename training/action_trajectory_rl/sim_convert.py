"""Convert VLA waypoint predictions to per-env IsaacGym rollout tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from generative_str_pipeline.sim_rollout.build_brush_trajectory import (
    build_trajectory,
    load_control_frame,
)
from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    flat_rest_object_pose,
)
from generative_str_pipeline.sim_workspace import (
    WIDE_TABLE_X_MAX_M,
    WIDE_TABLE_X_MIN_M,
    WIDE_TABLE_Y_MAX_M,
    WIDE_TABLE_Y_MIN_M,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_START = REPO_ROOT / (
    "policy_exec/dextoolbench/trajectories/brush/blue_brush/sweep_forward.json"
)
# Brush collision mesh (object-local frame); used to rest the tool flat on the
# table at spawn without clipping.
BRUSH_OBJ = REPO_ROOT / (
    "policy_exec/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj"
)


def scene_to_model_datapoint(scene: Dict[str, Any]) -> Dict[str, Any]:
    return scene


def waypoints_tensor_to_sim_batch(
    waypoints_world: torch.Tensor,
    scenes: List[Dict[str, Any]],
    *,
    control_frame_path: Path,
    tool_obj_path: Optional[Path] = None,
    canonical_start_pose: Optional[np.ndarray] = None,
    steps_per_segment: int = 1,
    # When True, the whole scene (brush trajectory + ball + goal) is rigidly
    # shifted so the anchor waypoint lands at a fixed canonical brush grasp.
    # Default False: the ball and goal stay at their true input world positions
    # and the brush simply follows the model's predicted world-space trajectory.
    align_to_canonical: bool = False,
    anchor_waypoint: int = 2,
    marker_surface_z: float = 0.53,
    # Flip: spawn the tool in its real contact-frame orientation (blade face up,
    # right-side up) instead of the brush flat-lay override.
    upright_start: bool = False,
    # Hammer: the "nail" starts protruding ABOVE the table, so keep the scene's
    # own material/destination z instead of flattening it to the table surface.
    # ``material_z_offset`` is added to the kept material z (e.g. to drop a
    # stood-up post so its TOP aligns with the dataset nail-head height).
    keep_marker_z: bool = False,
    material_z_offset: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Convert [N, K, 9] predicted waypoints to sim batch tensors.

    Returns:
        goals_batch [N, T, 7], start_batch [N, 7], material_xyz [N, 3], dest_xyz [N, 3], T
    """
    if align_to_canonical and canonical_start_pose is None:
        import json

        ref = json.loads(CANONICAL_START.read_text(encoding="utf-8"))
        canonical_start_pose = np.asarray(ref["start_pose"], dtype=np.float64).reshape(7)
    if not align_to_canonical:
        canonical_start_pose = None

    cf = Path(control_frame_path)
    if not cf.is_absolute():
        cf = REPO_ROOT / cf
    T_oc = load_control_frame(cf)

    tool_obj = Path(tool_obj_path) if tool_obj_path is not None else BRUSH_OBJ
    if not tool_obj.is_absolute():
        tool_obj = REPO_ROOT / tool_obj

    _wp_arr = waypoints_world.detach().cpu().numpy()
    wp = _wp_arr.reshape(_wp_arr.shape[0], -1, 9)
    n = wp.shape[0]
    goals_list = []
    starts = []
    materials = []
    destinations = []
    num_goals = None

    for i in range(n):
        sc = scenes[i]
        table_z = float(sc.get("table_xyz_world", [0, 0, marker_surface_z])[2])
        tool_contact = np.asarray(sc.get("tool_contact_xyz_world", wp[i, 0, 0:3]), dtype=np.float64)
        tool_heading = np.asarray(sc.get("tool_current_surface_dir", wp[i, 0, 6:9]), dtype=np.float64)
        tool_normal = np.asarray(sc.get("tool_current_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
        xyz, quat = flat_rest_object_pose(
            tool_contact,
            tool_heading,
            T_oc,
            table_z=table_z,
            mesh_path=str(tool_obj),
            contact_normal=tool_normal if upright_start else None,
            contact_surface_dir=tool_heading if upright_start else None,
            table_x_bounds=(WIDE_TABLE_X_MIN_M, WIDE_TABLE_X_MAX_M),
            table_y_bounds=(WIDE_TABLE_Y_MIN_M, WIDE_TABLE_Y_MAX_M),
        )
        start_pose = np.concatenate([xyz, quat])
        mat_in = np.asarray(sc["material_xyz_world"], dtype=np.float64)
        dest_in = np.asarray(sc["destination_xyz_world"], dtype=np.float64)
        traj = build_trajectory(
            waypoints=wp[i],
            T_oc=T_oc,
            start_pose=start_pose,
            steps_per_segment=int(steps_per_segment),
            canonical_start_pose=canonical_start_pose,
            align_mode="translation",
            material_xyz=mat_in,
            destination_xyz=dest_in,
            marker_surface_z=float(marker_surface_z),
            anchor_waypoint=int(anchor_waypoint),
        )
        goals = np.asarray(traj["goals"], dtype=np.float64)
        if num_goals is None:
            num_goals = int(goals.shape[0])
        elif int(goals.shape[0]) != num_goals:
            raise ValueError(
                f"Env {i} has {goals.shape[0]} goals, expected {num_goals}; "
                "use fixed steps_per_segment for RL batches."
            )
        goals_list.append(goals)
        starts.append(traj["start_pose"])
        if align_to_canonical:
            markers = traj.get("markers", {})
            materials.append(markers.get("material_xyz", mat_in))
            destinations.append(markers.get("destination_xyz", dest_in))
        elif keep_marker_z:
            # Hammer/nail: keep the protruding head/target heights (offset the
            # material so a stood-up post's TOP sits at the input head height).
            materials.append(
                [mat_in[0], mat_in[1], float(mat_in[2]) + float(material_z_offset)]
            )
            destinations.append([dest_in[0], dest_in[1], float(dest_in[2])])
        else:
            # No relocation: ball and goal at input x,y; z on this scene's table surface.
            materials.append([mat_in[0], mat_in[1], table_z])
            destinations.append([dest_in[0], dest_in[1], table_z])

    goals_batch = torch.tensor(np.stack(goals_list, axis=0), dtype=torch.float32)
    start_batch = torch.tensor(np.stack(starts, axis=0), dtype=torch.float32)
    material_batch = torch.tensor(np.stack(materials, axis=0), dtype=torch.float32)
    dest_batch = torch.tensor(np.stack(destinations, axis=0), dtype=torch.float32)
    return goals_batch, start_batch, material_batch, dest_batch, int(num_goals)
