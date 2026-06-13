"""Closed-loop receding-horizon sim rollout for reactive VLA GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    load_control_frame,
    matrix_from_quat_xyzw,
)
from training.action_trajectory_rl.sim_convert import waypoints_tensor_to_sim_batch


@dataclass
class ClosedLoopGenRecord:
    """One replanning generation for rollout visualization."""

    material_xyz: List[float]
    path_contacts: List[List[float]] = field(default_factory=list)
    # Per-executed-waypoint tool normal / surface_dir, so the rollout panel can
    # draw the same arrows the pretrain trajectory viz shows.
    path_normals: List[List[float]] = field(default_factory=list)
    path_surface_dirs: List[List[float]] = field(default_factory=list)


@dataclass
class ClosedLoopRolloutResult:
    tracking_frac: np.ndarray
    ball_start_xyz: np.ndarray
    ball_final_xyz: np.ndarray
    episode_lengths: np.ndarray
    material_displacement_m: np.ndarray
    frames_by_env: Optional[Dict[int, list]] = None
    generations_by_env: Optional[Dict[int, List[ClosedLoopGenRecord]]] = None
    num_replans: np.ndarray | None = None
    # Flip task: final object orientation (xyzw) and xyz per env.
    object_quat_final: np.ndarray | None = None
    object_xyz_final: np.ndarray | None = None


def _ball_in_goal_region(
    ball_xyz: np.ndarray,
    dest_xyz: np.ndarray,
    *,
    region_half: float = 0.05,
    ball_radius: float = 0.02,
) -> bool:
    dxy = np.abs(ball_xyz[:2] - dest_xyz[:2])
    hx = region_half + ball_radius
    return bool(dxy[0] <= hx and dxy[1] <= hx)


def _ball_off_table(
    ball_xyz: np.ndarray,
    dest_xyz: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    drop_m: float,
) -> bool:
    """True if the ball left the table footprint or dropped below the surface."""
    off_xy = (
        ball_xyz[0] < x_min
        or ball_xyz[0] > x_max
        or ball_xyz[1] < y_min
        or ball_xyz[1] > y_max
    )
    off_z = ball_xyz[2] < (dest_xyz[2] - drop_m)
    return bool(off_xy or off_z)


def observe_env_states(sim_runner) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env brush root xyz [N,3] and material xyz [N,3]."""
    env = sim_runner.env
    n = sim_runner.num_envs
    brush = (
        env.root_state_tensor[env.object_indices, :3].detach().cpu().numpy()
        if env.object_indices.numel() > 0
        else np.zeros((n, 3), dtype=np.float32)
    )
    if env.vla_material_indices.numel() > 0:
        mat = env.root_state_tensor[env.vla_material_indices, :3].detach().cpu().numpy()
    else:
        mat = np.zeros((n, 3), dtype=np.float32)
    return brush.astype(np.float32), mat.astype(np.float32)


def observe_tool_pose(sim_runner) -> Tuple[np.ndarray, np.ndarray]:
    """Per-env tool root pose: xyz [N,3] and quaternion (xyzw) [N,4]."""
    env = sim_runner.env
    n = sim_runner.num_envs
    if env.object_indices.numel() > 0:
        state = env.root_state_tensor[env.object_indices, :7].detach().cpu().numpy()
        xyz = state[:, :3].astype(np.float32)
        quat = state[:, 3:7].astype(np.float32)
    else:
        xyz = np.zeros((n, 3), dtype=np.float32)
        quat = np.zeros((n, 4), dtype=np.float32)
        quat[:, 3] = 1.0
    return xyz, quat


def _contact_frame_from_root(
    root_xyz: np.ndarray, root_quat_xyzw: np.ndarray, T_oc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the world contact point + (normal, surface_dir) from a tool root pose.

    The control point is rigidly attached to the tool by ``T_obj_from_contact``
    (``T_oc``): ``T_world_contact = T_world_root @ T_oc``. The contact frame's
    columns are ``[surface_dir, normal x surface_dir, normal]``, so we read the
    surface_dir off column 0 and the normal off column 2.
    """
    R = matrix_from_quat_xyzw(np.asarray(root_quat_xyzw, dtype=np.float64).reshape(4))
    T_wo = np.eye(4, dtype=np.float64)
    T_wo[:3, :3] = R
    T_wo[:3, 3] = np.asarray(root_xyz, dtype=np.float64).reshape(3)
    T_wc = T_wo @ np.asarray(T_oc, dtype=np.float64).reshape(4, 4)
    contact = T_wc[:3, 3].astype(np.float32)
    surface_dir = T_wc[:3, 0].astype(np.float32)
    normal = T_wc[:3, 2].astype(np.float32)
    return contact, normal, surface_dir


def observe_material_quat(sim_runner) -> np.ndarray:
    """Per-env material orientation quaternion (xyzw) [N,4]."""
    env = sim_runner.env
    n = sim_runner.num_envs
    if env.vla_material_indices.numel() > 0:
        q = env.root_state_tensor[env.vla_material_indices, 3:7].detach().cpu().numpy()
        return q.astype(np.float32)
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 3] = 1.0
    return out


def _object_inverted_settled(
    quat_xyzw: np.ndarray,
    xyz: np.ndarray,
    *,
    table_z: float,
    material_half_z: float,
    inverted_dot_max: float,
    settle_tol_m: float,
) -> bool:
    """True if the object's +z axis points down and it rests on the table."""
    qx, qy = float(quat_xyzw[0]), float(quat_xyzw[1])
    up_dot = 1.0 - 2.0 * (qx * qx + qy * qy)
    inverted = up_dot <= float(inverted_dot_max)
    rest_z = float(table_z) + float(material_half_z)
    settled = abs(float(xyz[2]) - rest_z) <= float(settle_tol_m)
    return bool(inverted and settled)


def _material_poured_settled(
    xyz: np.ndarray,
    dest_xyz: np.ndarray,
    *,
    table_z: float,
    material_half_z: float,
    region_half: float,
    material_radius: float,
    settle_tol_m: float,
) -> bool:
    """True if the material rests on the table inside the goal region (poured)."""
    if not _ball_in_goal_region(
        xyz, dest_xyz, region_half=region_half, ball_radius=material_radius
    ):
        return False
    rest_z = float(table_z) + float(material_half_z)
    return bool(abs(float(xyz[2]) - rest_z) <= float(settle_tol_m))


def update_scenes_from_observation(
    scenes: List[Dict[str, Any]],
    brush_xyz: np.ndarray,
    brush_quat: np.ndarray,
    material_xyz: np.ndarray,
    *,
    T_oc: np.ndarray,
    table_z: float = 0.53,
) -> List[Dict[str, Any]]:
    """Refresh tool contact/orientation and material xyz from sim observation.

    The model conditions on the tool's *contact point* (blade tip) and its current
    normal/surface_dir, not the tool root body origin. The contact frame is rigidly
    offset from the root by ~0.23 m along the handle, so feeding the raw root xyz
    back as the contact teleported the tool backward by that offset every replan
    (the "approach then jump back" oscillation). Recover the true contact frame
    from the observed root pose via ``T_oc`` instead.
    """
    out: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes):
        s = dict(sc)
        contact, normal, surface_dir = _contact_frame_from_root(
            brush_xyz[i], brush_quat[i], T_oc
        )
        s["tool_contact_xyz_world"] = contact.tolist()
        s["tool_current_normal"] = normal.tolist()
        s["tool_current_surface_dir"] = surface_dir.tolist()
        mz = np.asarray(s.get("material_xyz_world", material_xyz[i]), dtype=np.float32).reshape(3)
        mz[:3] = material_xyz[i]
        mz[2] = max(float(table_z), float(material_xyz[i, 2]))
        s["material_xyz_world"] = mz.tolist()
        out.append(s)
    return out


def run_closed_loop_rollout(
    sim_runner,
    *,
    scenes: List[Dict[str, Any]],
    wp_world: torch.Tensor,
    model_resample_fn,
    control_frame_path: Path,
    tool_obj_path: Path | None = None,
    steps_per_segment: int = 1,
    chunk_size: int = 2,
    max_replans: int = 15,
    max_steps_per_chunk: int = 300,
    max_total_steps: int = 900,
    goal_region_half_m: float = 0.05,
    ball_radius_m: float = 0.02,
    table_x_min_m: float | None = None,
    table_x_max_m: float | None = None,
    table_y_min_m: float | None = None,
    table_y_max_m: float | None = None,
    off_table_drop_m: float = 0.08,
    max_stall_replans: int = 3,
    capture: bool = False,
    capture_interval: int = 4,
    record_generations: bool = False,
    # Task selector: "sweep" (ball into goal region), "flip" (object inverted and
    # settled on the table), "pour" (material scooped, carried, and poured so it
    # rests inside the goal region on the table), or "hammer" (drive a z-locked
    # "nail" post straight down until its head reaches the target sink depth).
    task: str = "sweep",
    flip_inverted_dot_max: float = -0.5,
    flip_settle_tol_m: float = 0.04,
    flip_material_half_z_m: float = 0.006,
    pour_settle_tol_m: float = 0.03,
    pour_material_half_z_m: float = 0.009,
    # Hammer: the sim nail is a stood-up post whose CENTER is reported by the env
    # but whose TOP is the struck "nail head". ``nail_post_height_m`` lets us map
    # the observed post center back to the head height the model conditions on,
    # and ``material_z_offset`` lowers the spawned post so its top aligns with the
    # scene's head height. ``hammer_target_tol_m`` is the head-reaches-target slop.
    nail_post_height_m: float = 0.0,
    material_z_offset: float = 0.0,
    hammer_target_tol_m: float = 0.004,
) -> ClosedLoopRolloutResult:
    """Receding-horizon rollout: execute ``chunk_size`` goals, re-observe, replan.

    ``model_resample_fn(scenes) -> wp_world [B,K,9]`` is called each replan
    (including the first plan passed in ``wp_world`` for gen 0 only if replan>0).

    A per-env rollout stops early (the env is frozen at its current ball position)
    when any of these holds: the ball lands in the goal region (success), the ball
    falls off the table (failure), or the executed chunk makes no tracking progress
    for ``max_stall_replans`` consecutive generations (a waypoint stayed
    unreachable). All envs stop after ``max_replans`` generations regardless.
    """
    n = len(scenes)
    is_flip = str(task) == "flip"
    is_pour = str(task) == "pour"
    is_hammer = str(task) == "hammer"
    # Flip, pour and hammer spawn the tool in its real (upright) contact-frame
    # orientation rather than the brush flat-lay override. Flip/pour also carry
    # the material in the air, so the chunk does not early-stop merely because
    # the carried material's xy passes over the goal region; the hammer keeps the
    # nail pinned in xy so the in-region early-stop is likewise disabled.
    upright_start = is_flip or is_pour or is_hammer
    half_post = 0.5 * float(nail_post_height_m)
    current_scenes = [dict(s) for s in scenes]
    current_wp = wp_world.detach().cpu()
    # Control frame (T_obj_from_contact): used to recover the world contact frame
    # from the observed tool root pose when re-feeding the scene each replan.
    T_oc = load_control_frame(Path(control_frame_path))

    ball_start = None
    peak_succ = np.zeros(n, dtype=np.float32)
    total_lengths = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)
    num_replans = np.zeros(n, dtype=np.int32)
    # Flip: per-env final object orientation/xyz, frozen when an env stops.
    obj_quat_frozen: List[np.ndarray | None] = [None] * n
    obj_xyz_frozen: List[np.ndarray | None] = [None] * n
    generations_by_env: Dict[int, List[ClosedLoopGenRecord]] = (
        {i: [] for i in range(n)} if record_generations else {}
    )
    frames_by_env: Dict[int, list] = {}
    if capture and sim_runner.record_video:
        frames_by_env = {ei: [] for ei in sim_runner.video_env_indices}

    total_steps = 0
    gen_idx = 0
    # Per-env ball position frozen at the moment the env stops (success/failure/
    # stall). Envs that never stop use the final live observation.
    ball_frozen: List[np.ndarray | None] = [None] * n
    stall_count = np.zeros(n, dtype=np.int32)

    # Table footprint for the off-table failure check (defaults: symmetric table).
    x_min = float(table_x_min_m) if table_x_min_m is not None else -0.2375
    x_max = float(table_x_max_m) if table_x_max_m is not None else 0.2375
    y_min = float(table_y_min_m) if table_y_min_m is not None else -0.20
    y_max = float(table_y_max_m) if table_y_max_m is not None else 0.20

    while active.any() and gen_idx < int(max_replans) and total_steps < int(max_total_steps):
        if gen_idx > 0:
            current_wp = model_resample_fn(current_scenes).detach().cpu()

        goals_b, start_b, mat_b, dest_b, num_goals = waypoints_tensor_to_sim_batch(
            current_wp,
            current_scenes,
            control_frame_path=control_frame_path,
            tool_obj_path=tool_obj_path,
            steps_per_segment=int(steps_per_segment),
            upright_start=upright_start,
            keep_marker_z=is_hammer,
            material_z_offset=float(material_z_offset) if is_hammer else 0.0,
        )
        chunk_goals = min(int(chunk_size), int(num_goals))

        if record_generations:
            wp_np = current_wp.numpy()
            for i in range(n):
                if not active[i]:
                    continue
                mat = np.asarray(current_scenes[i]["material_xyz_world"], dtype=np.float64)
                contacts = [wp_np[i, j, 0:3].tolist() for j in range(chunk_goals)]
                normals = [wp_np[i, j, 3:6].tolist() for j in range(chunk_goals)]
                surface_dirs = [wp_np[i, j, 6:9].tolist() for j in range(chunk_goals)]
                generations_by_env[i].append(
                    ClosedLoopGenRecord(
                        material_xyz=mat.reshape(3).tolist(),
                        path_contacts=contacts,
                        path_normals=normals,
                        path_surface_dirs=surface_dirs,
                    )
                )

        # Hand the env the FULL plan so its internal goal counter never reaches
        # ``max_consecutive_successes`` at the chunk boundary (which would trigger
        # a full actor reset / teleport). We only roll until ``chunk_goals``
        # successes below, then re-observe and replan.
        if gen_idx == 0:
            pan_xyz = None
            if any("pan_xyz_world" in s for s in current_scenes):
                pan_xyz = torch.as_tensor(
                    np.array([s.get("pan_xyz_world", [0.0, 0.0, 0.0]) for s in current_scenes], dtype=np.float32),
                    dtype=torch.float32,
                )
            sim_runner.apply_scene_batch(goals_b, start_b, mat_b, dest_b, pan_xyz_batch=pan_xyz)
        else:
            sim_runner.update_goal_batch(goals_b)
        if ball_start is None and sim_runner.env.vla_material_indices.numel() > 0:
            ball_start = (
                sim_runner.env.root_state_tensor[sim_runner.env.vla_material_indices, :3]
                .clone()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        # Sweep stops a chunk early once the ball enters the goal region; flip and
        # pour have no in-region early stop (the carried material passes over the
        # goal in the air), so they run the chunk to track the plan.
        goal_xy_chunk = (
            None
            if (is_flip or is_pour or is_hammer)
            else np.array(
                [s["destination_xyz_world"] for s in current_scenes], dtype=np.float32
            )[:, :2]
        )
        chunk_out = sim_runner.roll_until_n_successes(
            n_successes=int(chunk_goals),
            max_steps=int(max_steps_per_chunk),
            capture=capture,
            capture_interval=int(capture_interval),
            success_denom=int(chunk_goals),
            goal_xy=goal_xy_chunk,
            goal_region_half=float(goal_region_half_m),
            material_radius=float(ball_radius_m),
        )
        total_steps += int(chunk_out.episode_lengths.max()) if chunk_out.episode_lengths.size else 0
        peak_succ = np.maximum(peak_succ, chunk_out.tracking_frac)
        total_lengths += chunk_out.episode_lengths.astype(np.int32)
        num_replans[active] += 1

        if capture and chunk_out.frames_by_env:
            for ei, frs in chunk_out.frames_by_env.items():
                frames_by_env.setdefault(ei, []).extend(frs)

        brush_xyz, mat_xyz = observe_env_states(sim_runner)
        _, brush_quat = observe_tool_pose(sim_runner)
        mat_quat = observe_material_quat(sim_runner) if is_flip else None
        dest_np = np.array(
            [s["destination_xyz_world"] for s in current_scenes], dtype=np.float64
        )
        table_z_now = float(current_scenes[0].get("table_xyz_world", [0, 0, 0.53])[2])
        # Per-chunk tracking fraction (successes reset between chunks via
        # update_goal_batch), used as the no-progress / unreachable signal.
        chunk_track = np.asarray(chunk_out.tracking_frac, dtype=np.float64).reshape(-1)
        for i in range(n):
            if not active[i]:
                continue
            if is_flip:
                # Success: object up-axis inverted AND resting on the table.
                if _object_inverted_settled(
                    mat_quat[i],
                    mat_xyz[i],
                    table_z=table_z_now,
                    material_half_z=float(flip_material_half_z_m),
                    inverted_dot_max=float(flip_inverted_dot_max),
                    settle_tol_m=float(flip_settle_tol_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    obj_quat_frozen[i] = mat_quat[i].copy()
                    obj_xyz_frozen[i] = mat_xyz[i].copy()
                    continue
                # Failure: object dropped off the table.
                if _ball_off_table(
                    mat_xyz[i],
                    np.array([mat_xyz[i, 0], mat_xyz[i, 1], table_z_now]),
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    drop_m=float(off_table_drop_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    obj_quat_frozen[i] = mat_quat[i].copy()
                    obj_xyz_frozen[i] = mat_xyz[i].copy()
                    continue
            elif is_pour:
                # Success: material poured -> rests on the table in the goal region.
                if _material_poured_settled(
                    mat_xyz[i],
                    dest_np[i],
                    table_z=table_z_now,
                    material_half_z=float(pour_material_half_z_m),
                    region_half=float(goal_region_half_m),
                    material_radius=float(ball_radius_m),
                    settle_tol_m=float(pour_settle_tol_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    continue
                # Failure: material dropped off the table.
                if _ball_off_table(
                    mat_xyz[i],
                    dest_np[i],
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    drop_m=float(off_table_drop_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    continue
            elif is_hammer:
                # Success: the nail head (post TOP = observed center + half the
                # post height) has been driven down to the target sink depth.
                head_z = float(mat_xyz[i, 2]) + half_post
                target_z = float(dest_np[i, 2])
                if head_z <= target_z + float(hammer_target_tol_m):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    continue
                # The nail is xy/orientation-pinned and cannot fall off the
                # table, so there is no off-table failure for the hammer task.
            else:
                # Success: ball landed in the goal region.
                if _ball_in_goal_region(
                    mat_xyz[i],
                    dest_np[i],
                    region_half=float(goal_region_half_m),
                    ball_radius=float(ball_radius_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    continue
                # Failure: ball fell off the table.
                if _ball_off_table(
                    mat_xyz[i],
                    dest_np[i],
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    drop_m=float(off_table_drop_m),
                ):
                    active[i] = False
                    ball_frozen[i] = mat_xyz[i].copy()
                    continue
            # Stall: this chunk reached none of its sub-goals -> the next
            # waypoint was effectively unreachable. Give up after a few in a row.
            if chunk_track[i] <= 1e-6:
                stall_count[i] += 1
            else:
                stall_count[i] = 0
            if stall_count[i] >= int(max_stall_replans):
                active[i] = False
                ball_frozen[i] = mat_xyz[i].copy()
                if is_flip and mat_quat is not None:
                    obj_quat_frozen[i] = mat_quat[i].copy()
                    obj_xyz_frozen[i] = mat_xyz[i].copy()

        if not active.any():
            break

        table_z = float(current_scenes[0].get("table_xyz_world", [0, 0, 0.53])[2])
        # For the hammer, the model conditions on the nail HEAD (post top), so
        # report the observed post center lifted by half the post height.
        mat_xyz_report = mat_xyz
        if is_hammer and half_post > 0.0:
            mat_xyz_report = mat_xyz.copy()
            mat_xyz_report[:, 2] = mat_xyz_report[:, 2] + half_post
        current_scenes = update_scenes_from_observation(
            current_scenes, brush_xyz, brush_quat, mat_xyz_report, T_oc=T_oc, table_z=table_z
        )
        gen_idx += 1

    brush_xyz, mat_xyz = observe_env_states(sim_runner)
    if ball_start is None:
        ball_start = mat_xyz.copy()
    ball_final = mat_xyz.copy()
    # Envs that stopped early (success/failure/stall) keep the ball position from
    # the moment they stopped, so later stepping of still-active envs can't drag a
    # finished env's ball out of the goal (or back onto the table).
    for i in range(n):
        if ball_frozen[i] is not None:
            ball_final[i] = ball_frozen[i]
    disp = np.linalg.norm(ball_final - ball_start, axis=1).astype(np.float32)

    object_quat_final = None
    object_xyz_final = None
    if is_flip:
        mat_quat = observe_material_quat(sim_runner)
        object_quat_final = mat_quat.copy()
        object_xyz_final = mat_xyz.copy()
        for i in range(n):
            if obj_quat_frozen[i] is not None:
                object_quat_final[i] = obj_quat_frozen[i]
            if obj_xyz_frozen[i] is not None:
                object_xyz_final[i] = obj_xyz_frozen[i]
        object_quat_final = object_quat_final.astype(np.float32)
        object_xyz_final = object_xyz_final.astype(np.float32)

    gens_out = generations_by_env if record_generations else None
    frames_out = frames_by_env if capture and frames_by_env else None
    return ClosedLoopRolloutResult(
        tracking_frac=peak_succ.astype(np.float32),
        ball_start_xyz=ball_start.astype(np.float32),
        ball_final_xyz=ball_final.astype(np.float32),
        episode_lengths=total_lengths.astype(np.int32),
        material_displacement_m=disp,
        frames_by_env=frames_out,
        generations_by_env=gens_out,
        num_replans=num_replans.astype(np.int32),
        object_quat_final=object_quat_final,
        object_xyz_final=object_xyz_final,
    )
