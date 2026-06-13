"""Procedurally generate reactive closed-loop spoon scoop-and-pour shards.

The spoon_spatula comes in from the front, scoops a small material portion,
lifts it over a goal, then rolls sideways about the handle axis to pour the
material out over the goal. This intentionally mirrors the DexToolBench
``serve_plate`` motion shape, but stays in the same compact waypoint schema used
by the action-trajectory VLA datasets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import (
    TABLE_LABEL,
    _datapoint_rng,
    _safe_format,
)
from generative_str_pipeline.build_dataset_0010_brush_sweep_sim import SweepGenConfig
from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    contact_frame_world,
    matrix_from_quat_xyzw,
    quat_xyzw_from_matrix,
)
from generative_str_pipeline.sim_workspace import (
    WORKSPACE_X_EXTENT_M,
    WORKSPACE_Y_EXTENT_M,
    clip_xy_rect,
)

DATASET_ID = "dataset_0013_spoon_pour_reactive"
TOOL_LABEL = "the spoon"

MATERIALS_POUR = (
    "the rice",
    "the beans",
    "the cereal",
    "the sugar",
    "the lentils",
    "the oats",
    "the granola",
    "the scoopful",
    "the small portion",
    "the food",
)
DESTINATIONS_POUR = (
    "the plate",
    "the goal plate",
    "the serving spot",
    "the bowl",
    "the dish",
    "the target bowl",
    "the second plate",
)
TEMPLATES_POUR = (
    "Scoop {material} with {tool}, carry it over {destination}, and pour it out to the side.",
    "Use {tool} to scoop {material} from the front, lift it above {destination}, then pour sideways.",
    "Serve {material} with {tool} by scooping it up and pouring it over {destination}.",
    "Scoop up {material} using {tool}, move it above {destination}, and tip it out to the side.",
    "With {tool}, dig into {material}, lift it over {destination}, and pour it off to one side.",
    "Pick up {material} with {tool} and empty it sideways onto {destination}.",
    "Use {tool} to scoop {material}, carry it across to {destination}, and tilt it to pour.",
    "Scoop {material} from the front with {tool} and pour it onto {destination}.",
    "Lift {material} with {tool} and tip it out to the side over {destination}.",
    "Gather {material} in {tool}, raise it above {destination}, and pour it out sideways.",
    "Scoop {material} up with {tool} and dump it to the side onto {destination}.",
    "Using {tool}, scoop {material}, carry it to {destination}, and roll your wrist to pour it out.",
    "Collect {material} with {tool} and pour it sideways into {destination}.",
    "Scoop {material} with {tool}, hold it over {destination}, and pour it off to the side.",
)

# A compact proxy for a spoonful of material.
MATERIAL_SIZE_M = np.array([0.035, 0.035, 0.018], dtype=np.float32)
MATERIAL_HALF_M = 0.5 * MATERIAL_SIZE_M

# Handleless pan geometry (matches the spatula-flip pan). The pour pan is
# cosmetic for the dataset (the spoon scoops the material out and pours it onto a
# plate), but it is placed material-relative so viz/sim draw it AROUND the
# sampled material instead of at a fixed world point.
PAN_RADIUS_M = 4.5 * 0.0254
PAN_WALL_HEIGHT_M = 2.0 * 0.0254
_POUR_HALF_XY_DIAG = float(np.linalg.norm(MATERIAL_HALF_M[:2]))
# Distance from the pan center to the object so it sits near the rim. The object
# is parked on the far (-x) side of the pan from the +x spoon entry, so the pan
# center is this far in +x from the object.
PAN_RIM_OFFSET_M = max(0.01, PAN_RADIUS_M - _POUR_HALF_XY_DIAG - 0.008)

POUR_GOAL_THETA_RAD = 1.35
POUR_THETA_TOL_RAD = 0.20


def _pour_pan_center_xy(material_xyz: np.ndarray) -> np.ndarray:
    """Material-relative pan center: object near the -x rim, pan extends +x."""
    mat = np.asarray(material_xyz, dtype=np.float64).reshape(3)
    return np.array([float(mat[0]) + PAN_RIM_OFFSET_M, float(mat[1])], dtype=np.float64)


@dataclass
class SpoonPourGenConfig(SweepGenConfig):
    """Reactive closed-loop spoon scoop-and-pour generator."""

    dense_step_spacing_m: float = 0.035
    num_output_waypoints: int = 15
    max_generations: int = 30
    executed_chunk: int = 5
    scenes_per_shard: int = 150
    min_material_goal_sep_m: float = 0.16
    min_tool_material_sep_m: float = 0.16
    behind_offset_m_range: tuple[float, float] = (0.07, 0.11)
    scoop_slide_m: float = 0.14
    scoop_contact_z_offset_m: float = 0.004
    grab_seat_m: float = 0.02
    lift_z_above_table_m: float = 0.085
    carry_z_above_table_m: float = 0.145
    pour_release_z_above_table_m: float = 0.11
    pour_lateral_offset_m: float = 0.018
    # --- Scoop (dig-and-rotate-up) parameters ---
    # Dramatic lip-down pitch (about the spoon's lateral axis) as it approaches
    # and digs in; ~60 degrees so it clearly looks like scooping food, not a
    # flat spatula slide. This rotates back to ~0 as the bowl cups the material.
    scoop_pitch_max_rad: float = 1.05
    # Pitch held at the mid-scoop keyframe (fraction of the max) to shape the arc.
    scoop_mid_pitch_frac: float = 0.5
    # How high the lip rises (above the dig height) by the end of the scoop arc,
    # so the contact path curves up into a cupping motion instead of sliding flat.
    scoop_rise_m: float = 0.03
    # The bowl latches the material only once the spoon has cupped back toward
    # level (pitch below this) after digging in.
    scoop_pitch_latch_rad: float = 0.35
    # Where the scooped material is seated, expressed in the spoon's CONTACT
    # frame (origin = front lip, +x toward the handle/bowl center, +z = bowl
    # opening/up). These place the material down inside the cup so it rides on
    # top of the bowl surface instead of hanging at the lip plane.
    bowl_seat_x_m: float = 0.05
    bowl_seat_z_m: float = -0.012
    # Reactive failure + retry. With ``scoop_fail_prob`` per scoop attempt the
    # pickup fails and the material ends up back on the ground, so the planner
    # naturally re-approaches and tries again (no special retry logic). Two
    # failure modes (orientation does not matter for a poured blob): (1) the
    # material slips off mid-lift, and (2) it drops too early partway through the
    # carry before the pour. The final allowed attempt is forced to succeed so
    # scenes still terminate with a completed pour.
    scoop_fail_prob: float = 0.25
    max_scoop_attempts: int = 5
    slip_trigger_height_m: float = 0.05
    slip_jitter_m: float = 0.02
    # Early-drop fires once the material has been carried up this high while
    # still at least this far (xy) from the goal, so it lands clearly mid-path.
    early_drop_height_m: float = 0.07
    early_drop_min_goal_dist_m: float = 0.08


def _unit_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3).copy()
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    x, y, z = a
    c, s = float(np.cos(theta)), float(np.sin(theta))
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _spoon_frame(
    motion_dir: np.ndarray,
    theta: float,
    pour_side: float,
    scoop_pitch: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (normal, surface_dir, R) composing a scoop pitch and a pour roll.

    Two independent rotations are layered on the rest frame (normal up,
    ``surface_dir`` = ``-motion_dir`` pointing from the front lip back toward the
    handle):

    * ``scoop_pitch`` (>= 0) tilts the front lip DOWN about the spoon's lateral
      (left-right) axis while keeping the handle UP -- this is the dig attitude
      used during approach/scoop. It rotates the normal forward toward the
      material (``+motion_dir``) and lifts the handle off the table.
    * ``theta`` rolls the spoon sideways about the handle axis (``surface_dir``)
      for the serve-style side pour.

    The two are applied on disjoint phases (scoop pitch during approach/scoop,
    pour roll during the pour), so they never fight; the recovery helpers below
    decode each one independently from the resulting normal.
    """
    a = _unit_xy(motion_dir).astype(np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    surface = -a
    lateral = np.array([-a[1], a[0], 0.0], dtype=np.float64)
    # Lip-down dig with the HANDLE UP: positive rotation about the lateral axis
    # tips the normal forward toward the material (+a) and lifts the lip->handle
    # direction (surface_dir) up off the table. The opposite sign would drive the
    # long handle straight down into the table.
    R_pitch = _rot_axis_angle(lateral, float(scoop_pitch))
    R_roll = _rot_axis_angle(surface, float(pour_side) * float(theta))
    R = R_roll @ R_pitch
    normal = (R @ up).astype(np.float32)
    surface_dir = (R @ surface).astype(np.float32)
    return normal, surface_dir, R.astype(np.float64)


def _theta_from_waypoint(normal: np.ndarray, motion_dir: np.ndarray, pour_side: float) -> float:
    """Recover the sideways pour roll angle, ignoring any scoop pitch.

    The pour roll moves the normal within the (up, lateral) plane, while the
    scoop pitch moves it within the (up, motion_dir) plane. Projecting onto the
    lateral component isolates the roll, so a pitched-but-unrolled scoop frame
    correctly reports ``theta == 0``.
    """
    a = _unit_xy(motion_dir).astype(np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = np.array([-a[1], a[0], 0.0], dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    psi = float(np.arctan2(float(np.dot(n, lateral)), float(np.dot(n, up))))
    return float(pour_side) * psi


def _pitch_from_waypoint(normal: np.ndarray, motion_dir: np.ndarray) -> float:
    """Recover the scoop pitch (lip-down is positive), ignoring any pour roll.

    The scoop pitch moves the normal within the (up, motion_dir) plane toward
    the material (+motion_dir); the pour roll leaves the motion_dir component at
    zero, so this isolates the dig attitude.
    """
    a = _unit_xy(motion_dir).astype(np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    return float(np.arctan2(float(np.dot(n, a)), float(np.dot(n, up))))


def _rows_to_waypoints(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    wp = np.zeros((len(rows), 9), dtype=np.float32)
    for i, (c, normal, surface_dir) in enumerate(rows):
        wp[i, 0:3] = c
        wp[i, 3:6] = normal
        wp[i, 6:9] = surface_dir
    return wp


def _pad_target_waypoints(target: np.ndarray, num_steps: int) -> np.ndarray:
    t = np.asarray(target, dtype=np.float32).reshape(-1, 9)
    if t.shape[0] >= num_steps:
        return t[:num_steps].copy()
    out = np.zeros((num_steps, 9), dtype=np.float32)
    out[: t.shape[0]] = t
    for i in range(t.shape[0], num_steps):
        out[i] = t[-1]
    return out


def _densify_with_frames(
    keyframes: list[tuple[np.ndarray, float, float]],
    *,
    step_spacing_m: float,
    motion_dir: np.ndarray,
    pour_side: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Densify ``(contact, pour_theta, scoop_pitch)`` keyframes into frames."""
    dense: list[tuple[np.ndarray, float, float]] = []
    for i in range(len(keyframes) - 1):
        pa, ta, pita = keyframes[i]
        pb, tb, pitb = keyframes[i + 1]
        pa = np.asarray(pa, dtype=np.float64).reshape(3)
        pb = np.asarray(pb, dtype=np.float64).reshape(3)
        seg_len = float(np.linalg.norm(pb - pa))
        n_steps = max(1, int(np.round(seg_len / max(float(step_spacing_m), 1e-4))))
        for j in range(n_steps):
            t = float(j) / float(n_steps)
            c = ((1.0 - t) * pa + t * pb).astype(np.float32)
            theta = (1.0 - t) * float(ta) + t * float(tb)
            pitch = (1.0 - t) * float(pita) + t * float(pitb)
            dense.append((c, theta, pitch))
    last = keyframes[-1]
    dense.append((np.asarray(last[0], dtype=np.float32), float(last[1]), float(last[2])))

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for contact, theta, pitch in dense:
        normal, surface_dir, _ = _spoon_frame(motion_dir, theta, pour_side, scoop_pitch=pitch)
        rows.append((contact, normal, surface_dir))
    return rows


def _sample_material_and_goal(
    rng: np.random.Generator,
    cfg: SpoonPourGenConfig,
) -> tuple[np.ndarray, np.ndarray]:
    table_z = float(cfg.table_xyz_world[2])
    # Mimic bowl-to-plate serving: material starts on the left/front-ish side,
    # goal/plate is offset to the right.
    for _ in range(100):
        mat_xy = np.array(
            [
                float(rng.uniform(-0.18, -0.05)),
                float(rng.uniform(-0.11, 0.10)),
            ],
            dtype=np.float32,
        )
        goal_xy = np.array(
            [
                float(rng.uniform(0.04, 0.18)),
                float(rng.uniform(-0.09, 0.12)),
            ],
            dtype=np.float32,
        )
        if float(np.linalg.norm(goal_xy - mat_xy)) >= float(cfg.min_material_goal_sep_m):
            break
    mat = np.array([float(mat_xy[0]), float(mat_xy[1]), table_z + float(MATERIAL_HALF_M[2])], dtype=np.float32)
    goal = np.array([float(goal_xy[0]), float(goal_xy[1]), table_z + float(MATERIAL_HALF_M[2])], dtype=np.float32)
    return mat, goal


def _sample_tool_pose_pour(
    rng: np.random.Generator,
    cfg: SpoonPourGenConfig,
    *,
    material_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table_z = float(cfg.table_xyz_world[2])
    mat = np.asarray(material_xyz, dtype=np.float32).reshape(3)
    # Come in from the front: mostly +x with small y jitter, pointing toward the
    # material. The spoon is staged high enough to clear the material before it
    # descends and scoops.
    start_xy = np.array(
        [
            float(mat[0]) + float(rng.uniform(0.16, 0.22)),
            float(mat[1]) + float(rng.uniform(-0.04, 0.04)),
        ],
        dtype=np.float32,
    )
    start_xy = clip_xy_rect(start_xy, x_extent=WORKSPACE_X_EXTENT_M, y_extent=WORKSPACE_Y_EXTENT_M)
    contact = np.array([float(start_xy[0]), float(start_xy[1]), table_z + 0.09], dtype=np.float32)
    motion = _unit_xy(mat - contact)
    # Stage the spoon already pitched lip-down so it visibly comes in at the
    # dramatic scoop attitude rather than flat like a spatula.
    normal, surface_dir, _ = _spoon_frame(motion, 0.0, 1.0, scoop_pitch=float(cfg.scoop_pitch_max_rad))
    return contact, normal, surface_dir, motion


def plan_next_pour_steps(
    rng: np.random.Generator,
    cfg: SpoonPourGenConfig,
    *,
    spoon_contact: np.ndarray,
    material_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    carried: bool,
    pour_theta: float,
    motion_dir: np.ndarray,
    pour_side: float,
    num_steps: int = 15,
) -> np.ndarray:
    table_z = float(cfg.table_xyz_world[2])
    contact_h = float(cfg.scoop_contact_z_offset_m)
    spoon = np.asarray(spoon_contact, dtype=np.float32).reshape(3)
    mat = np.asarray(material_xyz, dtype=np.float32).reshape(3)
    dest = np.asarray(destination_xyz, dtype=np.float32).reshape(3)
    a = _unit_xy(motion_dir)
    lateral = np.array([-a[1], a[0]], dtype=np.float32) * float(pour_side)

    table_contact_z = table_z + contact_h
    lift_z = table_z + float(cfg.lift_z_above_table_m)
    carry_z = table_z + float(cfg.carry_z_above_table_m)
    scoop_slide = float(cfg.scoop_slide_m)
    scoop_rise = float(cfg.scoop_rise_m)
    pitch_max = float(cfg.scoop_pitch_max_rad)
    mid_frac = float(cfg.scoop_mid_pitch_frac)

    # Start the dig right at the object's near edge so the lip pushes onto the
    # object and scoops it up, rather than starting on bare table before it.
    half_x = float(MATERIAL_HALF_M[0])
    front_xy = mat[:2] - a[:2] * half_x
    front_xy = clip_xy_rect(front_xy, x_extent=WORKSPACE_X_EXTENT_M, y_extent=WORKSPACE_Y_EXTENT_M)
    # Dig point: front edge of the material, lip pressed down near the table.
    front_low = np.array([float(front_xy[0]), float(front_xy[1]), table_contact_z], dtype=np.float32)
    # Mid-scoop: half-way through the slide, lip rising and pitch easing off.
    scoop_mid = (front_low + a * (scoop_slide * 0.5)).astype(np.float32)
    scoop_mid[2] = table_contact_z + scoop_rise * float(0.5 ** 1.6)
    scoop_mid[:2] = clip_xy_rect(scoop_mid[:2], x_extent=WORKSPACE_X_EXTENT_M, y_extent=WORKSPACE_Y_EXTENT_M)
    # Seated: bowl level and cupping the material, lifted off the table a touch.
    scoop_low = (front_low + a * scoop_slide).astype(np.float32)
    scoop_low[2] = table_contact_z + scoop_rise
    scoop_low[:2] = clip_xy_rect(scoop_low[:2], x_extent=WORKSPACE_X_EXTENT_M, y_extent=WORKSPACE_Y_EXTENT_M)
    lift = scoop_low.copy()
    lift[2] = lift_z
    above_goal = np.array([float(dest[0]), float(dest[1]), carry_z], dtype=np.float32)
    pour_pose = above_goal.copy()
    pour_pose[:2] = clip_xy_rect(
        pour_pose[:2] + lateral * float(cfg.pour_lateral_offset_m),
        x_extent=WORKSPACE_X_EXTENT_M,
        y_extent=WORKSPACE_Y_EXTENT_M,
    )

    def _pitch_at(pos: np.ndarray) -> float:
        """Lip-down pitch as a function of slide progress past the dig point."""
        s = float(np.dot(np.asarray(pos, dtype=np.float64)[:2] - front_low[:2].astype(np.float64), a[:2].astype(np.float64)))
        if s <= 0.0:
            return pitch_max
        if s >= scoop_slide:
            return 0.0
        return pitch_max * (1.0 - s / scoop_slide)

    if not carried:
        along_from_front = float(np.dot(spoon[:2] - front_low[:2], a[:2]))
        spat_low = float(spoon[2]) <= table_z + 0.05
        seated = along_from_front >= scoop_slide - 0.02
        dist_xy = float(np.linalg.norm(spoon[:2] - mat[:2]))
        approach_air = front_low.copy()
        approach_air[2] = max(float(spoon[2]), table_z + 0.08)
        spoon_pitch = _pitch_at(spoon)

        if spat_low and seated:
            keyframes = [
                (spoon, 0.0, 0.0),
                (lift, 0.0, 0.0),
                (above_goal, 0.0, 0.0),
                (pour_pose, float(POUR_GOAL_THETA_RAD), 0.0),
            ]
        elif spat_low and along_from_front >= -0.01:
            # Mid-scoop: keep rotating up from the current pitch as we cup.
            keyframes = [
                (spoon, 0.0, spoon_pitch),
                (scoop_low, 0.0, 0.0),
                (lift, 0.0, 0.0),
                (above_goal, 0.0, 0.0),
                (pour_pose, float(POUR_GOAL_THETA_RAD), 0.0),
            ]
        elif dist_xy < 0.22:
            # Close: descend into the dig at full lip-down, then scoop up.
            keyframes = [
                (spoon, 0.0, pitch_max),
                (front_low, 0.0, pitch_max),
                (scoop_mid, 0.0, pitch_max * mid_frac),
                (scoop_low, 0.0, 0.0),
                (lift, 0.0, 0.0),
                (above_goal, 0.0, 0.0),
                (pour_pose, float(POUR_GOAL_THETA_RAD), 0.0),
            ]
        else:
            # Far: stage above the dig point still pitched lip-down, then scoop.
            keyframes = [
                (spoon, 0.0, pitch_max),
                (approach_air, 0.0, pitch_max),
                (front_low, 0.0, pitch_max),
                (scoop_mid, 0.0, pitch_max * mid_frac),
                (scoop_low, 0.0, 0.0),
                (lift, 0.0, 0.0),
                (above_goal, 0.0, 0.0),
                (pour_pose, float(POUR_GOAL_THETA_RAD), 0.0),
            ]
    else:
        theta_now = float(pour_theta)
        mid_theta = min(float(POUR_GOAL_THETA_RAD), theta_now + 0.55)
        keyframes = [
            (spoon, theta_now, 0.0),
            (above_goal, 0.0 if theta_now < 0.05 else theta_now, 0.0),
            (pour_pose, mid_theta, 0.0),
            (pour_pose, float(POUR_GOAL_THETA_RAD), 0.0),
        ]

    rows = _densify_with_frames(
        keyframes,
        step_spacing_m=float(cfg.dense_step_spacing_m),
        motion_dir=a,
        pour_side=float(pour_side),
    )
    wp = _rows_to_waypoints(rows)
    if wp.shape[0] < num_steps:
        last = wp[-1].copy()
        wp = np.concatenate([wp, np.tile(last[None, :], (num_steps - wp.shape[0], 1))], axis=0)
    return wp[:num_steps]


def _bowl_seat_transform(cfg: SpoonPourGenConfig) -> np.ndarray:
    """Fixed material-in-contact-frame transform that seats it in the bowl cup.

    The material rides aligned with the contact frame, parked at the bowl center
    (``+x`` toward the handle) and lowered into the cup (``-z``), so it sits on
    top of the bowl surface rather than hanging at the lip plane.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([float(cfg.bowl_seat_x_m), 0.0, float(cfg.bowl_seat_z_m)], dtype=np.float64)
    return T


def _execute_chunk(
    cfg: SpoonPourGenConfig,
    plan: np.ndarray,
    *,
    material_xyz: np.ndarray,
    material_quat: np.ndarray,
    destination_xyz: np.ndarray,
    carried: bool,
    pour_theta: float,
    motion_dir: np.ndarray,
    pour_side: float,
    chunk: int,
    grasp: Optional[np.ndarray] = None,
    slip: bool = False,
    early_drop: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, bool, float, np.ndarray, np.ndarray,
    Optional[np.ndarray], bool, bool,
]:
    table_z = float(cfg.table_xyz_world[2])
    mat = np.asarray(material_xyz, dtype=np.float32).reshape(3).copy()
    quat = np.asarray(material_quat, dtype=np.float32).reshape(4).copy()
    a = _unit_xy(motion_dir)
    rng = rng if rng is not None else np.random.default_rng()
    rest_z = table_z + float(MATERIAL_HALF_M[2])
    half_x = float(MATERIAL_HALF_M[0])
    release_h = table_z + float(cfg.pour_release_z_above_table_m)
    release_theta = float(POUR_GOAL_THETA_RAD) - float(POUR_THETA_TOL_RAD)
    lateral = np.array([-a[1], a[0]], dtype=np.float32) * float(pour_side)
    slip_h = rest_z + float(cfg.slip_trigger_height_m)
    early_h = rest_z + float(cfg.early_drop_height_m)
    dest_xy = np.asarray(destination_xyz, dtype=np.float32).reshape(3)

    def _drop(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jitter = rng.uniform(-1.0, 1.0, size=2) * float(cfg.slip_jitter_m)
        out = np.array([float(xy[0]) + float(jitter[0]),
                        float(xy[1]) + float(jitter[1]), rest_z], dtype=np.float32)
        return out, _identity_quat()

    T_cm = None if grasp is None else np.asarray(grasp, dtype=np.float64).reshape(4, 4).copy()
    held = bool(carried or T_cm is not None)
    theta = float(pour_theta)
    released = False
    dropped = False

    n_exec = max(1, min(int(chunk), int(plan.shape[0])))
    xyz_trace: list[np.ndarray] = []
    quat_trace: list[np.ndarray] = []

    for i in range(n_exec):
        contact = plan[i, 0:3].astype(np.float32)
        normal = plan[i, 3:6].astype(np.float32)
        surface_dir = plan[i, 6:9].astype(np.float32)
        theta_i = _theta_from_waypoint(normal, a, pour_side)

        if not held:
            front_edge = mat - a * half_x
            passed_edge = float(np.dot(contact[:2] - front_edge[:2], a[:2])) >= -0.01
            past_center = float(np.dot(contact[:2] - mat[:2], a[:2]))
            low = float(contact[2]) <= rest_z + 0.04
            # Latch only after the scoop arc has cupped back toward level so the
            # material ends up seated in/on the bowl, not below the mesh.
            pitch_i = _pitch_from_waypoint(normal, a)
            cupped = pitch_i <= float(cfg.scoop_pitch_latch_rad)
            if passed_edge and low and past_center >= float(cfg.grab_seat_m) and cupped:
                # Seat the material in the bowl cup (rigid grab) so it sits on top
                # of the bowl and rides with it.
                T_cm = _bowl_seat_transform(cfg)
                held = True

        if held:
            T_wc = contact_frame_world(contact, normal, surface_dir)
            T_wm = T_wc @ T_cm
            mat = T_wm[:3, 3].astype(np.float32)
            quat = quat_xyzw_from_matrix(T_wm[:3, :3]).astype(np.float32)
            theta = max(0.0, float(theta_i))
            goal_dist = float(np.linalg.norm(mat[:2] - dest_xy[:2]))
            if slip and theta < release_theta and float(mat[2]) >= slip_h:
                # Mess up the pickup: material slips off mid-lift and falls back
                # to the ground, so the planner re-approaches and rescoops.
                mat, quat = _drop(mat[:2])
                held, T_cm, theta, dropped = False, None, 0.0, True
            elif (
                early_drop
                and theta < release_theta
                and float(mat[2]) >= early_h
                and goal_dist >= float(cfg.early_drop_min_goal_dist_m)
            ):
                # Dropped too early: material falls out partway through the carry,
                # landing mid-path (clearly short of the goal) for a retry.
                mat, quat = _drop(mat[:2])
                held, T_cm, theta, dropped = False, None, 0.0, True
            elif theta >= release_theta and float(mat[2]) >= release_h:
                # Contents pour out sideways and land on the goal/plate region.
                landing_xy = contact[:2] + lateral * float(cfg.pour_lateral_offset_m)
                mat = np.array([float(landing_xy[0]), float(landing_xy[1]), rest_z], dtype=np.float32)
                quat = _identity_quat()
                held = False
                T_cm = None
                released = True

        if float(mat[2]) < rest_z:
            mat[2] = rest_z
        xyz_trace.append(mat.copy())
        quat_trace.append(quat.copy())

    return (
        plan[n_exec - 1].astype(np.float32),
        mat,
        quat,
        bool(held),
        float(theta),
        np.stack(xyz_trace, axis=0).astype(np.float32),
        np.stack(quat_trace, axis=0).astype(np.float32),
        T_cm if held else None,
        bool(released),
        bool(dropped),
    )


def _reached_goal(material_xyz: np.ndarray, destination_xyz: np.ndarray, *, table_z: float, released: bool) -> bool:
    on_table = float(material_xyz[2]) <= table_z + float(MATERIAL_HALF_M[2]) + 0.02
    near_goal = float(np.linalg.norm(np.asarray(material_xyz)[:2] - np.asarray(destination_xyz)[:2])) <= 0.055
    return bool(released and on_table and near_goal)


def _gen_instruction(
    rng: np.random.Generator,
    *,
    material_word: str,
    destination_word: str,
) -> dict[str, Any]:
    instruction = _safe_format(
        str(rng.choice(TEMPLATES_POUR)),
        material=material_word,
        destination=destination_word,
        tool=TOOL_LABEL,
    )
    return {
        "instruction": instruction,
        "tool_label": TOOL_LABEL,
        "material_label": material_word,
        "destination_label": destination_word,
    }


def scene_to_datapoints(
    rng: np.random.Generator,
    cfg: SpoonPourGenConfig,
    *,
    shard_id: str,
    scene_index: int,
    base_datapoint_index: int,
) -> list[dict[str, Any]]:
    table_xyz = cfg.sample_table_xyz(rng)
    dp_cfg = replace(cfg, table_xyz_world=table_xyz)
    table_z = float(table_xyz[2])

    material_xyz, destination_xyz = _sample_material_and_goal(rng, dp_cfg)
    pan_center = _pour_pan_center_xy(material_xyz)
    spoon_home, spoon_normal, spoon_surface_dir, motion_dir = _sample_tool_pose_pour(
        rng, dp_cfg, material_xyz=material_xyz
    )
    pour_side = float(rng.choice([-1.0, 1.0]))
    body = _gen_instruction(
        rng,
        material_word=str(rng.choice(MATERIALS_POUR)),
        destination_word=str(rng.choice(DESTINATIONS_POUR)),
    )

    spoon = np.asarray(spoon_home, dtype=np.float32).reshape(3).copy()
    normal = np.asarray(spoon_normal, dtype=np.float32).reshape(3).copy()
    surface_dir = np.asarray(spoon_surface_dir, dtype=np.float32).reshape(3).copy()
    mat = np.asarray(material_xyz, dtype=np.float32).reshape(3).copy()
    mat_quat = _identity_quat()
    carried = False
    pour_theta = 0.0
    grasp: Optional[np.ndarray] = None

    chunk = max(1, int(cfg.executed_chunk))
    n_wp = int(cfg.num_output_waypoints)
    datapoints: list[dict[str, Any]] = []

    # Scoop attempt bookkeeping. ``attempting`` is True from the moment a fresh
    # approach begins until the attempt resolves (material dropped back on the
    # ground). ``pending_slip`` marks an attempt that slips off mid-lift;
    # ``pending_early_drop`` marks one that drops too early during the carry.
    # Both leave the material on the ground for an automatic retry.
    attempting = False
    pending_slip = False
    pending_early_drop = False
    scoop_attempts = 0

    for gen in range(int(cfg.max_generations)):
        if (not carried) and (not attempting):
            attempting = True
            force_success = scoop_attempts >= int(cfg.max_scoop_attempts) - 1
            fail = (not force_success) and (float(rng.random()) < float(cfg.scoop_fail_prob))
            pending_slip = bool(fail and float(rng.random()) < 0.5)
            pending_early_drop = bool(fail and not pending_slip)

        retrying_now = bool(scoop_attempts > 0 and not carried)
        target = plan_next_pour_steps(
            rng,
            dp_cfg,
            spoon_contact=spoon,
            material_xyz=mat,
            destination_xyz=destination_xyz,
            carried=carried,
            pour_theta=pour_theta,
            motion_dir=motion_dir,
            pour_side=pour_side,
            num_steps=n_wp,
        )
        target = _pad_target_waypoints(target, n_wp)
        (
            new_spoon,
            new_mat,
            new_mat_quat,
            new_carried,
            new_theta,
            mat_trace,
            quat_trace,
            new_grasp,
            released,
            dropped,
        ) = _execute_chunk(
            dp_cfg,
            target,
            material_xyz=mat,
            material_quat=mat_quat,
            destination_xyz=destination_xyz,
            carried=carried,
            pour_theta=pour_theta,
            motion_dir=motion_dir,
            pour_side=pour_side,
            chunk=chunk,
            grasp=grasp,
            slip=pending_slip,
            early_drop=pending_early_drop,
            rng=rng,
        )

        reached_goal = _reached_goal(new_mat, destination_xyz, table_z=table_z, released=released)
        dp_idx = int(base_datapoint_index + gen)
        datapoints.append(
            {
                "datapoint_id": f"{shard_id}_{dp_idx:06d}",
                "datapoint_index": dp_idx,
                "scene_index": int(scene_index),
                "window_index": int(gen),
                "rollout_step": int(gen * chunk),
                "in_contact": bool(carried),
                "reached_goal": bool(reached_goal),
                "scoop_failed": bool(dropped and not reached_goal),
                "retrying_scoop": bool(retrying_now),
                "scoop_attempt": int(scoop_attempts),
                "movement_token": "pour",
                "path_shape": "spoon_scoop_pour_side",
                "pour_side": float(pour_side),
                "instruction": body["instruction"],
                "tool_label": body["tool_label"],
                "tool_contact_xyz_world": spoon.tolist(),
                "tool_current_normal": normal.tolist(),
                "tool_current_surface_dir": surface_dir.tolist(),
                "has_material": True,
                "material_label": body["material_label"],
                "material_xyz_world": mat.tolist(),
                "material_quat_world": mat_quat.tolist(),
                "material_size": MATERIAL_SIZE_M.tolist(),
                "material_xyz_after_world": new_mat.tolist(),
                "material_quat_after_world": new_mat_quat.tolist(),
                "material_xyz_executed_world": mat_trace.tolist(),
                "material_quat_executed_world": quat_trace.tolist(),
                "has_destination": True,
                "destination_label": body["destination_label"],
                "destination_xyz_world": destination_xyz.tolist(),
                "pan_center_xy_world": [float(pan_center[0]), float(pan_center[1])],
                "table_label": TABLE_LABEL,
                "table_xyz_world": list(table_xyz),
                "waypoints": target.reshape(n_wp, 9).tolist(),
            }
        )

        if reached_goal:
            break

        # Resolve attempt state: a failure drop (slipped mid-lift or dropped too
        # early) leaves the material on the ground and ends the attempt, so the
        # next generation re-approaches with a freshly drawn outcome.
        if dropped:
            attempting = False
            pending_slip = False
            pending_early_drop = False
            scoop_attempts += 1

        spoon = new_spoon[0:3].copy()
        normal = new_spoon[3:6].copy()
        surface_dir = new_spoon[6:9].copy()
        mat = new_mat
        mat_quat = new_mat_quat
        carried = new_carried
        pour_theta = new_theta
        grasp = new_grasp

    return datapoints


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    cfg: SpoonPourGenConfig,
) -> dict[str, Any]:
    shard_id = f"spoon_pour_reactive_{shard_idx:04d}"
    datapoints: list[dict[str, Any]] = []
    dp_idx = 0
    for scene_i in range(int(cfg.scenes_per_shard)):
        rng = _datapoint_rng(seed, shard_idx, scene_i)
        scene_dps = scene_to_datapoints(
            rng,
            cfg,
            shard_id=shard_id,
            scene_index=scene_i,
            base_datapoint_index=dp_idx,
        )
        datapoints.extend(scene_dps)
        dp_idx += len(scene_dps)

    return {
        "dataset_id": DATASET_ID,
        "shard_id": shard_id,
        "scene_id": shard_id,
        "generator": "spoon_pour_reactive_v1",
        "seed": int(seed),
        "num_datapoints": len(datapoints),
        "scenes_per_shard": int(cfg.scenes_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reactive closed-loop spoon scoop-and-pour shards (dataset_0013)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0013_spoon_pour_reactive/shards",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--scenes_per_shard", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SpoonPourGenConfig(scenes_per_shard=int(args.scenes_per_shard))
    summary: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "num_shards": int(args.num_shards),
        "scenes_per_shard": int(args.scenes_per_shard),
        "seed": int(args.seed),
        "dense_step_spacing_m": cfg.dense_step_spacing_m,
        "shard_paths": [],
    }

    for shard_idx in range(int(args.start_shard), int(args.num_shards)):
        shard = build_shard(shard_idx=shard_idx, seed=int(args.seed), cfg=cfg)
        out_path = out_dir / f"{shard['shard_id']}_shard.json"
        out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        summary["shard_paths"].append(str(out_path))
        print(
            f"Wrote {out_path} ({shard['num_datapoints']} datapoints, "
            f"{shard['scenes_per_shard']} scenes)"
        )

    summary_path = out_dir.parent / f"{DATASET_ID}_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
