"""Procedurally generate reactive closed-loop spatula-flip shards (dataset_0012).

Each scene rolls out a receding-horizon flip: the spatula scoops under a flat
object, lifts, and rotates ~180 deg about the left-right axis until the object
is inverted and held at an apex in the air.
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
    TABLE_XY_JITTER_M,
    TABLE_Z_RANGE,
    WORKSPACE_X_EXTENT_M,
    WORKSPACE_Y_EXTENT_M,
    clip_xy_rect,
    sample_xy_rect,
)

DATASET_ID = "dataset_0012_spatula_flip_reactive"
TOOL_LABEL = "the spatula"

MATERIALS_FLIP = (
    "the toast",
    "the block",
    "the flat box",
    "the tile",
    "the card",
    "the pancake",
    "the patty",
    "the cookie",
    "the cracker",
    "the coaster",
    "the lid",
    "the slab",
)
DESTINATIONS_FLIP = (
    "the pan",
    "the pan floor",
    "the pan bottom",
    "the skillet",
    "the frying pan",
    "the pan base",
)
TEMPLATES_FLIP = (
    "Flip {material} over with {tool} so it lands inverted in {destination}.",
    "Use {tool} to scoop under {material} and flip it upside down in {destination}.",
    "Scoop and flip {material} with {tool}, leaving it inverted in {destination}.",
    "Slide {tool} under {material}, lift, and turn it over so it rests upside down in {destination}.",
    "Turn {material} over with {tool} and set it down inverted in {destination}.",
    "With {tool}, get under {material} and flip it so the underside faces up in {destination}.",
    "Invert {material} using {tool}, ending with it flipped over in {destination}.",
    "Use {tool} to lift {material} and rotate it upside down into {destination}.",
    "Flip {material} with {tool}, leaving it face down in {destination}.",
    "Slide the blade of {tool} beneath {material} and flip it onto its back in {destination}.",
    "Scoop {material} up with {tool} and flip it over so it lands inverted in {destination}.",
    "Using {tool}, flip {material} top-to-bottom and rest it inverted in {destination}.",
    "Work {tool} under {material}, then flip it upside down in {destination}.",
    "Flip {material} over in {destination} with {tool} so its top now faces down.",
)

# Flat box target (length, width, height) in meters.
MATERIAL_SIZE_M = np.array([0.09, 0.07, 0.012], dtype=np.float32)
MATERIAL_HALF_M = 0.5 * MATERIAL_SIZE_M

FLIP_APEX_Z_ABOVE_TABLE_M = 0.12
FLIP_GOAL_THETA_RAD = float(np.pi)
FLIP_THETA_TOL_RAD = 0.25

# Handleless pan used by the spatula flip task. The simulator approximates the
# curved wall with stacked circular rim segments; the dataset uses the same
# dimensions to keep generated trajectories inside the pan.
PAN_RADIUS_M = 4.5 * 0.0254
PAN_WALL_HEIGHT_M = 2.0 * 0.0254
PAN_CENTER_XY_M = (0.06, 0.0)


@dataclass
class FlipGenConfig(SweepGenConfig):
    """Reactive closed-loop spatula-flip generator."""

    dense_step_spacing_m: float = 0.04
    num_output_waypoints: int = 15
    max_generations: int = 40
    executed_chunk: int = 5
    scenes_per_shard: int = 150
    min_tool_material_sep_m: float = 0.16
    behind_offset_m_range: tuple[float, float] = (0.065, 0.09)
    # Slide the blade further under the object so it seats well onto the spatula
    # (object center over the blade) before the lift/flip begins.
    scoop_slide_m: float = 0.14
    # How far in front of the object's near (tool-side) edge the blade descends to
    # the table. Larger -> the blade touches down earlier, before the object, then
    # brushes/slides into and under it, instead of coming straight down on top of
    # its near edge. The slide is extended by this amount so the blade still seats
    # fully under the object.
    descend_lead_m: float = 0.04
    # Dramatic dig-in pitch (radians) applied as the blade descends into the pan
    # and slides under the object: tips the leading edge down and lifts the long
    # handle up so it clears the pan rim instead of colliding with it. Ramps back
    # to 0 as the blade lifts/levels for the flip.
    scoop_entry_pitch_rad: float = 0.85
    # The tool always enters over the +x rim, so spawn the object toward the far
    # (-x) half of the pan: the blade then slides in away from its entry rim and
    # the long handle crosses the near rim far along its length (well clear of the
    # wall). theta is measured from +x about the pan center; pi is the far rim.
    obj_spawn_theta_range_rad: tuple[float, float] = (0.55 * np.pi, 1.45 * np.pi)
    apex_z_above_table_m: float = FLIP_APEX_Z_ABOVE_TABLE_M
    blade_contact_z_offset_m: float = 0.003
    # Blade must be this far past the object center (at table height) before the
    # object is grabbed, so it seats well onto the spatula and sits on top.
    grab_seat_m: float = 0.03
    pan_center_xy_m: tuple[float, float] = PAN_CENTER_XY_M
    pan_radius_m: float = PAN_RADIUS_M
    pan_wall_height_m: float = PAN_WALL_HEIGHT_M
    pan_entry_clearance_m: float = 0.035
    pan_spawn_margin_m: float = 0.035
    # Reachable-workspace box for the flat object. The pan is placed AROUND the
    # sampled object (material-relative): the object xy is drawn first inside this
    # box, then the pan center is derived so the object sits inside the pan on the
    # far (-x) side from the +x tool entry. Bounds keep the whole pan + the tool
    # staging band (outside the +x rim) on the table.
    material_x_range_m: tuple[float, float] = (-0.10, 0.02)
    material_y_range_m: tuple[float, float] = (-0.07, 0.07)
    # Reactive failure + retry. Two failure modes, both leaving the object back on
    # the ground (right-side up) so the planner naturally re-approaches and tries
    # again (no special retry logic): (1) the object slips off mid-lift
    # (``scoop_fail_prob`` per lift), and (2) the flip completes the arc but the
    # object lands NOT inverted (``flip_fail_prob``). The final allowed attempt is
    # forced to succeed so scenes still terminate with a completed flip.
    scoop_fail_prob: float = 0.25
    flip_fail_prob: float = 0.25
    max_scoop_attempts: int = 5
    slip_trigger_height_m: float = 0.05
    slip_jitter_m: float = 0.02
    # Object is placed back on the ground once it reaches near the apex with the
    # flip (nearly) complete.
    place_apex_frac: float = 0.85

    # Per-scene jitter ranges for diversity (so the simulator filters what works
    # across a spread of styles). These are sampled once per scene and held
    # constant through that scene's rollout. The dig-in pitch is only ever
    # jittered *up* from the tuned baseline so the handle never drops closer to
    # the pan rim than the validated clearance.
    scoop_entry_pitch_rad_range: tuple[float, float] = (0.85, 1.00)
    apex_z_above_table_m_range: tuple[float, float] = (0.10, 0.15)
    scoop_slide_m_range: tuple[float, float] = (0.12, 0.16)
    descend_lead_m_range: tuple[float, float] = (0.025, 0.06)


def _unit_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _flip_axis(motion_dir: np.ndarray) -> np.ndarray:
    """Axis the blade rotates about during the flip.

    Using the forward/long axis (the spatula's sliding direction) rolls the blade
    sideways, so the object is flipped *to the side* and the long handle stays
    roughly horizontal instead of swinging end-over-end. The handle's in-plane
    direction (``surface_dir = -motion``) is invariant under this roll.
    """
    return _unit_xy(motion_dir).astype(np.float64)


def _pitch_axis(motion_dir: np.ndarray) -> np.ndarray:
    """Horizontal left-right axis used to pitch the blade during the scoop entry.

    A positive pitch about this axis digs the leading blade tip down toward the
    table while raising the long handle up and out of the pan, so the handle
    clears the rim wall instead of plowing into it as the blade descends.
    """
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    a = _unit_xy(motion_dir).astype(np.float64)
    p = np.cross(up, a)
    n = float(np.linalg.norm(p))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return (p / n).astype(np.float64)


def _rot_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-9)
    x, y, z = a
    c, s = float(np.cos(theta)), float(np.sin(theta))
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _blade_frame(
    motion_dir: np.ndarray,
    theta: float,
    pitch: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (normal, surface_dir, R_blade) for the blade orientation.

    ``theta`` rolls the blade about the forward/motion axis (the flip), while
    ``pitch`` tilts it about the horizontal left-right axis (the scoop dig-in
    that raises the handle clear of the pan rim). The pitch is applied on top of
    the flip so the handle stays lifted regardless of the current flip angle.
    """
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    a = _unit_xy(motion_dir).astype(np.float64)
    R_flip = _rot_axis_angle(_flip_axis(motion_dir), float(theta))
    R_pitch = _rot_axis_angle(_pitch_axis(motion_dir), float(pitch))
    R = R_pitch @ R_flip
    normal = (R @ up).astype(np.float32)
    surface_dir = (R @ (-a)).astype(np.float32)
    return normal, surface_dir, R.astype(np.float64)


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _object_up_dot_z(quat_xyzw: np.ndarray) -> float:
    R = matrix_from_quat_xyzw(quat_xyzw)
    up = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(up[2])


def _sample_tool_pose_flip(
    rng: np.random.Generator,
    cfg: FlipGenConfig,
    *,
    object_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_z = float(cfg.table_xyz_world[2])
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3)
    center = np.asarray(cfg.pan_center_xy_m, dtype=np.float64).reshape(2)
    # Stage just outside the +x pan rim (handle toward robot side) and above the
    # lip; the planner then descends into the pan before sliding under the toast.
    outside = float(cfg.pan_radius_m) + float(rng.uniform(0.03, 0.06))
    xy = np.array(
        [
            center[0] + outside,
            float(obj[1]) + float(rng.uniform(-0.03, 0.03)),
        ],
        dtype=np.float64,
    )
    z = table_z + float(cfg.pan_wall_height_m) + float(cfg.pan_entry_clearance_m)
    contact = np.array([float(xy[0]), float(xy[1]), z], dtype=np.float32)
    motion = _unit_xy(obj - contact)
    normal, surface_dir, _ = _blade_frame(motion, 0.0)
    return contact, normal, surface_dir


def _sample_material_pan_apex(
    rng: np.random.Generator,
    cfg: FlipGenConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the object first, then place the pan AROUND it (material-relative).

    The object xy is drawn inside the reachable-workspace box, then the pan
    center is derived so the object sits inside the pan on the far (-x) side from
    the +x tool entry -- the exact object-relative-to-pan offset distribution is
    preserved from the previous fixed-pan generator, only inverted so the object
    location is the free variable and the pan tracks it.

    Returns ``(material_xyz, apex_xyz, pan_center_xy)``.
    """
    table_z = float(cfg.table_xyz_world[2])
    mat_x = float(rng.uniform(*cfg.material_x_range_m))
    mat_y = float(rng.uniform(*cfg.material_y_range_m))
    mat_xy = np.array([mat_x, mat_y], dtype=np.float64)

    # Object offset from the pan center (same distribution as the old fixed pan).
    max_r = max(
        0.01,
        float(cfg.pan_radius_m)
        - float(cfg.pan_spawn_margin_m)
        - float(np.linalg.norm(MATERIAL_HALF_M[:2])),
    )
    radius = max_r * float(np.sqrt(rng.random()))
    theta = float(rng.uniform(*cfg.obj_spawn_theta_range_rad))
    offset = radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    # object = pan_center + offset  =>  pan_center = object - offset.
    pan_center = (mat_xy - offset).astype(np.float64)

    mat = np.array(
        [mat_x, mat_y, table_z + float(MATERIAL_HALF_M[2])],
        dtype=np.float32,
    )
    apex = np.array(
        [mat_x, mat_y, table_z + float(cfg.apex_z_above_table_m)],
        dtype=np.float32,
    )
    return mat, apex, pan_center


def _interpolate_contacts(
    start: np.ndarray,
    end: np.ndarray,
    *,
    n: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    a = np.asarray(start, dtype=np.float64).reshape(3)
    b = np.asarray(end, dtype=np.float64).reshape(3)
    for i in range(1, n + 1):
        t = float(i) / float(n)
        out.append(((1.0 - t) * a + t * b).astype(np.float32))
    return out


def _entry_pitch_for_contact(
    contact: np.ndarray,
    cfg: FlipGenConfig,
) -> float:
    """Dig-in pitch needed at a contact so the long handle clears the pan rim.

    The blade's contact point sits at its leading edge and the ~0.34 m handle
    extends back toward the arm, so whenever the blade is *inside the pan* and
    *below the rim* the handle would otherwise sweep straight through the rim
    wall. We pitch the handle up (tip down) by an amount that ramps from the full
    dig-in at table height to zero once the blade has risen clear above the rim.
    Outside the pan footprint no pitch is needed (the handle rests over the open
    table), so the blade lands flat to slide under the object.
    """
    pitch_in = float(cfg.scoop_entry_pitch_rad)
    if pitch_in <= 0.0:
        return 0.0
    table_z = float(cfg.table_xyz_world[2])
    rim_top = table_z + float(cfg.pan_wall_height_m)
    clear_z = rim_top + 0.03  # handle considered clear once this far above table
    pan_center = np.asarray(cfg.pan_center_xy_m, dtype=np.float64).reshape(2)
    c = np.asarray(contact, dtype=np.float64).reshape(3)
    r = float(np.linalg.norm(c[:2] - pan_center))
    if r > float(cfg.pan_radius_m):
        return 0.0
    z = float(c[2])
    if z >= clear_z:
        return 0.0
    if z <= table_z:
        return pitch_in
    return pitch_in * (clear_z - z) / (clear_z - table_z)


def _densify_with_frames(
    keyframes: list[tuple[np.ndarray, float]],
    *,
    step_spacing_m: float,
    motion_dir: np.ndarray,
    cfg: FlipGenConfig,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Densify (contact, theta) keyframes into [contact, normal, surface_dir] rows.

    The entry pitch is derived from each densified contact's geometry relative to
    the pan (see ``_entry_pitch_for_contact``) so the handle stays clear of the
    rim wall throughout the descent, slide, and lift, regardless of phase.
    """
    dense: list[tuple[np.ndarray, float]] = []
    for i in range(len(keyframes) - 1):
        ca, ta = keyframes[i]
        cb, tb = keyframes[i + 1]
        ca = np.asarray(ca, dtype=np.float64).reshape(3)
        cb = np.asarray(cb, dtype=np.float64).reshape(3)
        seg_len = float(np.linalg.norm(cb - ca))
        n_steps = max(1, int(np.round(seg_len / max(step_spacing_m, 1e-4))))
        for j in range(n_steps):
            t = float(j) / float(n_steps)
            c = ((1.0 - t) * ca + t * cb).astype(np.float32)
            th = (1.0 - t) * float(ta) + t * float(tb)
            dense.append((c, th))
    dense.append(
        (
            np.asarray(keyframes[-1][0], dtype=np.float32).reshape(3),
            float(keyframes[-1][1]),
        )
    )

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for contact, theta in dense:
        pitch = _entry_pitch_for_contact(contact, cfg)
        normal, surface_dir, _ = _blade_frame(motion_dir, theta, pitch)
        rows.append((contact, normal, surface_dir))
    return rows


def _rows_to_waypoints(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    n = len(rows)
    wp = np.zeros((n, 9), dtype=np.float32)
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


def plan_next_flip_steps(
    rng: np.random.Generator,
    cfg: FlipGenConfig,
    *,
    spatula_contact: np.ndarray,
    object_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    scooped: bool,
    flip_theta: float,
    motion_dir: np.ndarray,
    num_steps: int = 15,
) -> np.ndarray:
    """Plan the next dense spatula waypoints from the current flip state."""
    table_z = float(cfg.table_xyz_world[2])
    contact_h = float(cfg.blade_contact_z_offset_m)
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3)
    dest = np.asarray(destination_xyz, dtype=np.float32).reshape(3)
    spat = np.asarray(spatula_contact, dtype=np.float32).reshape(3)
    a = _unit_xy(motion_dir)

    behind_m = float(rng.uniform(*cfg.behind_offset_m_range))
    behind_xy = obj[:2] - a[:2] * behind_m
    pan_center = np.asarray(cfg.pan_center_xy_m, dtype=np.float32).reshape(2)
    pan_max_r = max(0.01, float(cfg.pan_radius_m) - float(cfg.pan_spawn_margin_m))

    def _clip_pan_xy(xy_in: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_in, dtype=np.float32).reshape(2)
        rel = xy - pan_center
        r = float(np.linalg.norm(rel))
        if r <= pan_max_r or r < 1e-6:
            return xy
        return (pan_center + rel / r * pan_max_r).astype(np.float32)

    behind_xy = _clip_pan_xy(behind_xy)
    table_contact_z = table_z + contact_h
    rim_clear_z = (
        table_z + float(cfg.pan_wall_height_m) + float(cfg.pan_entry_clearance_m)
    )
    if not scooped:
        dist_xy = float(np.linalg.norm(spat[:2] - obj[:2]))
        near_table = float(spat[2]) <= table_z + 0.06
        behind_table = np.array(
            [behind_xy[0], behind_xy[1], table_contact_z], dtype=np.float32
        )
        # Near (tool-side) edge of the object. The blade touches down a bit in
        # front of this edge (``descend_lead_m`` toward the tool, in -motion) so it
        # reaches the table beside/before the object and then brushes forward under
        # it, instead of dropping straight down onto the near edge. The slide is
        # lengthened by the same lead so the blade still seats fully under the obj.
        leading_edge = (obj - a * float(MATERIAL_HALF_M[0])).astype(np.float32)
        leading_edge[:2] = _clip_pan_xy(leading_edge[:2])
        descend_lead = float(cfg.descend_lead_m)
        descend_xy = leading_edge[:2] - a[:2] * descend_lead
        descend_xy = _clip_pan_xy(descend_xy)
        under_contact = np.array(
            [descend_xy[0], descend_xy[1], table_contact_z], dtype=np.float32
        )
        slide_contact = (
            under_contact + a * (float(cfg.scoop_slide_m) + descend_lead)
        ).astype(np.float32)
        slide_contact[:2] = _clip_pan_xy(slide_contact[:2])
        lift_start = slide_contact.copy()
        lift_start[2] = rim_clear_z + 0.02
        apex_mid = np.array(
            [dest[0], dest[1], 0.5 * (lift_start[2] + dest[2])], dtype=np.float32
        )

        # How far the blade has already slid under (past the near edge, in +motion).
        spat_low = float(spat[2]) <= table_z + 0.03
        along_from_near = float(np.dot(spat[:2] - leading_edge[:2], a[:2]))
        seated = along_from_near >= float(cfg.scoop_slide_m) - 0.03

        if spat_low and seated:
            # Already slid fully under the object: lift and flip from here, never
            # backtrack toward the near edge.
            lift_here = spat.copy()
            lift_here[2] = rim_clear_z + 0.02
            keyframes = [
                (spat, 0.0),
                (lift_here, 0.15),
                (apex_mid, 1.0),
                (dest, float(FLIP_GOAL_THETA_RAD)),
            ]
        elif spat_low and along_from_near >= -0.01:
            # Under the near edge but not fully seated: keep sliding forward, then lift.
            keyframes = [
                (spat, 0.0),
                (slide_contact, 0.0),
                (lift_start, 0.15),
                (apex_mid, 1.0),
                (dest, float(FLIP_GOAL_THETA_RAD)),
            ]
        elif dist_xy < 0.16 and near_table:
            # Close and low: drop to the near edge, slide under, then lift.
            keyframes = [
                (spat, 0.0),
                (under_contact, 0.0),
                (slide_contact, 0.0),
                (lift_start, 0.15),
                (apex_mid, 1.0),
                (dest, float(FLIP_GOAL_THETA_RAD)),
            ]
        elif dist_xy < 0.28:
            # Descend over the lip and come straight down ONTO the object's near
            # (leading) edge -- never behind it toward the entry rim, which would
            # bury the long handle in the near wall -- then slide under.
            entry_air = under_contact.copy()
            entry_air[2] = rim_clear_z
            keyframes = [
                (spat, 0.0),
                (entry_air, 0.0),
                (under_contact, 0.0),
                (slide_contact, 0.0),
                (lift_start, 0.15),
                (apex_mid, 1.0),
                (dest, float(FLIP_GOAL_THETA_RAD)),
            ]
        else:
            approach_air = np.array(
                [under_contact[0], under_contact[1], rim_clear_z], dtype=np.float32
            )
            keyframes = [
                (spat, 0.0),
                (approach_air, 0.0),
                (under_contact, 0.0),
                (slide_contact, 0.0),
                (lift_start, 0.15),
                (apex_mid, 1.0),
                (dest, float(FLIP_GOAL_THETA_RAD)),
            ]
    else:
        theta_now = float(flip_theta)
        mid_theta = min(float(FLIP_GOAL_THETA_RAD), theta_now + 0.6)
        apex_mid = np.array(
            [
                dest[0],
                dest[1],
                max(spat[2], table_z + 0.06) + 0.5 * (dest[2] - max(spat[2], table_z + 0.06)),
            ],
            dtype=np.float32,
        )
        keyframes = [
            (spat, theta_now),
            (np.array([obj[0], obj[1], apex_mid[2]], dtype=np.float32), mid_theta),
            (dest, float(FLIP_GOAL_THETA_RAD)),
        ]

    rows = _densify_with_frames(
        keyframes,
        step_spacing_m=float(cfg.dense_step_spacing_m),
        motion_dir=a,
        cfg=cfg,
    )
    wp = _rows_to_waypoints(rows)
    if wp.shape[0] < num_steps:
        last = wp[-1].copy()
        extra = np.tile(last[None, :], (num_steps - wp.shape[0], 1))
        wp = np.concatenate([wp, extra], axis=0)
    return wp[:num_steps]


def _theta_from_waypoint(normal: np.ndarray, motion_dir: np.ndarray) -> float:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = _flip_axis(motion_dir)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    # The scoop dig-in pitches the blade about ``axis`` (the motion direction),
    # which adds an out-of-plane component along ``axis``. Project it out so the
    # recovered flip angle reflects only the roll about ``axis`` and is not
    # polluted by the entry pitch.
    n_perp = n - float(np.dot(n, axis)) * axis
    if float(np.linalg.norm(n_perp)) < 1e-6:
        return 0.0
    n_perp = n_perp / float(np.linalg.norm(n_perp))
    cos_t = float(np.clip(np.dot(n_perp, up), -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if float(np.dot(np.cross(up, n_perp), axis)) < 0.0:
        theta = -theta
    return float(theta)


def _execute_chunk(
    cfg: FlipGenConfig,
    plan: np.ndarray,
    *,
    object_xyz: np.ndarray,
    object_quat: np.ndarray,
    scooped: bool,
    flip_theta: float,
    motion_dir: np.ndarray,
    chunk: int,
    grasp: Optional[np.ndarray] = None,
    under_latched: bool = False,
    slip: bool = False,
    flip_fail: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, bool, float, np.ndarray, np.ndarray,
    bool, bool, Optional[np.ndarray], bool,
]:
    """Execute first ``chunk`` planned steps; update object pose analytically.

    The object stays resting on the table while the blade slides underneath it
    (no teleport, no backward motion). ``under_latched`` records that the blade
    has slid under the near edge at table height; once it then starts to lift the
    object is rigidly grabbed at its current resting pose (locked into the blade
    contact frame ``grasp`` = the object-in-contact transform) so it lifts and
    flips with the blade seamlessly.

    The object always ends back on the ground:
      * ``slip`` -> it slips off mid-lift and drops flat (right-side up) early.
      * otherwise, once the flip arc nears the apex it is placed back on the
        ground -- inverted on success, or right-side up when ``flip_fail`` (the
        flip didn't take). Either way the grasp releases; a right-side-up landing
        looks like the start state so the planner re-approaches automatically.
    """
    table_z = float(cfg.table_xyz_world[2])
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3).copy()
    quat = np.asarray(object_quat, dtype=np.float32).reshape(4).copy()
    a = _unit_xy(motion_dir)
    rng = rng if rng is not None else np.random.default_rng()
    half_x = float(MATERIAL_HALF_M[0])
    rest_z = table_z + float(MATERIAL_HALF_M[2])
    slip_h = rest_z + float(cfg.slip_trigger_height_m)
    place_z = rest_z + float(cfg.place_apex_frac) * float(cfg.apex_z_above_table_m)
    flip_done_theta = float(FLIP_GOAL_THETA_RAD) - float(FLIP_THETA_TOL_RAD)

    def _drop(xy: np.ndarray, inverted_quat: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        jitter = rng.uniform(-1.0, 1.0, size=2) * float(cfg.slip_jitter_m)
        out = np.array([float(xy[0]) + float(jitter[0]),
                        float(xy[1]) + float(jitter[1]), rest_z], dtype=np.float32)
        q = _identity_quat() if inverted_quat is None else np.asarray(inverted_quat, dtype=np.float32)
        return out, q

    T_co = None if grasp is None else np.asarray(grasp, dtype=np.float64).reshape(4, 4).copy()
    grasped = T_co is not None
    under = bool(under_latched)
    theta = float(flip_theta)

    n_exec = max(1, min(int(chunk), int(plan.shape[0])))
    xyz_trace: list[np.ndarray] = []
    quat_trace: list[np.ndarray] = []
    made_attempt = False
    dropped = False

    for i in range(n_exec):
        contact = plan[i, 0:3].astype(np.float32)
        normal = plan[i, 3:6].astype(np.float32)
        surface_dir = plan[i, 6:9].astype(np.float32)
        theta_i = _theta_from_waypoint(normal, a)

        if not grasped:
            lead = obj - a * half_x  # near (tool-side) edge of the object
            passed_edge = float(np.dot(contact[:2] - lead[:2], a[:2])) >= -0.01
            past_center = float(np.dot(contact[:2] - obj[:2], a[:2]))
            low = float(contact[2]) <= float(obj[2]) + 0.02
            if passed_edge and low:
                # Blade is under the object footprint at table height.
                under = True
                made_attempt = True
            elif not passed_edge:
                # Backed out behind the near edge: no longer under.
                under = False
            if under and low and past_center >= float(cfg.grab_seat_m):
                # Seated under the object at table height: lock it to the blade
                # at its resting pose so it sits ON TOP of the blade and lifts
                # with it (no teleport, no backward motion).
                T_wc = contact_frame_world(contact, normal, surface_dir)
                T_wo = np.eye(4, dtype=np.float64)
                T_wo[:3, :3] = matrix_from_quat_xyzw(quat)
                T_wo[:3, 3] = obj.astype(np.float64)
                T_co = np.linalg.inv(T_wc) @ T_wo
                grasped = True

        if grasped:
            T_wc = contact_frame_world(contact, normal, surface_dir)
            T_wo = T_wc @ T_co
            obj = T_wo[:3, 3].astype(np.float32)
            quat = quat_xyzw_from_matrix(T_wo[:3, :3]).astype(np.float32)
            if float(obj[2]) < rest_z:
                obj[2] = rest_z
            theta = theta_i
            if slip and theta_i < flip_done_theta and float(obj[2]) >= slip_h:
                # Slips off the blade mid-lift: drops back flat (right-side up).
                obj, quat = _drop(obj[:2], None)
                grasped, T_co, theta, under, dropped = False, None, 0.0, False, True
            elif theta_i >= flip_done_theta and float(obj[2]) >= place_z:
                # Flip arc completed near the apex: place the object back on the
                # ground -- inverted on success, right-side up when the flip fails.
                landed = None if flip_fail else quat.copy()
                obj, quat = _drop(obj[:2], landed)
                grasped, T_co, under, dropped = False, None, False, True
                theta = 0.0 if flip_fail else float(theta_i)

        xyz_trace.append(obj.copy())
        quat_trace.append(quat.copy())

    new_spatula = plan[n_exec - 1].astype(np.float32)
    new_grasp = T_co if grasped else None
    new_under = bool(under and not grasped)
    return (
        new_spatula,
        obj,
        quat,
        grasped,
        theta,
        np.stack(xyz_trace, axis=0).astype(np.float32),
        np.stack(quat_trace, axis=0).astype(np.float32),
        made_attempt,
        dropped,
        new_grasp,
        new_under,
    )


def _flip_reached_goal(
    object_quat: np.ndarray,
    object_xyz: np.ndarray,
    *,
    table_z: float,
    cfg: FlipGenConfig,
    released: bool,
) -> bool:
    # Goal: the object is flipped (inverted) and resting on the ground, released
    # from the blade. A right-side-up object on the ground is just the start
    # state, so the planner re-approaches and tries again automatically.
    inverted = _object_up_dot_z(object_quat) < -0.5
    on_ground = float(object_xyz[2]) <= table_z + float(MATERIAL_HALF_M[2]) + 0.02
    return bool(inverted and on_ground and released)


def _gen_flip_instruction(
    rng: np.random.Generator,
    *,
    material_word: str,
    destination_word: str,
) -> dict[str, Any]:
    template = str(rng.choice(TEMPLATES_FLIP))
    instruction = _safe_format(
        template,
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
    cfg: FlipGenConfig,
    *,
    shard_id: str,
    scene_index: int,
    base_datapoint_index: int,
) -> list[dict[str, Any]]:
    table_xyz = cfg.sample_table_xyz(rng)
    # Sample per-scene style jitter (held fixed for this scene's rollout).
    scoop_pitch = float(rng.uniform(*cfg.scoop_entry_pitch_rad_range))
    apex_z = float(rng.uniform(*cfg.apex_z_above_table_m_range))
    scoop_slide = float(rng.uniform(*cfg.scoop_slide_m_range))
    descend_lead = float(rng.uniform(*cfg.descend_lead_m_range))
    dp_cfg = replace(
        cfg,
        table_xyz_world=table_xyz,
        scoop_entry_pitch_rad=scoop_pitch,
        apex_z_above_table_m=apex_z,
        scoop_slide_m=scoop_slide,
        descend_lead_m=descend_lead,
    )
    table_z = float(table_xyz[2])

    # Sample the flat object first, then place the pan AROUND it and re-bind the
    # per-scene pan center so the tool staging / planner / pitch all reference the
    # material-relative pan.
    material_xyz, destination_xyz, pan_center = _sample_material_pan_apex(rng, dp_cfg)
    dp_cfg = replace(dp_cfg, pan_center_xy_m=(float(pan_center[0]), float(pan_center[1])))
    tool_home, tool_normal, tool_surface_dir = _sample_tool_pose_flip(
        rng, dp_cfg, object_xyz=material_xyz
    )
    motion_dir = _unit_xy(material_xyz - tool_home)

    body = _gen_flip_instruction(
        rng,
        material_word=str(rng.choice(MATERIALS_FLIP)),
        destination_word=str(rng.choice(DESTINATIONS_FLIP)),
    )

    spatula = np.asarray(tool_home, dtype=np.float32).reshape(3).copy()
    spat_normal = np.asarray(tool_normal, dtype=np.float32).reshape(3).copy()
    spat_surface_dir = np.asarray(tool_surface_dir, dtype=np.float32).reshape(3).copy()
    obj = np.asarray(material_xyz, dtype=np.float32).reshape(3).copy()
    obj_quat = _identity_quat()
    scooped = False
    flip_theta = 0.0
    grasp: Optional[np.ndarray] = None
    under_latched = False

    chunk = max(1, int(cfg.executed_chunk))
    n_wp = int(cfg.num_output_waypoints)
    datapoints: list[dict[str, Any]] = []

    # Scoop attempt bookkeeping. ``attempting`` is True from the moment a fresh
    # approach begins until the attempt resolves (object dropped back on the
    # ground). ``pending_slip`` marks an attempt that slips off mid-lift;
    # ``pending_flip_fail`` marks one that completes the arc but lands the object
    # right-side up. Both leave the object on the ground for an automatic retry.
    attempting = False
    pending_slip = False
    pending_flip_fail = False
    scoop_attempts = 0

    for gen in range(int(cfg.max_generations)):
        if (not scooped) and (not attempting):
            attempting = True
            force_success = scoop_attempts >= int(cfg.max_scoop_attempts) - 1
            pending_slip = (not force_success) and (
                float(rng.random()) < float(cfg.scoop_fail_prob)
            )
            pending_flip_fail = (
                (not force_success)
                and (not pending_slip)
                and (float(rng.random()) < float(cfg.flip_fail_prob))
            )

        retrying_now = bool(scoop_attempts > 0 and not scooped)
        target = plan_next_flip_steps(
            rng,
            dp_cfg,
            spatula_contact=spatula,
            object_xyz=obj,
            destination_xyz=destination_xyz,
            scooped=scooped,
            flip_theta=flip_theta,
            motion_dir=motion_dir,
            num_steps=n_wp,
        )
        target = _pad_target_waypoints(target, num_steps=n_wp)

        (
            new_spatula,
            new_obj,
            new_quat,
            new_scooped,
            new_theta,
            mat_trace,
            quat_trace,
            made_attempt,
            dropped,
            new_grasp,
            new_under,
        ) = _execute_chunk(
            dp_cfg,
            target,
            object_xyz=obj,
            object_quat=obj_quat,
            scooped=scooped,
            flip_theta=flip_theta,
            motion_dir=motion_dir,
            chunk=chunk,
            grasp=grasp,
            under_latched=under_latched,
            slip=pending_slip,
            flip_fail=pending_flip_fail,
            rng=rng,
        )

        reached_goal = _flip_reached_goal(
            new_quat,
            new_obj,
            table_z=table_z,
            cfg=dp_cfg,
            released=not new_scooped,
        )

        dp_idx = int(base_datapoint_index + gen)
        datapoints.append(
            {
                "datapoint_id": f"{shard_id}_{dp_idx:06d}",
                "datapoint_index": dp_idx,
                "scene_index": int(scene_index),
                "window_index": int(gen),
                "rollout_step": int(gen * chunk),
                "in_contact": bool(scooped),
                "reached_goal": bool(reached_goal),
                "scoop_failed": bool(dropped and not reached_goal),
                "retrying_scoop": bool(retrying_now),
                "scoop_attempt": int(scoop_attempts),
                "movement_token": "flip",
                "path_shape": "flip_scoop_air",
                "instruction": body["instruction"],
                "tool_label": body["tool_label"],
                "tool_contact_xyz_world": spatula.tolist(),
                "tool_current_normal": spat_normal.tolist(),
                "tool_current_surface_dir": spat_surface_dir.tolist(),
                "has_material": True,
                "material_label": body["material_label"],
                "material_xyz_world": obj.tolist(),
                "material_quat_world": obj_quat.tolist(),
                "material_size": MATERIAL_SIZE_M.tolist(),
                "material_xyz_after_world": new_obj.tolist(),
                "material_quat_after_world": new_quat.tolist(),
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

        # Resolve attempt state: a drop that did NOT reach the goal (slipped
        # mid-lift, or the flip landed right-side up) leaves the object on the
        # ground and ends the attempt, so the next generation re-approaches with a
        # freshly drawn outcome. A successful grab keeps the attempt active.
        if dropped:
            attempting = False
            pending_slip = False
            pending_flip_fail = False
            scoop_attempts += 1
            under_latched = False
        else:
            under_latched = bool(new_under)

        spatula = new_spatula[0:3].copy()
        spat_normal = new_spatula[3:6].copy()
        spat_surface_dir = new_spatula[6:9].copy()
        obj = new_obj
        obj_quat = new_quat
        scooped = new_scooped
        flip_theta = new_theta
        grasp = new_grasp

    return datapoints


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    cfg: FlipGenConfig,
) -> dict[str, Any]:
    shard_id = f"spatula_flip_reactive_{shard_idx:04d}"
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
        "generator": "spatula_flip_reactive_v1",
        "seed": int(seed),
        "num_datapoints": len(datapoints),
        "scenes_per_shard": int(cfg.scenes_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reactive closed-loop spatula-flip shards (dataset_0012)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0012_spatula_flip_reactive/shards",
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

    cfg = FlipGenConfig(scenes_per_shard=int(args.scenes_per_shard))
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
