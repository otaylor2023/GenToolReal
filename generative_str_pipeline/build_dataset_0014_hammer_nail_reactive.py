"""Procedurally generate reactive closed-loop hammer-a-nail shards (dataset_0014).

Each scene rolls out a receding-horizon hammering motion: the hammer is staged
above a nail that protrudes from a board, then repeatedly strikes the nail head
straight down. Every solid strike drives the head down a step; the hammer
retracts and strikes again until the head has sunk to a target depth below its
starting height (not necessarily flush), then lifts clear and finishes.

The tool's working face strikes downward onto the nail head, so the contact
frame uses ``normal = +z`` (the nail-head top surface, hammer body above) with a
horizontal ``surface_dir`` for the handle -- the same convention as the brush at
rest. There is no in-plane motion; the stroke is purely vertical.

Waypoints are contact-frame rows ``[contact_xyz(3), normal(3), surface_dir(3)]``.
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

DATASET_ID = "dataset_0014_hammer_nail_reactive"
TOOL_LABEL = "the hammer"

MATERIALS_HAMMER = (
    "the nail",
    "the nail head",
    "the spike",
    "the pin",
    "the tack",
    "the brad",
    "the peg",
    "the fastener",
)
DESTINATIONS_HAMMER = (
    "the board",
    "the plank",
    "the wood",
    "the beam",
    "the workpiece",
    "the timber",
    "the block of wood",
)
TEMPLATES_HAMMER = (
    "Hammer {material} into {destination} with {tool}.",
    "Use {tool} to drive {material} down into {destination}.",
    "Strike {material} with {tool} until it sinks into {destination}.",
    "Pound {material} into {destination} using {tool}.",
    "Drive {material} flush into {destination} with {tool}.",
    "Tap {material} down into {destination} with {tool} until it seats.",
    "Use {tool} to sink {material} into {destination}.",
    "Hammer down on {material} with {tool} to drive it into {destination}.",
    "Knock {material} into {destination} with {tool}.",
    "Seat {material} into {destination} by striking it with {tool}.",
    "With {tool}, hit {material} repeatedly to drive it into {destination}.",
    "Drive {material} into {destination}, striking the head with {tool}.",
    "Use {tool} to hammer {material} down until it is buried in {destination}.",
    "Bring {tool} down on {material} to sink it into {destination}.",
)

# Nail head modeled as a short box (length x, width y, height z) so it reuses the
# generic material-box rendering / sim asset.
NAIL_HEAD_SIZE_M = np.array([0.016, 0.016, 0.004], dtype=np.float32)
NAIL_HEAD_HALF_M = 0.5 * NAIL_HEAD_SIZE_M
# Nail shaft (for visualization only): a thin cylinder under the head.
NAIL_SHAFT_RADIUS_M = 0.0025
# Board the nail is driven into (length x, width y, thickness z).
BOARD_SIZE_M = np.array([0.18, 0.14, 0.03], dtype=np.float32)


@dataclass
class HammerNailGenConfig(SweepGenConfig):
    """Reactive closed-loop hammer-a-nail generator."""

    dense_step_spacing_m: float = 0.02
    num_output_waypoints: int = 15
    max_generations: int = 40
    executed_chunk: int = 5
    scenes_per_shard: int = 150

    # Nail geometry / goal.
    board_size_m: tuple[float, float, float] = (
        float(BOARD_SIZE_M[0]),
        float(BOARD_SIZE_M[1]),
        float(BOARD_SIZE_M[2]),
    )
    nail_protrusion_start_m_range: tuple[float, float] = (0.045, 0.065)
    # Target depth below the *starting* head height to drive the nail (a specific
    # sink distance, not necessarily flush with the board).
    sink_target_m_range: tuple[float, float] = (0.020, 0.035)
    # Place the board/nail within a modest box around the table center.
    nail_xy_extent_m: float = 0.08

    # Tool starts resting on the table at an offset, then is picked up and
    # carried over the nail before the hitting motion (like the brush/spatula).
    tool_home_offset_m_range: tuple[float, float] = (0.12, 0.24)
    # Transit height (above the table) the hammer is carried at between the home
    # rest and the nail.
    transit_height_m: float = 0.18
    # Horizontal tolerance for a strike to actually land (impact must be at the
    # nail xy).
    over_nail_tol_m: float = 0.025
    # Looser horizontal tolerance for deciding the hammer is in striking range
    # (the cocked head sits handle_length*(1-cos(swing)) behind the nail, so this
    # must comfortably exceed that backswing offset across the jitter ranges).
    ready_tol_m: float = 0.16

    # Per-scene randomization ranges. Each scene samples a distinct-but-plausible
    # motion style and layout (in addition to the table / nail / home jitter), so
    # the simulator sees varied attempts and keeps the ones that succeed.
    swing_angle_rad_range: tuple[float, float] = (0.80, 1.10)
    handle_length_m_range: tuple[float, float] = (0.17, 0.22)
    transit_height_m_range: tuple[float, float] = (0.15, 0.22)
    strike_apex_above_head_m_range: tuple[float, float] = (0.05, 0.085)
    hit_depth_step_m_range: tuple[float, float] = (0.008, 0.013)
    # Board center offset from the nail (nail need not sit dead-center).
    board_center_jitter_m: float = 0.03

    # Striking dynamics.
    hit_depth_step_m: float = 0.010
    hit_depth_jitter_m: float = 0.003
    strike_apex_above_head_m: float = 0.06
    approach_above_head_m: float = 0.08
    # Swing: the hammer rotates about the grip (handle end) so the HEAD sweeps a
    # wide arc up and back on the backswing, then down through impact. swing_angle
    # is the backswing angle; handle_length is the head-to-grip lever arm (the
    # arc radius); grip_lift raises the pivot above the nail height.
    swing_angle_rad: float = 0.95
    handle_length_m: float = 0.20
    grip_lift_m: float = 0.02
    # Resting roll: the hammer lies on its side on the table (90deg roll about the
    # handle axis from the upright strike pose).
    home_roll_rad: float = float(np.pi / 2.0)
    # How far past the head top the face presses on contact (so the strike
    # registers and looks like it seats on the head).
    contact_press_m: float = 0.002
    # A weak/glancing strike that barely drives the nail, forcing another hit
    # (reactive retry, analogous to the flip's slip). The final allowed hits are
    # forced solid so scenes always terminate.
    weak_hit_prob: float = 0.22
    weak_hit_scale: float = 0.2
    max_hits: int = 14
    # Margins for detecting a strike (face reaches head top) and a clean retract.
    strike_tol_m: float = 0.006
    retract_margin_m: float = 0.02
    # Head must be lifted at least this far above the head top to count as
    # "released" / done after the target depth is reached.
    lift_done_margin_m: float = 0.05


def _unit_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation of ``angle`` rad about ``axis``."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.eye(3, dtype=np.float64)
    a = a / n
    c, s = float(np.cos(angle)), float(np.sin(angle))
    K = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64
    )
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _hammer_frame(
    handle_dir_xy: np.ndarray, roll: float, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Contact frame for the hammer parameterized by two angles, relative to the
    upright strike pose (face down, normal ``-z``, handle along ``handle_dir``):

    * ``roll``  -- rotation about the handle axis; ``pi/2`` lays the hammer on its
      side (resting on the table), ``0`` is upright ready-to-strike.
    * ``phi``   -- swing pitch about the horizontal axis perpendicular to the
      handle; ``0`` is impact (face straight down), positive cocks the head up
      and back for the backswing.
    """
    h = _unit_xy(handle_dir_xy).astype(np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Pitch about the horizontal axis perpendicular to the handle, signed so that
    # positive phi raises the head (and the face) up and back for the backswing.
    swing_axis = np.cross(up, h)
    R = _rot_axis_angle(swing_axis, float(phi)) @ _rot_axis_angle(h, float(roll))
    normal = (R @ (-up)).astype(np.float32)
    surface_dir = (R @ h).astype(np.float32)
    return normal, surface_dir


def _swing_contact(
    nail_xy: tuple[float, float],
    strike_z: float,
    handle_dir: np.ndarray,
    phi: float,
    cfg: "HammerNailGenConfig",
) -> np.ndarray:
    """Head (contact) position at swing angle ``phi``, rotating rigidly about the
    grip (handle end). At ``phi=0`` the head is at the nail (impact); positive phi
    sweeps the head up and back along an arc of radius ``handle_length`` so the
    head swings while the grip stays put."""
    h = _unit_xy(handle_dir).astype(np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    strike = np.array([float(nail_xy[0]), float(nail_xy[1]), float(strike_z)], dtype=np.float64)
    pivot = strike + h * float(cfg.handle_length_m) + up * float(cfg.grip_lift_m)
    R = _rot_axis_angle(np.cross(up, h), float(phi))
    contact = pivot + R @ (strike - pivot)
    return contact.astype(np.float32)


def _densify_swing(
    keyframes: list[tuple[np.ndarray, float, float]],
    *,
    step_spacing_m: float,
) -> list[tuple[np.ndarray, float, float]]:
    """Densify (contact, roll, phi) keyframes, interpolating contact by distance
    and the two angles linearly along the same parameter."""
    dense: list[tuple[np.ndarray, float, float]] = []
    for i in range(len(keyframes) - 1):
        c0, r0, p0 = keyframes[i]
        c1, r1, p1 = keyframes[i + 1]
        a = np.asarray(c0, dtype=np.float64).reshape(3)
        b = np.asarray(c1, dtype=np.float64).reshape(3)
        seg_len = float(np.linalg.norm(b - a))
        n_steps = max(1, int(np.round(seg_len / max(step_spacing_m, 1e-4))))
        for j in range(n_steps):
            t = float(j) / float(n_steps)
            c = ((1.0 - t) * a + t * b).astype(np.float32)
            dense.append((c, (1.0 - t) * r0 + t * r1, (1.0 - t) * p0 + t * p1))
    cN, rN, pN = keyframes[-1]
    dense.append((np.asarray(cN, dtype=np.float32).reshape(3), float(rN), float(pN)))
    return dense


def _rows_to_waypoints(
    rows: list[tuple[np.ndarray, float, float]],
    handle_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the [N,9] waypoint array and a parallel [N,2] (roll, phi) array from
    densified (contact, roll, phi) rows."""
    n = len(rows)
    wp = np.zeros((n, 9), dtype=np.float32)
    rp = np.zeros((n, 2), dtype=np.float32)
    for i, (c, roll, phi) in enumerate(rows):
        normal, surface_dir = _hammer_frame(handle_dir, roll, phi)
        wp[i, 0:3] = c
        wp[i, 3:6] = normal
        wp[i, 6:9] = surface_dir
        rp[i, 0] = float(roll)
        rp[i, 1] = float(phi)
    return wp, rp


def _pad_rp(rp: np.ndarray, num_steps: int) -> np.ndarray:
    r = np.asarray(rp, dtype=np.float32).reshape(-1, 2)
    if r.shape[0] >= num_steps:
        return r[:num_steps].copy()
    out = np.zeros((num_steps, 2), dtype=np.float32)
    out[: r.shape[0]] = r
    for i in range(r.shape[0], num_steps):
        out[i] = r[-1]
    return out


def _pad_target_waypoints(target: np.ndarray, num_steps: int) -> np.ndarray:
    t = np.asarray(target, dtype=np.float32).reshape(-1, 9)
    if t.shape[0] >= num_steps:
        return t[:num_steps].copy()
    out = np.zeros((num_steps, 9), dtype=np.float32)
    out[: t.shape[0]] = t
    for i in range(t.shape[0], num_steps):
        out[i] = t[-1]
    return out


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _sample_scene_geometry(
    rng: np.random.Generator,
    cfg: HammerNailGenConfig,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    board_bz = float(cfg.board_size_m[2])
    board_top_z = table_z + board_bz

    ext = float(cfg.nail_xy_extent_m)
    nail_x = float(rng.uniform(-ext, ext))
    nail_y = float(rng.uniform(-ext, ext))

    protrusion = float(rng.uniform(*cfg.nail_protrusion_start_m_range))
    head_half_z = float(NAIL_HEAD_HALF_M[2])
    # Head center starts a full protrusion above the board top.
    head_center_z0 = board_top_z + protrusion
    sink_target = float(rng.uniform(*cfg.sink_target_m_range))
    target_head_z = head_center_z0 - sink_target

    handle_yaw = float(rng.uniform(-np.pi, np.pi))
    handle_dir = np.array([np.cos(handle_yaw), np.sin(handle_yaw), 0.0], dtype=np.float32)

    # The board carries the nail but the nail need not be dead-center on it.
    bj = float(cfg.board_center_jitter_m)
    board_cx = nail_x + float(rng.uniform(-bj, bj))
    board_cy = nail_y + float(rng.uniform(-bj, bj))

    return {
        "table_z": table_z,
        "board_top_z": board_top_z,
        "board_center": np.array([board_cx, board_cy, table_z], dtype=np.float32),
        "nail_xy": np.array([nail_x, nail_y], dtype=np.float32),
        "head_half_z": head_half_z,
        "head_center_z0": head_center_z0,
        "target_head_z": target_head_z,
        "protrusion": protrusion,
        "sink_target": sink_target,
        "handle_dir": handle_dir,
    }


def _sample_tool_home_hammer(
    rng: np.random.Generator,
    cfg: HammerNailGenConfig,
    *,
    nail_xy: np.ndarray,
    table_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rest the hammer on the table at an offset from the nail (toward the robot
    +x side), and orient the handle pointing from the nail back toward home."""
    nail = np.asarray(nail_xy, dtype=np.float64).reshape(2)
    offset = float(rng.uniform(*cfg.tool_home_offset_m_range))
    # Bias the staging toward the reachable +x/front half.
    ang = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
    direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
    home_xy = cfg.clip_xy((nail + offset * direction).astype(np.float32))
    home_z = table_z + float(rng.uniform(*cfg.tool_home_z_above_table_m_range))
    home = np.array([float(home_xy[0]), float(home_xy[1]), home_z], dtype=np.float32)

    handle_vec = np.array(home_xy, dtype=np.float64) - nail
    if float(np.linalg.norm(handle_vec)) < 1e-6:
        handle_vec = np.array([1.0, 0.0], dtype=np.float64)
    handle_dir = np.array([handle_vec[0], handle_vec[1], 0.0], dtype=np.float32)
    return home, _unit_xy(handle_dir)


def plan_next_hammer_steps(
    cfg: HammerNailGenConfig,
    *,
    hammer_contact: np.ndarray,
    hammer_roll: float,
    hammer_phi: float,
    nail_head_xyz: np.ndarray,
    target_head_z: float,
    head_half_z: float,
    handle_dir: np.ndarray,
    transit_z: float,
    num_steps: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan the next dense hammer waypoints (with orientation) from the current
    state. Returns ``([N,9] waypoints, [N,2] (roll, phi))``.

    Three phases, selected reactively from where the hammer currently is:
      * approach: not yet over the nail -> roll upright off the side while lifting,
        carry over the nail at the transit height, and cock back ready to swing;
      * swing: lined up over the nail and target not reached -> swing the head
        down through impact, then recover to the cocked backswing;
      * done: target depth reached -> lift the hammer clear and finish.
    """
    contact = np.asarray(hammer_contact, dtype=np.float32).reshape(3)
    head = np.asarray(nail_head_xyz, dtype=np.float32).reshape(3)
    nx, ny = float(head[0]), float(head[1])
    head_top_z = float(head[2]) + float(head_half_z)

    apex_z = head_top_z + float(cfg.strike_apex_above_head_m)
    strike_z = head_top_z - float(cfg.contact_press_m)
    transit_z = max(float(transit_z), apex_z + 0.02)
    done = float(head[2]) <= float(target_head_z) + 1e-4
    dist_xy = float(np.linalg.norm(contact[:2] - np.array([nx, ny], dtype=np.float32)))
    over_nail = dist_xy < float(cfg.ready_tol_m)

    swing = float(cfg.swing_angle_rad)
    strike = np.array([nx, ny, strike_z], dtype=np.float32)
    # Cocked backswing apex and a mid-arc sample, both on the grip-pivot arc so
    # the head sweeps a real arc (the grip stays put, the head swings).
    cock = _swing_contact((nx, ny), strike_z, handle_dir, swing, cfg)
    mid = _swing_contact((nx, ny), strike_z, handle_dir, 0.5 * swing, cfg)

    cur = (contact, float(hammer_roll), float(hammer_phi))

    if done:
        # Target depth reached: lift the hammer clear (upright) and finish.
        lift = np.array(
            [nx, ny, head_top_z + float(cfg.lift_done_margin_m) + 0.04],
            dtype=np.float32,
        )
        keyframes = [cur, (lift, 0.0, 0.0)]
    elif not over_nail:
        # Pick up and carry: roll upright off the side as it lifts, carry over the
        # nail at the transit height, then settle into the cocked backswing.
        up = np.array([contact[0], contact[1], transit_z], dtype=np.float32)
        over = np.array([float(cock[0]), float(cock[1]), transit_z], dtype=np.float32)
        keyframes = [cur, (up, 0.0, 0.0), (over, 0.0, swing), (cock, 0.0, swing)]
    else:
        # Swing strike: drive the head down through impact FIRST (so the hit lands
        # within the executed chunk), then recover up the arc to the backswing.
        keyframes = [
            cur,
            (strike, 0.0, 0.0),
            (mid, 0.0, 0.5 * swing),
            (cock, 0.0, swing),
        ]

    dense = _densify_swing(keyframes, step_spacing_m=float(cfg.dense_step_spacing_m))
    wp, rp = _rows_to_waypoints(dense, handle_dir)
    if wp.shape[0] < num_steps:
        wp = np.concatenate(
            [wp, np.tile(wp[-1][None, :], (num_steps - wp.shape[0], 1))], axis=0
        )
        rp = np.concatenate(
            [rp, np.tile(rp[-1][None, :], (num_steps - rp.shape[0], 1))], axis=0
        )
    return wp[:num_steps], rp[:num_steps]


def _execute_chunk(
    cfg: HammerNailGenConfig,
    plan: np.ndarray,
    rp: np.ndarray,
    *,
    head_xyz: np.ndarray,
    target_head_z: float,
    head_half_z: float,
    hits_done: int,
    struck_this_descent: bool,
    chunk: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, int, bool, bool, bool]:
    """Execute first ``chunk`` planned steps; drive the nail head on solid strikes.

    The nail head only moves straight down. A strike is registered when the
    hammer face descends to the head top (and has not already struck in this
    descent); a solid strike drives the head down ``hit_depth_step`` (with
    jitter), a weak one barely moves it. The descent latch resets once the
    hammer retracts back above the head, so each down-stroke can land one hit.
    """
    head = np.asarray(head_xyz, dtype=np.float32).reshape(3).copy()
    n_exec = max(1, min(int(chunk), int(plan.shape[0])))

    xyz_trace: list[np.ndarray] = []
    quat_trace: list[np.ndarray] = []
    made_hit = False

    for i in range(n_exec):
        contact = plan[i, 0:3].astype(np.float32)
        head_top_z = float(head[2]) + float(head_half_z)
        already_at_target = float(head[2]) <= float(target_head_z) + 1e-4

        dist_xy = float(
            np.linalg.norm(contact[:2] - head[:2].astype(np.float32))
        )
        over_nail = dist_xy < float(cfg.over_nail_tol_m)
        low = float(contact[2]) <= head_top_z + float(cfg.strike_tol_m)
        high = float(contact[2]) >= head_top_z + float(cfg.retract_margin_m)
        if high:
            struck_this_descent = False

        # A strike only lands when the face descends onto the head AND the hammer
        # is lined up over the nail (so the table-rest pose can't false-trigger).
        if over_nail and low and not struck_this_descent and not already_at_target:
            force_solid = hits_done >= int(cfg.max_hits) - 1
            weak = (not force_solid) and (float(rng.random()) < float(cfg.weak_hit_prob))
            base = float(cfg.hit_depth_step_m) + float(
                rng.uniform(-cfg.hit_depth_jitter_m, cfg.hit_depth_jitter_m)
            )
            drive = max(0.0, base) * (float(cfg.weak_hit_scale) if weak else 1.0)
            new_head_z = max(float(target_head_z), float(head[2]) - drive)
            head[2] = float(new_head_z)
            struck_this_descent = True
            hits_done += 1
            made_hit = True

        xyz_trace.append(head.copy())
        quat_trace.append(_identity_quat())

    new_contact = plan[n_exec - 1].astype(np.float32)
    new_roll = float(rp[n_exec - 1, 0])
    new_phi = float(rp[n_exec - 1, 1])
    reached_target = float(head[2]) <= float(target_head_z) + 1e-4
    head_top_z = float(head[2]) + float(head_half_z)
    released = float(new_contact[2]) >= head_top_z + float(cfg.lift_done_margin_m)
    return (
        new_contact,
        new_roll,
        new_phi,
        head,
        np.stack(xyz_trace, axis=0).astype(np.float32),
        hits_done,
        struck_this_descent,
        made_hit,
        bool(reached_target and released),
    )


def _gen_hammer_instruction(
    rng: np.random.Generator,
    *,
    material_word: str,
    destination_word: str,
) -> dict[str, Any]:
    template = str(rng.choice(TEMPLATES_HAMMER))
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
    cfg: HammerNailGenConfig,
    *,
    shard_id: str,
    scene_index: int,
    base_datapoint_index: int,
) -> list[dict[str, Any]]:
    table_xyz = cfg.sample_table_xyz(rng)
    # Per-scene motion style: each scene gets a distinct swing/lever/heights/drive
    # so the dataset spans many plausible hammering executions.
    dp_cfg = replace(
        cfg,
        table_xyz_world=table_xyz,
        swing_angle_rad=float(rng.uniform(*cfg.swing_angle_rad_range)),
        handle_length_m=float(rng.uniform(*cfg.handle_length_m_range)),
        transit_height_m=float(rng.uniform(*cfg.transit_height_m_range)),
        strike_apex_above_head_m=float(rng.uniform(*cfg.strike_apex_above_head_m_range)),
        hit_depth_step_m=float(rng.uniform(*cfg.hit_depth_step_m_range)),
    )
    table_z = float(table_xyz[2])

    geo = _sample_scene_geometry(rng, dp_cfg)
    nail_xy = geo["nail_xy"]
    head_half_z = float(geo["head_half_z"])
    target_head_z = float(geo["target_head_z"])

    body = _gen_hammer_instruction(
        rng,
        material_word=str(rng.choice(MATERIALS_HAMMER)),
        destination_word=str(rng.choice(DESTINATIONS_HAMMER)),
    )

    # Nail head state (only z changes).
    head = np.array(
        [float(nail_xy[0]), float(nail_xy[1]), float(geo["head_center_z0"])],
        dtype=np.float32,
    )
    head_quat = _identity_quat()

    # Tool starts resting on the table at an offset, oriented so the handle
    # points from the nail back toward home. It is then picked up and carried.
    head_top_z0 = float(geo["head_center_z0"]) + head_half_z
    hammer, handle_dir = _sample_tool_home_hammer(
        rng, dp_cfg, nail_xy=nail_xy, table_z=table_z
    )
    transit_z = max(
        table_z + float(dp_cfg.transit_height_m),
        head_top_z0 + float(dp_cfg.strike_apex_above_head_m) + 0.03,
    )

    destination_xyz = np.array(
        [float(nail_xy[0]), float(nail_xy[1]), target_head_z], dtype=np.float32
    )

    chunk = max(1, int(cfg.executed_chunk))
    n_wp = int(cfg.num_output_waypoints)
    hits_done = 0
    struck_this_descent = False
    # Orientation state: the hammer starts lying on its side (rolled), face flat.
    hammer_roll = float(cfg.home_roll_rad)
    hammer_phi = 0.0

    datapoints: list[dict[str, Any]] = []
    for gen in range(int(cfg.max_generations)):
        target, target_rp = plan_next_hammer_steps(
            dp_cfg,
            hammer_contact=hammer,
            hammer_roll=hammer_roll,
            hammer_phi=hammer_phi,
            nail_head_xyz=head,
            target_head_z=target_head_z,
            head_half_z=head_half_z,
            handle_dir=handle_dir,
            transit_z=transit_z,
            num_steps=n_wp,
        )
        target = _pad_target_waypoints(target, num_steps=n_wp)
        target_rp = _pad_rp(target_rp, num_steps=n_wp)

        # The tool's current pose (what the model conditions on this step).
        cur_normal, cur_surface_dir = _hammer_frame(handle_dir, hammer_roll, hammer_phi)

        (
            new_hammer,
            new_roll,
            new_phi,
            new_head,
            head_trace,
            hits_done,
            struck_this_descent,
            made_hit,
            reached_goal,
        ) = _execute_chunk(
            dp_cfg,
            target,
            target_rp,
            head_xyz=head,
            target_head_z=target_head_z,
            head_half_z=head_half_z,
            hits_done=hits_done,
            struck_this_descent=struck_this_descent,
            chunk=chunk,
            rng=rng,
        )

        quat_trace = np.tile(head_quat[None, :], (head_trace.shape[0], 1)).astype(np.float32)
        sink_now = float(geo["head_center_z0"]) - float(new_head[2])

        dp_idx = int(base_datapoint_index + gen)
        datapoints.append(
            {
                "datapoint_id": f"{shard_id}_{dp_idx:06d}",
                "datapoint_index": dp_idx,
                "scene_index": int(scene_index),
                "window_index": int(gen),
                "rollout_step": int(gen * chunk),
                "in_contact": bool(made_hit),
                "reached_goal": bool(reached_goal),
                "hit_landed": bool(made_hit),
                "hits_done": int(hits_done),
                "nail_sink_m": float(sink_now),
                "movement_token": "hammer",
                "path_shape": "hammer_strike_cycle",
                "instruction": body["instruction"],
                "tool_label": body["tool_label"],
                "tool_contact_xyz_world": hammer.tolist(),
                "tool_current_normal": cur_normal.tolist(),
                "tool_current_surface_dir": cur_surface_dir.tolist(),
                "has_material": True,
                "material_label": body["material_label"],
                "material_xyz_world": head.tolist(),
                "material_quat_world": head_quat.tolist(),
                "material_size": NAIL_HEAD_SIZE_M.tolist(),
                "material_xyz_after_world": new_head.tolist(),
                "material_quat_after_world": head_quat.tolist(),
                "material_xyz_executed_world": head_trace.tolist(),
                "material_quat_executed_world": quat_trace.tolist(),
                "has_destination": True,
                "destination_label": body["destination_label"],
                "destination_xyz_world": destination_xyz.tolist(),
                "table_label": TABLE_LABEL,
                "table_xyz_world": list(table_xyz),
                # Hammer-task scene extras (for visualization; ignored by the
                # generic trajectory loader).
                "board_xyz_world": geo["board_center"].tolist(),
                "board_size": list(cfg.board_size_m),
                "nail_head_size": NAIL_HEAD_SIZE_M.tolist(),
                "nail_shaft_radius": float(NAIL_SHAFT_RADIUS_M),
                "nail_target_z": float(target_head_z),
                "waypoints": target.reshape(n_wp, 9).tolist(),
            }
        )

        if reached_goal:
            break

        hammer = new_hammer[0:3].copy()
        hammer_roll = float(new_roll)
        hammer_phi = float(new_phi)
        head = new_head

    return datapoints


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    cfg: HammerNailGenConfig,
) -> dict[str, Any]:
    shard_id = f"hammer_nail_reactive_{shard_idx:04d}"
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
        "generator": "hammer_nail_reactive_v1",
        "seed": int(seed),
        "num_datapoints": len(datapoints),
        "scenes_per_shard": int(cfg.scenes_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reactive closed-loop hammer-a-nail shards (dataset_0014)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0014_hammer_nail_reactive/shards",
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

    cfg = HammerNailGenConfig(scenes_per_shard=int(args.scenes_per_shard))
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
