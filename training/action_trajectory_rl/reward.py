"""Combined sim reward: low-level tracking success + ball placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RewardConfig:
    w_track: float = 1.0
    w_ball: float = 1.0
    ball_success_radius_m: float = 0.06
    # Max ball term for out-of-region progress; kept < 1 so that landing the
    # ball anywhere inside the goal region (term = 1.0) is the unique maximum.
    ball_partial_scale: float = 0.8
    # Goal-region (blue patch) half-extents on the table. The ball counts as
    # "in the goal" (full ball reward) when its center lies inside this square
    # region (matching the visual sim marker), not merely near the center point.
    goal_region_half_x_m: float = 0.05
    goal_region_half_y_m: float = 0.05
    ball_radius_m: float = 0.02
    # Table top extents; if the ball ends beyond these (or drops below the
    # surface) it fell off and the ball term is set to -off_table_penalty.
    # ``table_x_min/max_m`` allow an asymmetric (e.g. +x-extended) table; when
    # left None they fall back to the symmetric +/- table_half_x_m.
    table_half_x_m: float = 0.2375
    table_half_y_m: float = 0.20
    table_x_min_m: Optional[float] = None
    table_x_max_m: Optional[float] = None
    table_y_min_m: Optional[float] = None
    table_y_max_m: Optional[float] = None
    off_table_drop_m: float = 0.08
    off_table_penalty: float = 1.0


def _ball_terms(
    ball_start_xyz: np.ndarray | torch.Tensor,
    ball_final_xyz: np.ndarray | torch.Tensor,
    destination_xyz: np.ndarray | torch.Tensor,
    cfg: RewardConfig,
):
    """Return (ball_term, d1, in_region) for the ball-placement reward.

    ``in_region`` is a box test against the goal-region half-extents (+ the ball
    radius) so the ball is "in the goal" when it overlaps the visual patch.
    A ball in the region gets the full ball term (1.0, the unique maximum);
    otherwise it earns partial credit (< 1) for closing the start->goal distance.
    """
    b0 = np.asarray(ball_start_xyz, dtype=np.float64).reshape(-1, 3)
    b1 = np.asarray(ball_final_xyz, dtype=np.float64).reshape(-1, 3)
    dest = np.asarray(destination_xyz, dtype=np.float64).reshape(-1, 3)

    d0 = np.linalg.norm(b0 - dest, axis=1)
    d1 = np.linalg.norm(b1 - dest, axis=1)
    progress = np.clip((d0 - d1) / np.maximum(d0, 1e-6), 0.0, 1.0)

    dxy = np.abs(b1[:, :2] - dest[:, :2])
    hx = float(cfg.goal_region_half_x_m) + float(cfg.ball_radius_m)
    hy = float(cfg.goal_region_half_y_m) + float(cfg.ball_radius_m)
    in_region = ((dxy[:, 0] <= hx) & (dxy[:, 1] <= hy)).astype(np.float64)

    partial = progress * float(cfg.ball_partial_scale)
    ball_term = np.where(in_region > 0.0, 1.0, partial)

    # Ball fell off the table: beyond the table footprint in xy, or dropped well
    # below the table surface (dest z sits at the table top). Override to a
    # negative term and clear the in-region flag.
    x_min = float(cfg.table_x_min_m) if cfg.table_x_min_m is not None else -float(cfg.table_half_x_m)
    x_max = float(cfg.table_x_max_m) if cfg.table_x_max_m is not None else float(cfg.table_half_x_m)
    y_min = float(cfg.table_y_min_m) if cfg.table_y_min_m is not None else -float(cfg.table_half_y_m)
    y_max = float(cfg.table_y_max_m) if cfg.table_y_max_m is not None else float(cfg.table_half_y_m)
    off_xy = (
        (b1[:, 0] < x_min) | (b1[:, 0] > x_max) | (b1[:, 1] < y_min) | (b1[:, 1] > y_max)
    )
    off_z = b1[:, 2] < (dest[:, 2] - float(cfg.off_table_drop_m))
    fell_off = off_xy | off_z
    ball_term = np.where(fell_off, -float(cfg.off_table_penalty), ball_term)
    in_region = np.where(fell_off, 0.0, in_region)
    return ball_term, d1, in_region


def compute_combined_reward(
    *,
    tracking_frac: np.ndarray | torch.Tensor,
    ball_start_xyz: np.ndarray | torch.Tensor,
    ball_final_xyz: np.ndarray | torch.Tensor,
    destination_xyz: np.ndarray | torch.Tensor,
    cfg: RewardConfig,
) -> np.ndarray:
    """Return per-sample reward in [0, ~2] range (higher is better).

    tracking_frac: successes / max_consecutive_successes per env, in [0, 1].
    ball_* and destination_*: world/scene-frame xyz, shape [N, 3].
    """
    track = np.asarray(tracking_frac, dtype=np.float64).reshape(-1)
    ball_term, _, _ = _ball_terms(
        ball_start_xyz, ball_final_xyz, destination_xyz, cfg
    )
    return (
        float(cfg.w_track) * track + float(cfg.w_ball) * ball_term
    ).astype(np.float32)


def reward_breakdown(
    *,
    tracking_frac: np.ndarray | torch.Tensor,
    ball_start_xyz: np.ndarray | torch.Tensor,
    ball_final_xyz: np.ndarray | torch.Tensor,
    destination_xyz: np.ndarray | torch.Tensor,
    cfg: RewardConfig,
) -> dict:
    """Per-sample reward components (already weighted) for logging/captions."""
    track = np.asarray(tracking_frac, dtype=np.float64).reshape(-1)
    ball_term, d1, in_region = _ball_terms(
        ball_start_xyz, ball_final_xyz, destination_xyz, cfg
    )

    track_w = (float(cfg.w_track) * track).astype(np.float32)
    ball_w = (float(cfg.w_ball) * ball_term).astype(np.float32)
    return {
        "track": track_w,
        "ball": ball_w,
        "total": (track_w + ball_w).astype(np.float32),
        "ball_dist_final_m": d1.astype(np.float32),
        "ball_in_region": in_region.astype(np.float32),
    }


@dataclass
class FlipRewardConfig:
    """Reward for the spatula flip: object up-axis inverted AND settled on table."""

    w_flip: float = 1.0
    # Up-axis dot world-z below this counts as "inverted" (180deg ~ -1.0).
    inverted_dot_max: float = -0.5
    # Object center must rest within this band of the table top (m) to count as
    # "settled" (not still lifted in the air, not fallen through).
    settle_tol_m: float = 0.04
    material_half_z_m: float = 0.006
    # Partial credit (< full flip) scaled by how inverted the object is, even if
    # it did not fully settle, so GRPO gets a gradient toward the flip.
    partial_scale: float = 0.5
    # Off-table failure -> negative term.
    table_x_min_m: float = -0.2375
    table_x_max_m: float = 0.2375
    table_y_min_m: float = -0.20
    table_y_max_m: float = 0.20
    off_table_drop_m: float = 0.10
    off_table_penalty: float = 1.0


def _up_dot_z_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """World-z component of the object's body +z axis, for xyzw quaternions."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
    qx, qy = q[:, 0], q[:, 1]
    # Third column z-entry of the rotation matrix: 1 - 2(qx^2 + qy^2).
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def _flip_terms(
    object_quat_final: np.ndarray,
    object_xyz_final: np.ndarray,
    *,
    table_z: np.ndarray | float,
    cfg: FlipRewardConfig,
):
    quat = np.asarray(object_quat_final, dtype=np.float64).reshape(-1, 4)
    xyz = np.asarray(object_xyz_final, dtype=np.float64).reshape(-1, 3)
    tz = np.asarray(table_z, dtype=np.float64).reshape(-1)
    if tz.shape[0] == 1:
        tz = np.full((xyz.shape[0],), float(tz[0]))

    up_dot = _up_dot_z_from_quat(quat)
    inverted = up_dot <= float(cfg.inverted_dot_max)

    rest_z = tz + float(cfg.material_half_z_m)
    settled = np.abs(xyz[:, 2] - rest_z) <= float(cfg.settle_tol_m)

    # How-inverted in [0, 1]: up=+1 -> 0, up=-1 -> 1.
    inv01 = np.clip((1.0 - up_dot) / 2.0, 0.0, 1.0)
    partial = inv01 * float(cfg.partial_scale)
    flip_term = np.where(inverted & settled, 1.0, partial)

    off_xy = (
        (xyz[:, 0] < float(cfg.table_x_min_m))
        | (xyz[:, 0] > float(cfg.table_x_max_m))
        | (xyz[:, 1] < float(cfg.table_y_min_m))
        | (xyz[:, 1] > float(cfg.table_y_max_m))
    )
    off_z = xyz[:, 2] < (tz - float(cfg.off_table_drop_m))
    fell_off = off_xy | off_z
    flip_term = np.where(fell_off, -float(cfg.off_table_penalty), flip_term)

    success = (inverted & settled & ~fell_off).astype(np.float64)
    return flip_term, up_dot, success


def compute_flip_reward(
    *,
    object_quat_final: np.ndarray,
    object_xyz_final: np.ndarray,
    table_z: np.ndarray | float,
    cfg: FlipRewardConfig,
) -> np.ndarray:
    flip_term, _, _ = _flip_terms(
        object_quat_final, object_xyz_final, table_z=table_z, cfg=cfg
    )
    return (float(cfg.w_flip) * flip_term).astype(np.float32)


def flip_reward_breakdown(
    *,
    object_quat_final: np.ndarray,
    object_xyz_final: np.ndarray,
    table_z: np.ndarray | float,
    cfg: FlipRewardConfig,
) -> dict:
    flip_term, up_dot, success = _flip_terms(
        object_quat_final, object_xyz_final, table_z=table_z, cfg=cfg
    )
    flip_w = (float(cfg.w_flip) * flip_term).astype(np.float32)
    return {
        "flip": flip_w,
        "total": flip_w,
        "up_dot_z": up_dot.astype(np.float32),
        "flip_success": success.astype(np.float32),
    }


@dataclass
class PourRewardConfig:
    """Reward for the spoon scoop-and-pour: material poured onto the goal plate.

    Full reward when the material's final position lands inside the goal region
    AND settles on the table (poured, not still carried in the bowl). Partial
    credit scales with how much start->goal distance was closed, giving GRPO a
    gradient even when the pour misses. Off-table -> negative term.
    """

    w_pour: float = 1.0
    goal_region_half_x_m: float = 0.05
    goal_region_half_y_m: float = 0.05
    material_radius_m: float = 0.02
    # Material must rest within this band of the table top to count as "poured".
    settle_tol_m: float = 0.03
    material_half_z_m: float = 0.009
    partial_scale: float = 0.8
    table_x_min_m: float = -0.2375
    table_x_max_m: float = 0.2375
    table_y_min_m: float = -0.20
    table_y_max_m: float = 0.20
    off_table_drop_m: float = 0.10
    off_table_penalty: float = 1.0


def _pour_terms(
    material_start_xyz: np.ndarray,
    material_final_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    *,
    table_z: np.ndarray | float,
    cfg: PourRewardConfig,
):
    b0 = np.asarray(material_start_xyz, dtype=np.float64).reshape(-1, 3)
    b1 = np.asarray(material_final_xyz, dtype=np.float64).reshape(-1, 3)
    dest = np.asarray(destination_xyz, dtype=np.float64).reshape(-1, 3)
    tz = np.asarray(table_z, dtype=np.float64).reshape(-1)
    if tz.shape[0] == 1:
        tz = np.full((b1.shape[0],), float(tz[0]))

    d0 = np.linalg.norm(b0[:, :2] - dest[:, :2], axis=1)
    d1 = np.linalg.norm(b1[:, :2] - dest[:, :2], axis=1)
    progress = np.clip((d0 - d1) / np.maximum(d0, 1e-6), 0.0, 1.0)

    dxy = np.abs(b1[:, :2] - dest[:, :2])
    hx = float(cfg.goal_region_half_x_m) + float(cfg.material_radius_m)
    hy = float(cfg.goal_region_half_y_m) + float(cfg.material_radius_m)
    in_region = (dxy[:, 0] <= hx) & (dxy[:, 1] <= hy)

    rest_z = tz + float(cfg.material_half_z_m)
    settled = np.abs(b1[:, 2] - rest_z) <= float(cfg.settle_tol_m)

    partial = progress * float(cfg.partial_scale)
    pour_term = np.where(in_region & settled, 1.0, partial)

    off_xy = (
        (b1[:, 0] < float(cfg.table_x_min_m))
        | (b1[:, 0] > float(cfg.table_x_max_m))
        | (b1[:, 1] < float(cfg.table_y_min_m))
        | (b1[:, 1] > float(cfg.table_y_max_m))
    )
    off_z = b1[:, 2] < (tz - float(cfg.off_table_drop_m))
    fell_off = off_xy | off_z
    pour_term = np.where(fell_off, -float(cfg.off_table_penalty), pour_term)

    success = (in_region & settled & ~fell_off).astype(np.float64)
    return pour_term, d1, success


def compute_pour_reward(
    *,
    material_start_xyz: np.ndarray,
    material_final_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    table_z: np.ndarray | float,
    cfg: PourRewardConfig,
) -> np.ndarray:
    pour_term, _, _ = _pour_terms(
        material_start_xyz, material_final_xyz, destination_xyz, table_z=table_z, cfg=cfg
    )
    return (float(cfg.w_pour) * pour_term).astype(np.float32)


def pour_reward_breakdown(
    *,
    material_start_xyz: np.ndarray,
    material_final_xyz: np.ndarray,
    destination_xyz: np.ndarray,
    table_z: np.ndarray | float,
    cfg: PourRewardConfig,
) -> dict:
    pour_term, d1, success = _pour_terms(
        material_start_xyz, material_final_xyz, destination_xyz, table_z=table_z, cfg=cfg
    )
    pour_w = (float(cfg.w_pour) * pour_term).astype(np.float32)
    return {
        "pour": pour_w,
        "total": pour_w,
        "material_dist_final_m": d1.astype(np.float32),
        "pour_success": success.astype(np.float32),
    }


@dataclass
class HammerRewardConfig:
    """Reward for hammering a nail head down to its target sink depth."""

    w_hammer: float = 1.0
    target_tol_m: float = 0.004
    partial_scale: float = 0.8


def _hammer_terms(
    *,
    head_start_xyz: np.ndarray,
    head_final_xyz: np.ndarray,
    target_xyz: np.ndarray,
    cfg: HammerRewardConfig,
):
    h0 = np.asarray(head_start_xyz, dtype=np.float64).reshape(-1, 3)
    h1 = np.asarray(head_final_xyz, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(-1, 3)
    z0 = h0[:, 2]
    z1 = h1[:, 2]
    zt = target[:, 2]
    denom = np.maximum(z0 - zt, 1e-6)
    sink_frac = np.clip((z0 - z1) / denom, 0.0, 1.0)
    success = z1 <= (zt + float(cfg.target_tol_m))
    partial = sink_frac * float(cfg.partial_scale)
    hammer_term = np.where(success, 1.0, partial)
    return hammer_term, sink_frac, success.astype(np.float64)


def compute_hammer_reward(
    *,
    head_start_xyz: np.ndarray,
    head_final_xyz: np.ndarray,
    target_xyz: np.ndarray,
    cfg: HammerRewardConfig,
) -> np.ndarray:
    hammer_term, _, _ = _hammer_terms(
        head_start_xyz=head_start_xyz,
        head_final_xyz=head_final_xyz,
        target_xyz=target_xyz,
        cfg=cfg,
    )
    return (float(cfg.w_hammer) * hammer_term).astype(np.float32)


def hammer_reward_breakdown(
    *,
    head_start_xyz: np.ndarray,
    head_final_xyz: np.ndarray,
    target_xyz: np.ndarray,
    cfg: HammerRewardConfig,
) -> dict:
    hammer_term, sink_frac, success = _hammer_terms(
        head_start_xyz=head_start_xyz,
        head_final_xyz=head_final_xyz,
        target_xyz=target_xyz,
        cfg=cfg,
    )
    hammer_w = (float(cfg.w_hammer) * hammer_term).astype(np.float32)
    return {
        "hammer": hammer_w,
        "total": hammer_w,
        "sink_frac": sink_frac.astype(np.float32),
        "hammer_success": success.astype(np.float32),
    }


def group_relative_advantages(
    rewards: np.ndarray,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """GRPO-style advantages: (r - mean_group) / (std_group + eps)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = r.shape[0]
    assert n % int(group_size) == 0, f"rewards len {n} not divisible by group_size {group_size}"
    g = int(group_size)
    groups = r.reshape(-1, g)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1).astype(np.float32)
