"""Flow loss and geometric constraint checks for action-expert pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RegionConstraintConfig:
    # Horizontal tolerance for `exact_above` (meters), e.g. 0.01 ≈ 1 cm disk around ref in XY.
    exact_above_xy_radius_m: float = 0.01
    z_min_offset_m: float = -0.02
    z_max_offset_m: float = 0.20
    above_xy_min_m: float = 0.025
    above_xy_max_m: float = 0.05
    near_xy_min_m: float = 0.075
    near_xy_max_m: float = 0.10
    directional_axis_min_m: float = 0.075
    directional_axis_max_m: float = 0.10
    directional_lateral_tol_m: float = 0.03
    between_midpoint_fraction_of_span: float = 0.40
    eps: float = 1e-6
    # Triangular wedge for left/right/in_front/behind/above/below/over (world axes): start
    # offset from ref along primitive axis, depth along axis, full width at far end (m).
    wedge_start_offset_m: float = 0.01
    wedge_depth_m: float = 0.08
    wedge_width_m: float = 0.10
    # Multiplier on z_min/z_max band for left/right/in_front/behind (halved vs legacy).
    directional_z_scale: float = 0.5
    # For above/below/over wedges in XZ, max |ΔY| from ref still allowed (m).
    wedge_planar_thickness_m: float = 0.02


def _point_in_triangle_2d_torch(
    p: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Barycentric inside test; p,v0,v1,v2 are (2,) same device/dtype."""
    v0b = v1 - v0
    v1b = v2 - v0
    v2b = p - v0
    d00 = (v0b * v0b).sum()
    d01 = (v0b * v1b).sum()
    d11 = (v1b * v1b).sum()
    d20 = (v2b * v0b).sum()
    d21 = (v2b * v1b).sum()
    denom = d00 * d11 - d01 * d01
    denom = denom.clamp(min=float(eps))
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= -eps) & (v >= -eps) & (w >= -eps)


def _directional_lateral_z_band_ok(z_off: torch.Tensor, cfg: RegionConstraintConfig) -> torch.Tensor:
    z_lo = float(cfg.z_min_offset_m) * float(cfg.directional_z_scale)
    z_hi = float(cfg.z_max_offset_m) * float(cfg.directional_z_scale)
    return _band_ok(z_off, z_lo, z_hi)


def _wedge_triangle_xy_world_offsets(
    token: str,
    *,
    cfg: RegionConstraintConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CCW triangle in (ΔX, ΔY) for lateral primitives (left/right/in_front/behind)."""
    s = float(cfg.wedge_start_offset_m)
    far = s + float(cfg.wedge_depth_m)
    h = 0.5 * float(cfg.wedge_width_m)
    zt = torch.zeros((), device=device, dtype=dtype)
    t = str(token).strip().lower()
    if t == "left":
        v0 = torch.stack((torch.tensor(-s, device=device, dtype=dtype), zt))
        v1 = torch.stack(
            (torch.tensor(-far, device=device, dtype=dtype), torch.tensor(-h, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(-far, device=device, dtype=dtype), torch.tensor(h, device=device, dtype=dtype))
        )
    elif t == "right":
        v0 = torch.stack((torch.tensor(s, device=device, dtype=dtype), zt))
        v1 = torch.stack(
            (torch.tensor(far, device=device, dtype=dtype), torch.tensor(-h, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(far, device=device, dtype=dtype), torch.tensor(h, device=device, dtype=dtype))
        )
    elif t == "in_front":
        v0 = torch.stack((zt, torch.tensor(-s, device=device, dtype=dtype)))
        v1 = torch.stack(
            (torch.tensor(-h, device=device, dtype=dtype), torch.tensor(-far, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(h, device=device, dtype=dtype), torch.tensor(-far, device=device, dtype=dtype))
        )
    else:
        v0 = torch.stack((zt, torch.tensor(s, device=device, dtype=dtype)))
        v1 = torch.stack(
            (torch.tensor(-h, device=device, dtype=dtype), torch.tensor(far, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(h, device=device, dtype=dtype), torch.tensor(far, device=device, dtype=dtype))
        )
    return v0, v1, v2


def _wedge_triangle_xz_world_offsets(
    token: str,
    *,
    cfg: RegionConstraintConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triangle in (ΔX, ΔZ) for above / below / over."""
    s = float(cfg.wedge_start_offset_m)
    far = s + float(cfg.wedge_depth_m)
    h = 0.5 * float(cfg.wedge_width_m)
    zt = torch.zeros((), device=device, dtype=dtype)
    t = str(token).strip().lower()
    if t == "below":
        v0 = torch.stack((zt, torch.tensor(-s, device=device, dtype=dtype)))
        v1 = torch.stack(
            (torch.tensor(-h, device=device, dtype=dtype), torch.tensor(-far, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(h, device=device, dtype=dtype), torch.tensor(-far, device=device, dtype=dtype))
        )
    else:
        v0 = torch.stack((zt, torch.tensor(s, device=device, dtype=dtype)))
        v1 = torch.stack(
            (torch.tensor(-h, device=device, dtype=dtype), torch.tensor(far, device=device, dtype=dtype))
        )
        v2 = torch.stack(
            (torch.tensor(h, device=device, dtype=dtype), torch.tensor(far, device=device, dtype=dtype))
        )
    return v0, v1, v2


def flow_matching_loss(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_velocity, target_velocity)


def _band_ok(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return (value >= low) & (value <= high)


def movement_token_eval_bucket(movement_token: str, constraint_type: str) -> str:
    """Bucket a sample for eval breakdown (order matches `satisfies_constraint` branches)."""
    token = str(movement_token or "").strip().lower()
    ctype = str(constraint_type or "").strip().lower()
    if token == "exact_above" or "exact_above" in ctype:
        return "exact_above"
    if token in {"over", "above"} or "above_v0" in ctype:
        return "above"
    if token in {"near", "next_to"}:
        return "near"
    if token in {"left", "right", "in_front", "behind"} or "directional" in ctype:
        if token in {"left", "right", "in_front", "behind"}:
            return f"directional_{token}"
        return "directional_generic"
    if token == "between" or "between" in ctype:
        return "between"
    return "other"


def _goal_region_xy_z_radius(
    movement_token: str,
    constraint_type: str,
    constraint_params: Dict[str, Any] | None,
    cfg: RegionConstraintConfig,
) -> Tuple[float, float]:
    token = str(movement_token or "").strip().lower()
    ctype = str(constraint_type or "").strip().lower()
    cp = dict(constraint_params or {})

    if token == "exact_above" or "exact_above" in ctype:
        xy_r = float(cp.get("xy_radius_m", cfg.exact_above_xy_radius_m))
    elif token in {"over", "above"} or "above_v0" in ctype:
        far = float(cfg.wedge_start_offset_m) + float(cfg.wedge_depth_m)
        hw = 0.5 * float(cfg.wedge_width_m)
        xy_r = float(max(far, hw, 1e-4))
    elif token == "below":
        far = float(cfg.wedge_start_offset_m) + float(cfg.wedge_depth_m)
        hw = 0.5 * float(cfg.wedge_width_m)
        xy_r = float(max(far, hw, 1e-4))
    elif token in {"left", "right", "in_front", "behind"} or "directional" in ctype:
        far = float(cfg.wedge_start_offset_m) + float(cfg.wedge_depth_m)
        hw = 0.5 * float(cfg.wedge_width_m)
        xy_r = float(max(far, hw, 1e-4))
    elif token in {"near", "next_to"}:
        xy_r = float(cfg.near_xy_max_m)
    elif token == "between" or "between" in ctype:
        xy_r = float(max(cfg.near_xy_max_m, cfg.above_xy_max_m))
    else:
        xy_r = float(max(cfg.exact_above_xy_radius_m, 0.03))

    z_min_cp = cp.get("z_min_offset_m")
    z_max_cp = cp.get("z_max_offset_m")
    if isinstance(z_min_cp, (int, float)) and isinstance(z_max_cp, (int, float)):
        z_r = float(max(abs(float(z_min_cp)), abs(float(z_max_cp))))
    elif token in {"left", "right", "in_front", "behind"} or "directional" in ctype:
        z_r = float(
            max(
                abs(float(cfg.z_min_offset_m) * float(cfg.directional_z_scale)),
                abs(float(cfg.z_max_offset_m) * float(cfg.directional_z_scale)),
                1e-4,
            )
        )
    elif token in {"over", "above", "below"} or "above_v0" in ctype:
        far = float(cfg.wedge_start_offset_m) + float(cfg.wedge_depth_m)
        z_r = float(max(far, float(cfg.wedge_planar_thickness_m), 1e-4))
    else:
        z_r = float(max(abs(cfg.z_min_offset_m), abs(cfg.z_max_offset_m)))

    return max(xy_r, 1e-4), max(z_r, 1e-4)


def goal_region_contains(
    pred_xyz: torch.Tensor,
    goal_xyz: torch.Tensor,
    movement_token: str,
    constraint_type: str,
    constraint_params: Dict[str, Any] | None,
    cfg: RegionConstraintConfig,
) -> bool:
    pred = pred_xyz.float()
    goal = goal_xyz.float()
    xy_r, z_r = _goal_region_xy_z_radius(movement_token, constraint_type, constraint_params, cfg)
    d_xy = float(torch.linalg.norm((pred[:2] - goal[:2])).item())
    d_z = abs(float((pred[2] - goal[2]).item()))
    return d_xy <= float(xy_r) and d_z <= float(z_r)


def satisfies_constraint(
    pred_xyz: torch.Tensor,
    movement_token: str,
    constraint_type: str,
    reference_xyz: torch.Tensor,
    secondary_reference_xyz: torch.Tensor,
    has_secondary_reference: bool,
    constraint_params: Dict[str, Any] | None,
    cfg: RegionConstraintConfig,
) -> bool:
    del constraint_params
    token = str(movement_token or "").strip().lower()
    ctype = str(constraint_type or "").strip().lower()
    pred = pred_xyz.float()
    ref = reference_xyz.float()
    ref2 = secondary_reference_xyz.float()

    if float(ref.abs().sum().item()) <= cfg.eps:
        return True

    z_off = pred[2] - ref[2]
    z_ok = bool(_band_ok(z_off, cfg.z_min_offset_m, cfg.z_max_offset_m).item())
    xy = pred[:2] - ref[:2]
    r_xy = float(torch.linalg.norm(xy).item())

    if token == "exact_above" or "exact_above" in ctype:
        return r_xy <= float(cfg.exact_above_xy_radius_m) and z_ok
    if token in {"over", "above"} or "above_v0" in ctype:
        dtok = token if token in {"over", "above"} else "above"
        d = pred - ref
        if float(torch.abs(d[1]).item()) > float(cfg.wedge_planar_thickness_m):
            return False
        v0, v1, v2 = _wedge_triangle_xz_world_offsets(
            dtok, cfg=cfg, device=pred.device, dtype=pred.dtype
        )
        p2 = torch.stack((d[0], d[2]))
        return bool(_point_in_triangle_2d_torch(p2, v0, v1, v2).item())
    if token == "below":
        d = pred - ref
        if float(torch.abs(d[1]).item()) > float(cfg.wedge_planar_thickness_m):
            return False
        v0, v1, v2 = _wedge_triangle_xz_world_offsets(
            "below", cfg=cfg, device=pred.device, dtype=pred.dtype
        )
        p2 = torch.stack((d[0], d[2]))
        return bool(_point_in_triangle_2d_torch(p2, v0, v1, v2).item())
    if token in {"near", "next_to"}:
        return (cfg.near_xy_min_m <= r_xy <= cfg.near_xy_max_m) and z_ok
    if token in {"left", "right", "in_front", "behind"} or "directional" in ctype:
        dtok = token if token in {"left", "right", "in_front", "behind"} else "behind"
        d = pred - ref
        if not bool(_directional_lateral_z_band_ok(d[2], cfg).item()):
            return False
        v0, v1, v2 = _wedge_triangle_xy_world_offsets(
            dtok, cfg=cfg, device=pred.device, dtype=pred.dtype
        )
        p2 = torch.stack((d[0], d[1]))
        return bool(_point_in_triangle_2d_torch(p2, v0, v1, v2).item())
    if token == "between" or "between" in ctype:
        if not has_secondary_reference:
            return True
        midpoint = 0.5 * (ref + ref2)
        span_xy = torch.linalg.norm(ref[:2] - ref2[:2]).clamp(min=cfg.eps)
        max_mid_dist = cfg.between_midpoint_fraction_of_span * span_xy
        mid_xy_dist = torch.linalg.norm(pred[:2] - midpoint[:2])
        z_mid_off = pred[2] - midpoint[2]
        return bool((mid_xy_dist <= max_mid_dist).item()) and bool(
            _band_ok(z_mid_off, cfg.z_min_offset_m, cfg.z_max_offset_m).item()
        )
    return True


def deterministic_goal_sample_seed(
    scene_id: str,
    shard_path: str,
    datapoint_index: int,
    instruction_variant_index: int,
    *,
    algorithm_version: str = "goal_rs_v2",
) -> int:
    """Stable 63-bit seed from dataset identity (same across runs / workers)."""
    payload = "|".join(
        [
            str(algorithm_version),
            str(scene_id),
            str(shard_path),
            str(int(datapoint_index)),
            str(int(instruction_variant_index)),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # Avoid 0; numpy accepts large positive int.
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 2) + 1


def _proposal_bounds_world(
    ref: np.ndarray,
    ref2: np.ndarray | None,
    has_secondary: bool,
    cfg: RegionConstraintConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64).reshape(3)
    pts = [ref]
    if has_secondary and ref2 is not None:
        r2 = np.asarray(ref2, dtype=np.float64).reshape(3)
        if float(np.abs(r2).sum()) > float(cfg.eps):
            pts.append(r2)
    stacked = np.stack(pts, axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span_xy = float(np.linalg.norm(pts[0][:2] - pts[-1][:2])) if len(pts) >= 2 else 0.0
    extra_xy = float(cfg.between_midpoint_fraction_of_span) * span_xy + 0.05
    pad_xy = (
        max(
            float(cfg.near_xy_max_m),
            float(cfg.above_xy_max_m),
            float(cfg.directional_axis_max_m),
            float(cfg.directional_lateral_tol_m),
            float(cfg.exact_above_xy_radius_m),
            float(cfg.wedge_start_offset_m) + float(cfg.wedge_depth_m),
            0.5 * float(cfg.wedge_width_m),
            extra_xy,
            0.12,
        )
        + 0.08
    )
    pad_z = max(abs(float(cfg.z_min_offset_m)), abs(float(cfg.z_max_offset_m)), 0.08) + 0.08
    lo = lo - np.array([pad_xy, pad_xy, pad_z], dtype=np.float64)
    hi = hi + np.array([pad_xy, pad_xy, pad_z], dtype=np.float64)
    return lo, hi


def sample_goal_xyz_world_rejection(
    *,
    goal_xyz_world: np.ndarray,
    movement_token: str,
    constraint_type: str,
    constraint_params: Dict[str, Any] | None,
    reference_xyz_world: np.ndarray | None,
    secondary_reference_xyz_world: np.ndarray | None,
    has_secondary_reference: bool,
    cfg: RegionConstraintConfig,
    rng: np.random.Generator,
    max_attempts: int = 512,
) -> Tuple[np.ndarray, bool]:
    """Sample a target inside a region centered on the original goal pose.

    Notes:
    - Region width is token-aware (`_goal_region_xy_z_radius`) and deterministic with `rng`.
    - For backwards compatibility we keep the function name/signature.
    """
    del reference_xyz_world, secondary_reference_xyz_world, has_secondary_reference, max_attempts
    goal = np.asarray(goal_xyz_world, dtype=np.float64).reshape(3)
    xy_r, z_r = _goal_region_xy_z_radius(movement_token, constraint_type, constraint_params, cfg)

    theta = float(rng.uniform(0.0, 2.0 * float(np.pi)))
    r = float(np.sqrt(rng.uniform(0.0, 1.0))) * float(xy_r)
    x = float(goal[0]) + r * float(np.cos(theta))
    y = float(goal[1]) + r * float(np.sin(theta))
    z = float(goal[2]) + float(rng.uniform(-float(z_r), float(z_r)))
    out = np.array([x, y, z], dtype=np.float32)
    ok = goal_region_contains(
        pred_xyz=torch.from_numpy(out),
        goal_xyz=torch.from_numpy(goal.astype(np.float32)),
        movement_token=movement_token,
        constraint_type=constraint_type,
        constraint_params=constraint_params,
        cfg=cfg,
    )
    return out, bool(ok)

