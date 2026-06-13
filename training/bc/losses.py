"""Loss functions for behavior cloning training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F


@dataclass
class RegionLossConfig:
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


def canonical_mse_loss(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_xyz, target_xyz)


def _band_penalty(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    low_t = value.new_tensor(low)
    high_t = value.new_tensor(high)
    under = torch.relu(low_t - value)
    over = torch.relu(value - high_t)
    return under + over


def _z_band_penalty(pred: torch.Tensor, ref: torch.Tensor, cfg: RegionLossConfig) -> torch.Tensor:
    z_off = pred[2] - ref[2]
    return _band_penalty(z_off, cfg.z_min_offset_m, cfg.z_max_offset_m)


def region_aware_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    movement_tokens: Sequence[str],
    constraint_types: Sequence[str],
    reference_xyz: torch.Tensor,
    secondary_reference_xyz: torch.Tensor,
    has_secondary_reference: Sequence[bool],
    constraint_params: Sequence[Dict[str, Any]],
    cfg: RegionLossConfig,
) -> torch.Tensor:
    losses = []
    for i in range(pred_xyz.shape[0]):
        pred = pred_xyz[i]
        target = target_xyz[i]
        token = str(movement_tokens[i] or "").strip().lower()
        constraint_type = str(constraint_types[i] or "").strip().lower()
        ref = reference_xyz[i]
        ref2 = secondary_reference_xyz[i]

        # Fallback to canonical target if no usable reference exists.
        if float(ref.abs().sum().detach().cpu().item()) <= cfg.eps:
            losses.append(F.mse_loss(pred, target))
            continue

        xy = pred[:2] - ref[:2]
        r_xy = torch.linalg.norm(xy)
        z_penalty = _z_band_penalty(pred, ref, cfg)
        p = pred.new_tensor(0.0)

        if token == "exact_above" or "exact_above" in constraint_type:
            p = r_xy + z_penalty
        elif token in {"over", "above"}:
            p = _band_penalty(r_xy, cfg.above_xy_min_m, cfg.above_xy_max_m) + z_penalty
        elif token in {"near", "next_to"}:
            p = _band_penalty(r_xy, cfg.near_xy_min_m, cfg.near_xy_max_m) + z_penalty
        elif token in {"left", "right", "in_front", "behind"} or "directional" in constraint_type:
            if token == "left":
                signed_axis = ref[0] - pred[0]
                lateral = pred[1] - ref[1]
            elif token == "right":
                signed_axis = pred[0] - ref[0]
                lateral = pred[1] - ref[1]
            elif token == "in_front":
                signed_axis = ref[1] - pred[1]
                lateral = pred[0] - ref[0]
            else:  # behind
                signed_axis = pred[1] - ref[1]
                lateral = pred[0] - ref[0]
            p = (
                _band_penalty(
                    signed_axis,
                    cfg.directional_axis_min_m,
                    cfg.directional_axis_max_m,
                )
                + torch.relu(torch.abs(lateral) - cfg.directional_lateral_tol_m)
                + z_penalty
            )
        elif token == "between" or "between" in constraint_type:
            if bool(has_secondary_reference[i]):
                midpoint = 0.5 * (ref + ref2)
                span_xy = torch.linalg.norm(ref[:2] - ref2[:2]).clamp(min=cfg.eps)
                max_mid_dist = cfg.between_midpoint_fraction_of_span * span_xy
                mid_xy_dist = torch.linalg.norm(pred[:2] - midpoint[:2])
                z_penalty = _z_band_penalty(pred, midpoint, cfg)
                p = torch.relu(mid_xy_dist - max_mid_dist) + z_penalty
            else:
                p = F.mse_loss(pred, target)
        else:
            # Unknown token falls back to canonical supervision.
            p = F.mse_loss(pred, target)
        losses.append(p)
    return torch.stack(losses).mean()

