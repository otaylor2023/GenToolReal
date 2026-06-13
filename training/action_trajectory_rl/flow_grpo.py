"""Stochastic flow sampling with per-step log-probs for GRPO / DDPO-style updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from training.action_trajectory.model import ActionTrajectoryModel


@dataclass
class FlowSampleRecord:
    x: torch.Tensor  # [B, 54]
    t: torch.Tensor  # [B]
    x_next: torch.Tensor  # [B, 54]
    logprob: torch.Tensor  # [B]


@dataclass
class FlowSampleResult:
    final: torch.Tensor  # [B, 54]
    steps: List[FlowSampleRecord]
    logprob_sum: torch.Tensor  # [B]


def _model_velocity(
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return model(
        instr_clip=batch_tensors["instr_clip"],
        tool_clip=batch_tensors["tool_clip"],
        material_clip=batch_tensors["material_clip"],
        destination_clip=batch_tensors["destination_clip"],
        table_clip=batch_tensors["table_clip"],
        tool_contact_xyz_norm=batch_tensors["tool_contact_xyz_norm"],
        tool_normal=batch_tensors["tool_normal"],
        tool_surface_dir=batch_tensors["tool_surface_dir"],
        material_xyz_norm=batch_tensors["material_xyz_norm"],
        destination_xyz_norm=batch_tensors["destination_xyz_norm"],
        table_xyz_norm=batch_tensors["table_xyz_norm"],
        xt=x,
        t=t,
        has_material=batch_tensors["has_material"],
        has_destination=batch_tensors["has_destination"],
    )


def gaussian_step_logprob(
    x: torch.Tensor,
    mean: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Log p(x_next | mean) under diag Gaussian with std=sigma, summed over dims."""
    var = float(sigma) ** 2
    log_2pi = torch.log(torch.tensor(2.0 * 3.14159265, device=x.device, dtype=x.dtype))
    return (
        -0.5
        * (
            ((x - mean) ** 2) / var
            + log_2pi
            + torch.log(torch.tensor(var, device=x.device, dtype=x.dtype))
        ).sum(dim=-1)
    )


def sde_step(
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    x: torch.Tensor,
    t_scalar: float,
    *,
    dt: float,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, FlowSampleRecord]:
    bsz = x.shape[0]
    t = torch.full((bsz,), float(t_scalar), device=x.device, dtype=torch.float32)
    v = _model_velocity(model, batch_tensors, x, t)
    mean = x + v * float(dt)
    if sigma > 0.0:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        x_next = mean + float(sigma) * noise
    else:
        x_next = mean
    lp = gaussian_step_logprob(x_next, mean, sigma)
    return x_next, FlowSampleRecord(
        x=x.detach().clone(),
        t=t.detach().clone(),
        x_next=x_next.detach().clone(),
        logprob=lp.detach(),
    )


def sample_with_logprobs(
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    *,
    steps: int,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> FlowSampleResult:
    """Sample trajectories with recorded step log-probs (trainable via recompute)."""
    device = batch_tensors["tool_contact_xyz_norm"].device
    bsz = batch_tensors["tool_contact_xyz_norm"].shape[0]
    dt = 1.0 / max(1, int(steps))
    x = torch.randn(
        bsz,
        ActionTrajectoryModel.ACTION_DIM,
        device=device,
        generator=generator,
    )
    records: List[FlowSampleRecord] = []
    for i in range(int(steps)):
        x, rec = sde_step(
            model,
            batch_tensors,
            x,
            float(i) / float(max(1, steps)),
            dt=dt,
            sigma=float(sigma),
            generator=generator,
        )
        records.append(rec)
    logprob_sum = torch.stack([r.logprob for r in records], dim=0).sum(dim=0)
    return FlowSampleResult(final=x, steps=records, logprob_sum=logprob_sum)


def recompute_logprob_sum(
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    sample: FlowSampleResult,
    *,
    sigma: float,
    dt: float,
) -> torch.Tensor:
    """Re-evaluate log-probs under `model` for a fixed denoising path."""
    total = torch.zeros(sample.final.shape[0], device=sample.final.device)
    for rec in sample.steps:
        v = _model_velocity(model, batch_tensors, rec.x, rec.t)
        mean = rec.x + v * float(dt)
        total = total + gaussian_step_logprob(rec.x_next, mean, sigma)
    return total


def grpo_policy_loss(
    *,
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(new_logprob - old_logprob.detach())
    adv = advantage.detach()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    return -torch.min(surr1, surr2).mean()


def kl_ref_loss(
    model: ActionTrajectoryModel,
    ref_model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    sample: FlowSampleResult,
) -> torch.Tensor:
    """Penalty when trainable velocities differ from frozen reference on the sampled path."""
    losses = []
    for rec in sample.steps:
        v = _model_velocity(model, batch_tensors, rec.x, rec.t)
        with torch.no_grad():
            v_ref = _model_velocity(ref_model, batch_tensors, rec.x, rec.t)
        losses.append(F.mse_loss(v, v_ref))
    return torch.stack(losses).mean()
