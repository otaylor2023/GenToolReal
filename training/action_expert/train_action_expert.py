"""Flow-matching trainer for PaliGemma + action-expert pipeline."""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.config import ActionExpertConfig, load_config
from training.action_expert.dataset import (
    ActionExpertDataset,
    action_expert_collate,
    load_action_samples_from_shards,
    split_shards_85_10_5,
)
from training.action_expert.losses import (
    RegionConstraintConfig,
    goal_region_contains,
    movement_token_eval_bucket,
)
from training.action_expert.model import ActionExpertModel
from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.vlm import PaliGemmaContextEncoder
from training.action_expert.xyz_normalization import denormalize_xyz_torch, load_xyz_normalization_stats


def _prepare_run_dir(output_root: Path, run_tag: str = "") -> Tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in output_root.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        suffix = p.name.split("_", 1)[1]
        numeric = suffix.split("_", 1)[0]
        if numeric.isdigit():
            max_idx = max(max_idx, int(numeric))
    next_idx = max_idx + 1
    run_id = f"run_{next_idx:04d}"
    clean_tag = str(run_tag).strip().replace(" ", "_")
    if clean_tag:
        run_id = f"{run_id}_{clean_tag}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _wandb_session(cfg: ActionExpertConfig, *, run_id: str, out_dir: Path):
    if not bool(cfg.use_wandb):
        yield
        return
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "use_wandb is true but wandb is not installed. Install with: pip install wandb"
        ) from exc
    init_kw: Dict[str, Any] = {
        "project": str(cfg.wandb_project),
        "name": str(cfg.wandb_run_name).strip() or run_id,
        "config": {**asdict(cfg), "run_id": run_id, "output_dir": str(out_dir)},
        "dir": str(out_dir),
    }
    entity = str(cfg.wandb_entity).strip()
    if entity:
        init_kw["entity"] = entity
    group = str(cfg.wandb_group).strip()
    if group:
        init_kw["group"] = group
    tags = [str(t) for t in cfg.wandb_tags if str(t).strip()]
    if tags:
        init_kw["tags"] = tags
    notes = str(cfg.wandb_notes).strip()
    if notes:
        init_kw["notes"] = notes
    wandb.init(**init_kw)
    try:
        yield
    finally:
        wandb.finish()


def _region_cfg(cfg: ActionExpertConfig) -> RegionConstraintConfig:
    return RegionConstraintConfig(
        exact_above_xy_radius_m=float(cfg.exact_above_xy_radius_m),
        z_min_offset_m=float(cfg.z_min_offset_m),
        z_max_offset_m=float(cfg.z_max_offset_m),
        above_xy_min_m=float(cfg.above_xy_min_m),
        above_xy_max_m=float(cfg.above_xy_max_m),
        near_xy_min_m=float(cfg.near_xy_min_m),
        near_xy_max_m=float(cfg.near_xy_max_m),
        directional_axis_min_m=float(cfg.directional_axis_min_m),
        directional_axis_max_m=float(cfg.directional_axis_max_m),
        directional_lateral_tol_m=float(cfg.directional_lateral_tol_m),
        between_midpoint_fraction_of_span=float(cfg.between_midpoint_fraction_of_span),
    )


def _batch_to_device(
    batch: Dict[str, Any], device: torch.device, cfg: ActionExpertConfig
) -> Dict[str, Any]:
    goals_norm = batch["goal_xyz_norm"].to(device=device, dtype=torch.float32)
    goals_world = batch["goal_xyz_world"].to(device=device, dtype=torch.float32)
    dataset_goals_world = batch["dataset_goal_xyz_world"].to(device=device, dtype=torch.float32)
    ref = batch["reference_xyz_world"].to(device=device, dtype=torch.float32)
    ref2 = batch["secondary_reference_xyz_world"].to(device=device, dtype=torch.float32)
    has_ref2 = [bool(x) for x in batch["has_secondary_reference"]]
    keypoint_positions = [p.to(device=device, dtype=torch.float32) for p in batch["keypoint_positions"]]
    system_prompts = [str(cfg.system_prompt)] * len(batch["instruction_text"])
    return {
        "image": batch["image"].to(device=device, dtype=torch.uint8),
        "scene_id": list(batch["scene_id"]),
        "shard_path": list(batch["shard_path"]),
        "datapoint_index": [int(x) for x in batch["datapoint_index"]],
        "goal_sample_group_index": [int(x) for x in batch["goal_sample_group_index"]],
        "instruction_variant_index": [int(x) for x in batch["instruction_variant_index"]],
        "system_prompt": system_prompts,
        "instruction_text": list(batch["instruction_text"]),
        "object_labels": list(batch["object_labels"]),
        "keypoint_labels": list(batch["keypoint_labels"]),
        "keypoint_positions": keypoint_positions,
        "goal_xyz_norm": goals_norm,
        "goal_xyz_world": goals_world,
        "dataset_goal_xyz_world": dataset_goals_world,
        "sampled_goal_in_region": [bool(x) for x in batch["sampled_goal_in_region"]],
        "movement_token": list(batch["movement_token"]),
        "constraint_type": list(batch["constraint_type"]),
        "constraint_params": list(batch["constraint_params"]),
        "reference_xyz_world": ref,
        "secondary_reference_xyz_world": ref2,
        "has_secondary_reference": has_ref2,
    }


def _grad_norm_tensors(params: List[torch.Tensor]) -> float:
    """L2 norm of gradients for a parameter list (0.0 if no grads)."""
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.norm(2).item() ** 2)
    return float(total_sq**0.5)


def _cosine_lr_multiplier(step_idx: int, total_steps: int, warmup_steps: int) -> float:
    total = max(1, int(total_steps))
    warmup = max(0, int(warmup_steps))
    s = max(0, int(step_idx))
    if warmup > 0 and s < warmup:
        return float((s + 1) / warmup)
    if total <= warmup:
        return 1.0
    progress = float(s - warmup) / float(max(1, total - warmup))
    progress = min(max(progress, 0.0), 1.0)
    return float(0.5 * (1.0 + np.cos(np.pi * progress)))


def _effective_label_proj_lr(cfg: ActionExpertConfig) -> float:
    v = getattr(cfg, "label_proj_lr", None)
    return float(v) if v is not None else float(cfg.action_expert_lr)


def _cosine_anneal_to_eta_multiplier(
    step_idx: int, *, T_max: int, max_lr: float, eta_min: float
) -> float:
    """Piecewise cosine from max_lr -> eta_min over T_max optimizer steps, then flat at eta_min."""
    T = max(1, int(T_max))
    s = max(0, int(step_idx))
    mx = float(max_lr)
    mn = float(eta_min)
    if mx <= 0.0:
        return 1.0
    if s >= T:
        return float(mn / mx)
    # Match torch.optim.lr_scheduler.CosineAnnealingLR at integer epochs 0..T-1, then clamp flat.
    cos = float(
        mn
        + (mx - mn)
        * (1.0 + np.cos(np.pi * float(s) / float(T)))
        / 2.0
    )
    return float(min(1.0, cos / mx))


def _log_bad_batch_samples(
    *,
    global_step: int,
    loss_f: float,
    batch: Dict[str, Any],
    flow: Dict[str, torch.Tensor],
    bad_batch_jsonl_path: Path | None = None,
) -> None:
    """Emit per-sample diagnostics for high-loss batches."""
    bsz = int(batch["goal_xyz_norm"].shape[0])
    goal_norm = batch["goal_xyz_norm"].detach().cpu()
    goal_world = batch["goal_xyz_world"].detach().cpu()
    dataset_goal_world = batch["dataset_goal_xyz_world"].detach().cpu()
    ref_world = batch["reference_xyz_world"].detach().cpu()
    ref2_world = batch["secondary_reference_xyz_world"].detach().cpu()
    x0 = flow["x0"].detach().cpu()
    x1 = flow["x1"].detach().cpu()
    xt = flow["xt"].detach().cpu()
    tv = flow["target_v"].detach().cpu()
    t = flow["t"].detach().cpu()
    print(f"[bad_batch] step={global_step} loss={loss_f:.6f} batch_size={bsz}")
    for i in range(bsz):
        t_val = float(t[i].reshape(-1)[0].item()) if t[i].numel() > 0 else float("nan")
        sample_id = (
            f"{batch['scene_id'][i]}|dp={batch['datapoint_index'][i]}"
            f"|iv={batch['instruction_variant_index'][i]}"
        )
        row = {
            "step": int(global_step),
            "loss": float(loss_f),
            "batch_pos": int(i),
            "sample_id": sample_id,
            "scene_id": str(batch["scene_id"][i]),
            "shard_path": str(batch["shard_path"][i]),
            "datapoint_index": int(batch["datapoint_index"][i]),
            "goal_sample_group_index": int(batch["goal_sample_group_index"][i]),
            "instruction_variant_index": int(batch["instruction_variant_index"][i]),
            "instruction": str(batch["instruction_text"][i]),
            "movement_token": str(batch["movement_token"][i]),
            "constraint_type": str(batch["constraint_type"][i]),
            "constraint_params": batch["constraint_params"][i],
            "sampled_goal_in_region": bool(batch["sampled_goal_in_region"][i]),
            "has_secondary_reference": bool(batch["has_secondary_reference"][i]),
            "keypoint_count": int(batch["keypoint_positions"][i].shape[0]),
            "goal_xyz_norm": goal_norm[i].tolist(),
            "goal_xyz_world": goal_world[i].tolist(),
            "dataset_goal_xyz_world": dataset_goal_world[i].tolist(),
            "reference_xyz_world": ref_world[i].tolist(),
            "secondary_reference_xyz_world": ref2_world[i].tolist(),
            "target_v_norm": float(tv[i].norm().item()),
            "x0_norm": float(x0[i].norm().item()),
            "x1_norm": float(x1[i].norm().item()),
            "xt_norm": float(xt[i].norm().item()),
            "t": t_val,
        }
        print(f"[bad_batch_sample] {json.dumps(row)}")
        if bad_batch_jsonl_path is not None:
            bad_batch_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with bad_batch_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")


def _fixed_eval_indices_by_token_and_variants(samples: List[Any]) -> List[int]:
    """Choose one goal-group per movement token, then keep up to 4 NL variants."""
    token_group_to_rows: Dict[str, Dict[Tuple[str, str, int], List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, s in enumerate(samples):
        tok = str(getattr(s, "movement_token", "")).strip()
        grp = (
            str(getattr(s, "scene_id", "")),
            str(getattr(s, "shard_path", "")),
            int(getattr(s, "goal_sample_group_index", getattr(s, "datapoint_index", 0))),
        )
        iv = int(getattr(s, "instruction_variant_index", 0))
        token_group_to_rows[tok][grp].append((i, iv))

    out: List[int] = []
    for tok in sorted(token_group_to_rows.keys()):
        groups = token_group_to_rows[tok]
        chosen = None
        # Prefer a group containing all 4 instruction variants.
        for grp_key in sorted(groups.keys()):
            rows = groups[grp_key]
            ivs = {iv for _, iv in rows}
            if len(ivs) >= 4:
                chosen = rows
                break
        if chosen is None:
            # Fallback: first group for this token.
            chosen = groups[sorted(groups.keys())[0]]
        chosen_sorted = sorted(chosen, key=lambda x: x[1])[:4]
        out.extend([idx for idx, _ in chosen_sorted])
    return out


def _fixed_eval_indices_uniform_movement_tokens(
    samples: List[Any], *, seed: int, target_n: int = 200, variants_per_group: int = 4
) -> List[int]:
    """Deterministic fixed-size subset sampled uniformly across tokens by goal-group.

    Target layout defaults to 50 groups x 4 NL variants = 200 rows.
    """
    if not samples:
        return []
    n_target = max(1, int(target_n))
    n_per_group = max(1, int(variants_per_group))
    target_groups = max(1, n_target // n_per_group)
    tok_to_groups: Dict[str, Dict[Tuple[str, str, int], List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, s in enumerate(samples):
        tok = str(getattr(s, "movement_token", "")).strip().lower()
        grp = (
            str(getattr(s, "scene_id", "")),
            str(getattr(s, "shard_path", "")),
            int(getattr(s, "goal_sample_group_index", getattr(s, "datapoint_index", 0))),
        )
        iv = int(getattr(s, "instruction_variant_index", 0))
        tok_to_groups[tok][grp].append((i, iv))
    if not tok_to_groups:
        return []
    rng = np.random.default_rng(int(seed))
    tokens = sorted(tok_to_groups.keys())
    token_group_rows: Dict[str, List[List[Tuple[int, int]]]] = {}
    for tok in tokens:
        rows_per_group = []
        for grp in sorted(tok_to_groups[tok].keys()):
            rows = sorted(tok_to_groups[tok][grp], key=lambda x: x[1])[:n_per_group]
            rows_per_group.append(rows)
        # Prefer complete 4-variant groups first.
        rows_per_group.sort(key=lambda rows: (len(rows) < n_per_group, len(rows)))
        rng.shuffle(rows_per_group)
        token_group_rows[tok] = rows_per_group
    base = target_groups // max(1, len(tokens))
    chosen_groups: List[List[Tuple[int, int]]] = []
    leftovers: List[List[Tuple[int, int]]] = []
    for tok in tokens:
        groups = token_group_rows[tok]
        take = min(base, len(groups))
        chosen_groups.extend(groups[:take])
        leftovers.extend(groups[take:])
    remaining_groups = target_groups - len(chosen_groups)
    if remaining_groups > 0 and leftovers:
        rng.shuffle(leftovers)
        chosen_groups.extend(leftovers[:remaining_groups])
    keep: List[int] = []
    for rows in chosen_groups:
        keep.extend([idx for idx, _ in rows[:n_per_group]])
    # If short due incomplete groups, top-up from any remaining rows deterministically.
    if len(keep) < n_target:
        pool: List[int] = []
        for rows in leftovers[remaining_groups:]:
            pool.extend([idx for idx, _ in rows])
        rng.shuffle(pool)
        need = n_target - len(keep)
        keep.extend(pool[:need])
    if len(keep) > n_target:
        keep = keep[:n_target]
    return sorted(set(keep))


def _wandb_log_split_metrics(metrics: Dict[str, Any], *, step: int, prefix: str) -> None:
    import wandb

    p = str(prefix).strip("/")
    flat: Dict[str, Any] = {
        f"{p}/mean_l2_error_m": metrics["val_mean_l2_error_m"],
        f"{p}/l2_std_error_m": metrics.get("val_l2_std_error_m", 0.0),
        f"{p}/success_rate": metrics["val_constraint_satisfaction_rate"],
    }
    if "val_nl_variance_mean_m" in metrics:
        flat[f"{p}/nl_variance/mean_m"] = metrics["val_nl_variance_mean_m"]
    if "val_nl_variance_std_m" in metrics:
        flat[f"{p}/nl_variance/std_m"] = metrics["val_nl_variance_std_m"]
    by_tok_success = metrics.get("val_success_by_movement_token") or {}
    if isinstance(by_tok_success, dict):
        for tok, stats in by_tok_success.items():
            if not isinstance(stats, dict):
                continue
            t = str(tok).replace("/", "_")
            if "success_rate" in stats:
                flat[f"{p}/movement_token/{t}/success_rate"] = stats["success_rate"]
    by_tok_l2 = metrics.get("val_l2_by_movement_token") or {}
    if isinstance(by_tok_l2, dict):
        for tok, stats in by_tok_l2.items():
            if not isinstance(stats, dict):
                continue
            t = str(tok).replace("/", "_")
            if "mean_l2_error_m" in stats:
                flat[f"{p}/movement_token/{t}/mean_l2_error_m"] = stats["mean_l2_error_m"]
            if "l2_std_error_m" in stats:
                flat[f"{p}/movement_token/{t}/l2_std_error_m"] = stats["l2_std_error_m"]
    by_tok_nl = metrics.get("val_nl_variance_by_movement_token") or {}
    if isinstance(by_tok_nl, dict):
        for tok, stats in by_tok_nl.items():
            if not isinstance(stats, dict):
                continue
            t = str(tok).replace("/", "_")
            if "mean_m" in stats:
                flat[f"{p}/nl_variance/{t}/mean_m"] = stats["mean_m"]
            if "std_m" in stats:
                flat[f"{p}/nl_variance/{t}/std_m"] = stats["std_m"]
    wandb.log(flat, step=int(step))


@torch.no_grad()
def _wandb_log_fixed_prediction_examples(
    *,
    step: int,
    cfg: ActionExpertConfig,
    region_cfg: RegionConstraintConfig,
    vlm: PaliGemmaContextEncoder,
    action_model: ActionExpertModel,
    loader: DataLoader[Dict[str, Any]],
    device: torch.device,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> None:
    import wandb

    vlm.eval()
    action_model.eval()
    table = wandb.Table(
        columns=[
            "sample_id",
            "movement_bucket",
            "movement_token",
            "instruction",
            "pred_x",
            "pred_y",
            "pred_z",
            "sampled_target_x",
            "sampled_target_y",
            "sampled_target_z",
            "original_goal_x",
            "original_goal_y",
            "original_goal_z",
            "l2_pred_to_sampled_m",
            "l2_pred_to_original_m",
            "pred_in_region_around_original_goal",
        ]
    )
    row_id = 0
    for raw in loader:
        batch = _batch_to_device(raw, device, cfg)
        ctx = vlm.forward_context(
            images_uint8=batch["image"],
            system_prompts=batch["system_prompt"],
            instructions=batch["instruction_text"],
            object_labels=batch["object_labels"],
        )
        label_emb = vlm.embed_labels(batch["keypoint_labels"])
        sample_positions = _rollout_prediction(
            action_model=action_model,
            context=ctx["context"],
            context_mask=ctx["attention_mask"],
            label_embeddings=label_emb,
            keypoint_positions=batch["keypoint_positions"],
            steps=int(cfg.integration_steps),
            n_samples=int(cfg.inference_samples),
        )
        for i in range(sample_positions.shape[0]):
            valid_rows: List[torch.Tensor] = []
            for s in range(sample_positions.shape[1]):
                p_world = denormalize_xyz_torch(sample_positions[i, s], xyz_mean, xyz_std, float(norm_eps))
                ok = goal_region_contains(
                    pred_xyz=p_world,
                    goal_xyz=batch["dataset_goal_xyz_world"][i],
                    movement_token=batch["movement_token"][i],
                    constraint_type=batch["constraint_type"][i],
                    constraint_params=batch["constraint_params"][i],
                    cfg=region_cfg,
                )
                if ok:
                    valid_rows.append(sample_positions[i, s])
            if valid_rows:
                pred_norm = torch.stack(valid_rows, dim=0).mean(dim=0)
            else:
                pred_norm = sample_positions[i].mean(dim=0)
            pred_world = denormalize_xyz_torch(pred_norm, xyz_mean, xyz_std, float(norm_eps))
            sampled_target = batch["goal_xyz_world"][i]
            dataset_goal = batch["dataset_goal_xyz_world"][i]
            l2_pred_to_sampled = float(torch.linalg.norm(pred_world - sampled_target).item())
            l2_pred_to_original = float(torch.linalg.norm(pred_world - dataset_goal).item())
            pred_in_region = bool(
                goal_region_contains(
                    pred_xyz=pred_world,
                    goal_xyz=dataset_goal,
                    movement_token=batch["movement_token"][i],
                    constraint_type=batch["constraint_type"][i],
                    constraint_params=batch["constraint_params"][i],
                    cfg=region_cfg,
                )
            )
            bucket = movement_token_eval_bucket(
                batch["movement_token"][i],
                batch["constraint_type"][i],
            )
            table.add_data(
                int(row_id),
                str(bucket),
                str(batch["movement_token"][i]),
                str(batch["instruction_text"][i])[:240],
                float(pred_world[0].item()),
                float(pred_world[1].item()),
                float(pred_world[2].item()),
                float(sampled_target[0].item()),
                float(sampled_target[1].item()),
                float(sampled_target[2].item()),
                float(dataset_goal[0].item()),
                float(dataset_goal[1].item()),
                float(dataset_goal[2].item()),
                l2_pred_to_sampled,
                l2_pred_to_original,
                pred_in_region,
            )
            row_id += 1
    wandb.log({"qualitative/prediction_table": table}, step=int(step))
    action_model.train()
    vlm.train()


def _sample_flow_inputs(goal_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
    bsz = goal_xyz.shape[0]
    x0 = torch.randn_like(goal_xyz)
    t = torch.rand(bsz, device=goal_xyz.device, dtype=torch.float32)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * goal_xyz
    target_v = goal_xyz - x0
    return {"x0": x0, "x1": goal_xyz, "t": t, "xt": xt, "target_v": target_v}


@torch.no_grad()
def _rollout_prediction(
    *,
    action_model: ActionExpertModel,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    label_embeddings: List[torch.Tensor],
    keypoint_positions: List[torch.Tensor],
    steps: int,
    n_samples: int,
) -> torch.Tensor:
    bsz = context.shape[0]
    dt = 1.0 / max(1, int(steps))
    finals: List[torch.Tensor] = []
    for _ in range(int(n_samples)):
        x = torch.randn(bsz, 3, device=context.device)
        for i in range(int(steps)):
            t = torch.full((bsz,), float(i) / float(max(1, steps)), device=context.device)
            v = action_model(
                label_embeddings=label_embeddings,
                keypoint_positions=keypoint_positions,
                xt=x,
                t=t,
                context=context,
                context_attention_mask=context_mask,
            )
            x = x + v * dt
        finals.append(x)
    return torch.stack(finals, dim=1)  # [B, S, 3]


@torch.no_grad()
def evaluate(
    *,
    cfg: ActionExpertConfig,
    region_cfg: RegionConstraintConfig,
    vlm: PaliGemmaContextEncoder,
    action_model: ActionExpertModel,
    loader: DataLoader[Dict[str, Any]],
    device: torch.device,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> Dict[str, Any]:
    vlm.eval()
    action_model.eval()
    all_l2: List[float] = []
    all_buckets: List[str] = []
    all_tokens: List[str] = []
    all_success: List[bool] = []
    all_group_keys: List[Tuple[str, str, int]] = []
    all_pred_world: List[np.ndarray] = []
    all_scene_ids: List[str] = []
    all_goal_group_ids: List[int] = []
    in_region_count = 0
    total_count = 0
    for raw in loader:
        batch = _batch_to_device(raw, device, cfg)
        ctx = vlm.forward_context(
            images_uint8=batch["image"],
            system_prompts=batch["system_prompt"],
            instructions=batch["instruction_text"],
            object_labels=batch["object_labels"],
        )
        label_emb = vlm.embed_labels(batch["keypoint_labels"])
        sample_positions = _rollout_prediction(
            action_model=action_model,
            context=ctx["context"],
            context_mask=ctx["attention_mask"],
            label_embeddings=label_emb,
            keypoint_positions=batch["keypoint_positions"],
            steps=int(cfg.integration_steps),
            n_samples=int(cfg.inference_samples),
        )  # [B, S, 3]

        preds: List[torch.Tensor] = []
        for i in range(sample_positions.shape[0]):
            valid_rows: List[torch.Tensor] = []
            for s in range(sample_positions.shape[1]):
                p = denormalize_xyz_torch(
                    sample_positions[i, s], xyz_mean, xyz_std, float(norm_eps)
                )
                ok = goal_region_contains(
                    pred_xyz=p,
                    goal_xyz=batch["dataset_goal_xyz_world"][i],
                    movement_token=batch["movement_token"][i],
                    constraint_type=batch["constraint_type"][i],
                    constraint_params=batch["constraint_params"][i],
                    cfg=region_cfg,
                )
                if ok:
                    valid_rows.append(sample_positions[i, s])
            if valid_rows:
                pred_n = torch.stack(valid_rows, dim=0).mean(dim=0)
                in_region_count += 1
                sample_success = True
            else:
                pred_n = sample_positions[i].mean(dim=0)
                sample_success = False
            pred_w = denormalize_xyz_torch(pred_n, xyz_mean, xyz_std, float(norm_eps))
            preds.append(pred_w)
            all_tokens.append(str(batch["movement_token"][i]).strip().lower())
            all_success.append(bool(sample_success))
            all_pred_world.append(pred_w.detach().cpu().numpy().astype(np.float64))
            all_scene_ids.append(str(batch["scene_id"][i]))
            all_goal_group_ids.append(int(batch["goal_sample_group_index"][i]))
            total_count += 1
        pred_xyz = torch.stack(preds, dim=0)
        goal_for_err = batch["goal_xyz_world"]
        err = torch.linalg.norm(pred_xyz - goal_for_err, dim=-1).cpu().tolist()
        for i, l2 in enumerate(err):
            all_l2.append(float(l2))
            all_buckets.append(
                movement_token_eval_bucket(batch["movement_token"][i], batch["constraint_type"][i])
            )
            all_group_keys.append(
                (
                    str(batch["scene_id"][i]),
                    str(batch["shard_path"][i]),
                    int(batch["goal_sample_group_index"][i]),
                )
            )

    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for l2, b in zip(all_l2, all_buckets):
        by_bucket[b].append(l2)

    val_by_movement: Dict[str, Dict[str, float | int]] = {}
    for b, errs in sorted(by_bucket.items()):
        arr = np.asarray(errs, dtype=np.float64)
        val_by_movement[b] = {
            "mean_l2_m": float(np.mean(arr)),
            "median_l2_m": float(np.median(arr)),
            "n": int(len(errs)),
        }
    success_by_token_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for tok, succ in zip(all_tokens, all_success):
        success_by_token_counts[tok][1] += 1
        if succ:
            success_by_token_counts[tok][0] += 1
    val_success_by_movement_token: Dict[str, Dict[str, float | int]] = {}
    for tok in sorted(success_by_token_counts.keys()):
        ok_n, n = success_by_token_counts[tok]
        val_success_by_movement_token[tok] = {
            "success_rate": float(ok_n / max(1, n)),
            "n": int(n),
            "num_success": int(ok_n),
        }
    val_l2_by_movement_token: Dict[str, Dict[str, float]] = {}
    tok_to_l2: Dict[str, List[float]] = defaultdict(list)
    for tok, l2 in zip(all_tokens, all_l2):
        tok_to_l2[str(tok)].append(float(l2))
    for tok in sorted(tok_to_l2.keys()):
        arr = np.asarray(tok_to_l2[tok], dtype=np.float64)
        val_l2_by_movement_token[tok] = {
            "mean_l2_error_m": float(np.mean(arr)) if arr.size else 0.0,
            "l2_std_error_m": float(np.std(arr)) if arr.size else 0.0,
        }
    group_to_errs: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    group_to_token: Dict[Tuple[str, str, int], str] = {}
    for gk, l2, tok in zip(all_group_keys, all_l2, all_tokens):
        group_to_errs[gk].append(float(l2))
        if gk not in group_to_token:
            group_to_token[gk] = str(tok)
    group_mean_l2: List[float] = []
    group_std_l2: List[float] = []
    by_tok_group_mean: Dict[str, List[float]] = defaultdict(list)
    by_tok_group_std: Dict[str, List[float]] = defaultdict(list)
    for gk, errs in group_to_errs.items():
        if not errs:
            continue
        arr = np.asarray(errs, dtype=np.float64)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        tok = group_to_token.get(gk, "")
        group_mean_l2.append(mu)
        group_std_l2.append(sigma)
        by_tok_group_mean[tok].append(mu)
        by_tok_group_std[tok].append(sigma)
    val_group_l2_by_movement_token: Dict[str, Dict[str, float | int]] = {}
    for tok in sorted(by_tok_group_mean.keys()):
        means = np.asarray(by_tok_group_mean[tok], dtype=np.float64)
        stds = np.asarray(by_tok_group_std[tok], dtype=np.float64)
        val_group_l2_by_movement_token[tok] = {
            "mean_l2_to_sample_m": float(np.mean(means)) if means.size else 0.0,
            "mean_group_std_l2_m": float(np.mean(stds)) if stds.size else 0.0,
            "n_groups": int(means.size),
        }
    nl_spreads_all: List[float] = []
    nl_spreads_by_token: Dict[str, List[float]] = defaultdict(list)
    nl_groups: Dict[Tuple[str, str, int], List[np.ndarray]] = defaultdict(list)
    for scene_id, tok, goal_group, pred in zip(
        all_scene_ids, all_tokens, all_goal_group_ids, all_pred_world
    ):
        nl_groups[(scene_id, tok, int(goal_group))].append(pred)
    for (scene_id, tok, goal_group), preds_xyz in nl_groups.items():
        del scene_id, goal_group
        if len(preds_xyz) != 4:
            continue
        arr = np.asarray(preds_xyz, dtype=np.float64).reshape(4, 3)
        centroid = arr.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(arr - centroid[None, :], axis=1)))
        nl_spreads_all.append(spread)
        nl_spreads_by_token[tok].append(spread)
    val_nl_variance_by_movement_token: Dict[str, Dict[str, float]] = {}
    for tok in sorted(nl_spreads_by_token.keys()):
        arr = np.asarray(nl_spreads_by_token[tok], dtype=np.float64)
        val_nl_variance_by_movement_token[tok] = {
            "mean_m": float(np.mean(arr)) if arr.size else 0.0,
            "std_m": float(np.std(arr)) if arr.size else 0.0,
        }

    sat_rate = float(in_region_count / max(1, total_count))
    action_model.train()
    vlm.train()
    return {
        "val_mean_l2_error_m": float(np.mean(all_l2)) if all_l2 else 0.0,
        "val_l2_std_error_m": float(np.std(np.asarray(all_l2, dtype=np.float64))) if all_l2 else 0.0,
        "val_median_l2_error_m": float(np.median(all_l2)) if all_l2 else 0.0,
        "val_in_region_rate": sat_rate,
        "val_constraint_satisfaction_rate": sat_rate,
        "val_by_movement": val_by_movement,
        "val_success_by_movement_token": val_success_by_movement_token,
        "val_l2_by_movement_token": val_l2_by_movement_token,
        "val_group_l2_mean_to_sample_m": float(np.mean(group_mean_l2)) if group_mean_l2 else 0.0,
        "val_group_l2_std_to_sample_m": float(np.mean(group_std_l2)) if group_std_l2 else 0.0,
        "val_group_count": int(len(group_mean_l2)),
        "val_group_l2_by_movement_token": val_group_l2_by_movement_token,
        "val_nl_variance_mean_m": float(np.mean(nl_spreads_all)) if nl_spreads_all else 0.0,
        "val_nl_variance_std_m": float(np.std(np.asarray(nl_spreads_all, dtype=np.float64)))
        if nl_spreads_all
        else 0.0,
        "val_nl_variance_by_movement_token": val_nl_variance_by_movement_token,
    }


def train(cfg: ActionExpertConfig) -> None:
    _seed_everything(int(cfg.seed))
    device = torch.device(cfg.device)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    out_dir, run_id = _prepare_run_dir(Path(cfg.output_dir), cfg.run_tag)
    print(f"[run] output_dir={out_dir}")

    region_cfg = _region_cfg(cfg)
    splits = split_shards_85_10_5(
        Path(cfg.dataset_dir),
        seed=int(cfg.seed),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
    )
    train_samples = load_action_samples_from_shards(
        splits["train"],
        max_keypoints=int(cfg.max_keypoints),
        region_cfg=region_cfg,
        explode_instruction_variants=bool(cfg.explode_instruction_variants),
    )
    val_samples = load_action_samples_from_shards(
        splits["val"],
        max_keypoints=int(cfg.max_keypoints),
        region_cfg=region_cfg,
        explode_instruction_variants=bool(cfg.explode_instruction_variants),
    )
    test_samples = load_action_samples_from_shards(
        splits["test"],
        max_keypoints=int(cfg.max_keypoints),
        region_cfg=region_cfg,
        explode_instruction_variants=bool(cfg.explode_instruction_variants),
    )

    stats_path = Path(cfg.normalization_stats_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run: python -m training.action_expert.compute_xyz_normalization "
            f"--dataset_dir {cfg.dataset_dir} --output {stats_path}"
        )
    xyz_mean_np, xyz_std_np, norm_eps_f = load_xyz_normalization_stats(stats_path)
    xyz_mean_t = torch.as_tensor(xyz_mean_np, dtype=torch.float32, device=device)
    xyz_std_t = torch.as_tensor(xyz_std_np, dtype=torch.float32, device=device)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(cfg.dataset_dir),
        "split_counts_shards": {k: len(v) for k, v in splits.items()},
        "split_counts_samples": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "config": asdict(cfg),
        "region_constraint_config": asdict(region_cfg),
        "xyz_normalization": {
            "stats_path": str(stats_path),
            "xyz_mean": xyz_mean_np.tolist(),
            "xyz_std": xyz_std_np.tolist(),
            "norm_eps": float(norm_eps_f),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_path = out_dir / "progress.json"
    train_metrics_jsonl = out_dir / "train_metrics.jsonl"
    bad_batch_jsonl = out_dir / "bad_batch_samples.jsonl"
    metrics_interval = max(1, int(cfg.metrics_log_every_steps))

    with _wandb_session(cfg, run_id=run_id, out_dir=out_dir):
        train_ds = ActionExpertDataset(
            train_samples,
            image_size=(cfg.image_size, cfg.image_size),
            xyz_mean=xyz_mean_np,
            xyz_std=xyz_std_np,
            norm_eps=float(norm_eps_f),
            region_cfg=region_cfg,
            sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
            goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
        )
        val_ds = ActionExpertDataset(
            val_samples,
            image_size=(cfg.image_size, cfg.image_size),
            xyz_mean=xyz_mean_np,
            xyz_std=xyz_std_np,
            norm_eps=float(norm_eps_f),
            region_cfg=region_cfg,
            sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
            goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
        )
        fixed_idxs = _fixed_eval_indices_by_token_and_variants(val_samples)
        if not fixed_idxs:
            fixed_n = max(1, int(cfg.wandb_prediction_num_examples))
            fixed_idxs = list(range(min(fixed_n, len(val_ds))))
        fixed_val_loader = DataLoader(
            Subset(val_ds, fixed_idxs),
            batch_size=max(1, min(int(cfg.batch_size), len(fixed_idxs))),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=action_expert_collate,
        )
        fixed_eval_n = min(200, len(val_ds))
        fixed200_idxs = _fixed_eval_indices_uniform_movement_tokens(
            val_samples,
            seed=int(cfg.seed),
            target_n=int(fixed_eval_n),
            variants_per_group=4,
        )
        if not fixed200_idxs:
            fixed200_idxs = list(range(fixed_eval_n))
        fixed200_token_counts: Dict[str, int] = defaultdict(int)
        for idx in fixed200_idxs:
            tok = str(getattr(val_samples[idx], "movement_token", "")).strip().lower()
            fixed200_token_counts[tok] += 1
        fixed200_val_loader = DataLoader(
            Subset(val_ds, fixed200_idxs),
            batch_size=max(1, min(int(cfg.batch_size), len(fixed200_idxs))),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=action_expert_collate,
        )
        print(
            f"[fixed_eval] size={len(fixed200_idxs)} seed={int(cfg.seed)} "
            f"tokens_uniform=true token_counts={json.dumps(dict(sorted(fixed200_token_counts.items())))}"
        )
        test_ds = ActionExpertDataset(
            test_samples,
            image_size=(cfg.image_size, cfg.image_size),
            xyz_mean=xyz_mean_np,
            xyz_std=xyz_std_np,
            norm_eps=float(norm_eps_f),
            region_cfg=region_cfg,
            sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
            goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=int(cfg.num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=action_expert_collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=max(0, int(cfg.num_workers) // 2),
            pin_memory=(device.type == "cuda"),
            collate_fn=action_expert_collate,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=max(0, int(cfg.num_workers) // 2),
            pin_memory=(device.type == "cuda"),
            collate_fn=action_expert_collate,
        )

        vlm = PaliGemmaContextEncoder(
            model_id=str(cfg.paligemma_model_id),
            device=device,
            cache_dir=str(cfg.hf_cache_dir),
            local_files_only=bool(cfg.local_files_only),
            lora_rank=int(cfg.lora_rank),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            enable_gradient_checkpointing=bool(cfg.enable_gradient_checkpointing),
        ).to(device)
        action_d_model = int(vlm.d_model)
        if int(cfg.d_model) != action_d_model:
            print(
                f"[init] overriding cfg.d_model={cfg.d_model} with VLM hidden size action_d_model={action_d_model}"
            )
        action_model = ActionExpertModel(
            d_model=action_d_model,
            num_heads=int(cfg.num_heads),
            num_layers=int(cfg.num_action_expert_layers),
            dropout=float(cfg.action_dropout),
            ffn_multiplier=int(cfg.ffn_multiplier),
            pos_norm_denom=float(cfg.pos_norm_denom),
        ).to(device)

        lora_params = vlm.lora_parameters()
        # Explicitly include full label_head params (Linear + LayerNorm gamma/beta).
        label_proj_params = list(vlm.label_head.parameters())
        expert_params = [p for p in action_model.parameters() if p.requires_grad]
        param_groups = []
        if lora_params:
            param_groups.append(
                {
                    "params": lora_params,
                    "lr": float(cfg.lora_lr),
                    "weight_decay": float(cfg.weight_decay),
                    "group_name": "lora",
                }
            )
        if label_proj_params:
            label_param_names = [name for name, _ in vlm.label_head.named_parameters()]
            print(f"[init] label_head_optimizer_params={label_param_names}")
            param_groups.append(
                {
                    "params": label_proj_params,
                    "lr": float(_effective_label_proj_lr(cfg)),
                    "weight_decay": float(cfg.label_proj_weight_decay),
                    "group_name": "label_proj",
                }
            )
        if expert_params:
            param_groups.append(
                {
                    "params": expert_params,
                    "lr": float(cfg.action_expert_lr),
                    "weight_decay": float(cfg.weight_decay),
                    "group_name": "action_expert",
                }
            )
        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer")
        optimizer = optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        accum_steps = max(1, int(cfg.gradient_accumulation_steps))
        optimizer_steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps
        total_optimizer_steps = max(1, int(cfg.epochs) * int(optimizer_steps_per_epoch))
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        amp_dtype = torch.bfloat16
        t0 = time.perf_counter()
        best_val = float("inf")
        best_fixed200_mean_l2 = float("inf")
        best_fixed200_success = float("-inf")
        metrics: List[Dict[str, Any]] = []
        global_step = 0
        optimizer_step = 0
        bad_batch_retries_for_pending_step = 0
        start_epoch = 1

        resume_path_raw = str(getattr(cfg, "resume_checkpoint_path", "")).strip()
        is_resumed = bool(resume_path_raw)
        resume_optimizer_step_base = 0
        if is_resumed:
            resume_path = Path(resume_path_raw)
            if not resume_path.is_absolute():
                resume_path = (REPO_ROOT / resume_path).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            mts = dict(ckpt.get("model_trainable_state", {}) or {})
            vlm_state = dict(mts.get("vlm_lora", {}) or {})
            if vlm_state:
                vlm.load_state_dict(vlm_state, strict=False)
            action_state = mts.get("action_expert")
            if action_state is None:
                raise KeyError(f"Checkpoint missing action_expert state: {resume_path}")
            action_model.load_state_dict(action_state)
            opt_state = ckpt.get("optimizer_state")
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
                # Option 2 resume: keep optimizer moments/state, but override
                # resumed learning rates from the current config.
                for g in optimizer.param_groups:
                    gname = str(g.get("group_name", "")).strip()
                    if gname == "lora":
                        new_lr = float(cfg.lora_lr)
                    elif gname == "label_proj":
                        new_lr = float(_effective_label_proj_lr(cfg))
                    elif gname == "action_expert":
                        new_lr = float(cfg.action_expert_lr)
                    else:
                        new_lr = float(cfg.lr)
                    g["lr"] = float(new_lr)
                    g["initial_lr"] = float(new_lr)
            scaler_state = ckpt.get("scaler_state")
            if scaler.is_enabled() and scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            best_val = float(ckpt.get("best_val_mean_l2_error_m", best_val))
            best_fixed200_mean_l2 = float(
                ckpt.get("best_fixed200_mean_l2_error_m", best_fixed200_mean_l2)
            )
            best_fixed200_success = float(
                ckpt.get("best_fixed200_success_rate", best_fixed200_success)
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            optimizer_step = int(ckpt.get("optimizer_step", 0))
            resume_optimizer_step_base = int(optimizer_step)
            print(
                f"[resume] checkpoint={resume_path} start_epoch={start_epoch} "
                f"global_step={global_step} optimizer_step={optimizer_step}"
            )

        warmup_steps = 0 if is_resumed else int(getattr(cfg, "cosine_warmup_steps", 500))
        action_tail_T = int(getattr(cfg, "action_expert_cosine_T_max_steps", 0) or 0)
        action_eta_min = float(getattr(cfg, "action_expert_cosine_eta_min", 1e-5) or 0.0)
        lora_tail_T = int(getattr(cfg, "lora_cosine_T_max_steps", 0) or 0)
        lora_eta_min = float(getattr(cfg, "lora_cosine_eta_min", 0.0) or 0.0)
        use_resume_group_tails = bool(is_resumed and (action_tail_T > 0 or lora_tail_T > 0))
        if use_resume_group_tails:
            opt_step0 = int(resume_optimizer_step_base)

            def _make_resume_tail_lambda(
                *,
                max_lr_g: float,
                T_max_g: int,
                eta_min_g: float,
            ) -> Callable[[int], float]:
                def _tail_lambda(
                    s: int,
                    *,
                    _T: int = int(T_max_g),
                    _mx: float = float(max_lr_g),
                    _mn: float = float(eta_min_g),
                    _base: int = int(opt_step0),
                ) -> float:
                    rel = max(0, int(s) - int(_base))
                    return _cosine_anneal_to_eta_multiplier(
                        int(rel), T_max=int(_T), max_lr=float(_mx), eta_min=float(_mn)
                    )

                return _tail_lambda

            lr_lambdas = []
            for g in optimizer.param_groups:
                gname = str(g.get("group_name", "")).strip()

                def _lora_lambda(
                    s: int,
                    *,
                    _warmup_steps: int = int(warmup_steps),
                    _total_steps: int = int(total_optimizer_steps),
                ) -> float:
                    return _cosine_lr_multiplier(
                        step_idx=int(s),
                        total_steps=int(_total_steps),
                        warmup_steps=int(_warmup_steps),
                    )

                if gname == "lora":
                    if lora_tail_T > 0:
                        lr_lambdas.append(
                            _make_resume_tail_lambda(
                                max_lr_g=float(cfg.lora_lr),
                                T_max_g=int(lora_tail_T),
                                eta_min_g=float(lora_eta_min),
                            )
                        )
                    else:
                        lr_lambdas.append(_lora_lambda)
                elif gname == "label_proj":
                    if action_tail_T > 0:
                        lr_lambdas.append(
                            _make_resume_tail_lambda(
                                max_lr_g=float(_effective_label_proj_lr(cfg)),
                                T_max_g=int(action_tail_T),
                                eta_min_g=float(action_eta_min),
                            )
                        )
                    else:
                        lr_lambdas.append(_lora_lambda)
                elif gname == "action_expert":
                    if action_tail_T > 0:
                        lr_lambdas.append(
                            _make_resume_tail_lambda(
                                max_lr_g=float(cfg.action_expert_lr),
                                T_max_g=int(action_tail_T),
                                eta_min_g=float(action_eta_min),
                            )
                        )
                    else:
                        lr_lambdas.append(_lora_lambda)
                else:
                    lr_lambdas.append(_lora_lambda)
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambdas,
                last_epoch=max(-1, int(optimizer_step) - 1),
            )
            print(
                "[scheduler] resumed=True warmup_steps="
                f"{warmup_steps} lora="
                f"{'cosine_anneal_T_max=' + str(lora_tail_T) + ' eta_min=' + str(lora_eta_min) if lora_tail_T > 0 else 'global_cosine'} "
                f"action_groups="
                f"{'cosine_anneal_T_max=' + str(action_tail_T) + ' eta_min=' + str(action_eta_min) if action_tail_T > 0 else 'global_cosine'}"
            )
        else:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda s: _cosine_lr_multiplier(
                    step_idx=int(s),
                    total_steps=int(total_optimizer_steps),
                    warmup_steps=int(warmup_steps),
                ),
                last_epoch=max(-1, int(optimizer_step) - 1),
            )
            print(
                f"[scheduler] resumed={is_resumed} warmup_steps={warmup_steps} "
                f"schedule=global_cosine total_optimizer_steps={total_optimizer_steps}"
            )

        def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")

        def _write_progress(payload: Dict[str, Any]) -> None:
            progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        def _run_fixed200_eval(*, epoch: int, global_step: int, source: str) -> Dict[str, Any]:
            fixed200_metrics = evaluate(
                cfg=cfg,
                region_cfg=region_cfg,
                vlm=vlm,
                action_model=action_model,
                loader=fixed200_val_loader,
                device=device,
                xyz_mean=xyz_mean_t,
                xyz_std=xyz_std_t,
                norm_eps=float(norm_eps_f),
            )
            print(
                f"[eval_fixed200] source={source} epoch={epoch} step={global_step} n={len(fixed200_idxs)} "
                f"constraint_sat={fixed200_metrics['val_constraint_satisfaction_rate']:.4f} "
                f"group_l2_mean={fixed200_metrics.get('val_group_l2_mean_to_sample_m', 0.0):.6f} "
                f"group_l2_std={fixed200_metrics.get('val_group_l2_std_to_sample_m', 0.0):.6f} "
                f"by_token={json.dumps(fixed200_metrics.get('val_success_by_movement_token', {}), default=str)}"
            )
            _append_jsonl(
                train_metrics_jsonl,
                {
                    "kind": "eval_fixed200",
                    "source": str(source),
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "fixed200_size": int(len(fixed200_idxs)),
                    "fixed200_constraint_satisfaction_rate": float(
                        fixed200_metrics["val_constraint_satisfaction_rate"]
                    ),
                    "fixed200_group_l2_mean_to_sample_m": float(
                        fixed200_metrics.get("val_group_l2_mean_to_sample_m", 0.0)
                    ),
                    "fixed200_group_l2_std_to_sample_m": float(
                        fixed200_metrics.get("val_group_l2_std_to_sample_m", 0.0)
                    ),
                    "fixed200_group_count": int(fixed200_metrics.get("val_group_count", 0)),
                    "fixed200_success_by_movement_token": fixed200_metrics.get(
                        "val_success_by_movement_token", {}
                    ),
                    "fixed200_group_l2_by_movement_token": fixed200_metrics.get(
                        "val_group_l2_by_movement_token", {}
                    ),
                },
            )
            _write_progress(
                {
                    "phase": "eval_fixed200",
                    "source": str(source),
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "fixed200_size": int(len(fixed200_idxs)),
                    "fixed200_constraint_satisfaction_rate": float(
                        fixed200_metrics["val_constraint_satisfaction_rate"]
                    ),
                    "fixed200_success_by_movement_token": fixed200_metrics.get(
                        "val_success_by_movement_token", {}
                    ),
                }
            )
            if bool(cfg.use_wandb):
                _wandb_log_split_metrics(
                    fixed200_metrics, step=int(global_step), prefix="val/fixed200"
                )
            return fixed200_metrics

        def _checkpoint_payload(epoch: int, best_val_metric: float) -> Dict[str, Any]:
            return {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "optimizer_step": int(optimizer_step),
                "run_id": run_id,
                "config": asdict(cfg),
                "region_constraint_config": asdict(region_cfg),
                "model_trainable_state": {
                    "vlm_lora": {
                        k: v
                        for k, v in vlm.state_dict().items()
                        if "lora_" in k or "label_head" in k or "label_proj" in k
                    },
                    "action_expert": action_model.state_dict(),
                },
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val_mean_l2_error_m": float(best_val_metric),
                "best_fixed200_mean_l2_error_m": float(best_fixed200_mean_l2),
                "best_fixed200_success_rate": float(best_fixed200_success),
                "frozen_backbone": {
                    "paligemma_model_id": str(cfg.paligemma_model_id),
                    "hf_cache_dir": str(cfg.hf_cache_dir),
                },
            }

        interrupt_epoch_state: Dict[str, int] = {"epoch": int(start_epoch)}
        interrupt_saved_state: Dict[str, bool] = {"saved": False}

        def _save_interrupt_checkpoint() -> None:
            if bool(interrupt_saved_state["saved"]):
                return
            interrupt_saved_state["saved"] = True
            ep = int(interrupt_epoch_state["epoch"])
            step = int(global_step)
            ckpt_name = f"checkpoint_interrupt_step_{step:07d}.pt"
            ckpt_path = out_dir / ckpt_name
            torch.save(_checkpoint_payload(ep, best_val), ckpt_path)
            _write_progress(
                {
                    "phase": "interrupted",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "epoch": int(ep),
                    "global_step": int(step),
                    "optimizer_step": int(optimizer_step),
                    "checkpoint": str(ckpt_path),
                }
            )
            print(f"[interrupt] saved_checkpoint={ckpt_path} global_step={step} epoch={ep}")

        def _handle_sigint(signum: int, frame: Any) -> None:
            del signum, frame
            _save_interrupt_checkpoint()
            raise KeyboardInterrupt

        def _handle_sigterm(signum: int, frame: Any) -> None:
            del signum, frame
            _save_interrupt_checkpoint()
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_sigint)
        signal.signal(signal.SIGTERM, _handle_sigterm)

        for epoch in range(int(start_epoch), int(cfg.epochs) + 1):
            interrupt_epoch_state["epoch"] = int(epoch)
            batch_losses: List[float] = []
            vlm.train()
            action_model.train()
            optimizer.zero_grad(set_to_none=True)
            print(f"[epoch_start] epoch={epoch}/{cfg.epochs} num_train_batches={len(train_loader)}")
            for batch_idx, raw in enumerate(train_loader, start=1):
                proposed_step = int(global_step) + 1
                batch = _batch_to_device(raw, device, cfg)
                flow = _sample_flow_inputs(batch["goal_xyz_norm"])
                sample_weights = torch.ones(flow["x1"].shape[0], device=device, dtype=torch.float32)

                with torch.autocast(device_type=autocast_device, enabled=use_amp, dtype=amp_dtype):
                    ctx = vlm.forward_context(
                        images_uint8=batch["image"],
                        system_prompts=batch["system_prompt"],
                        instructions=batch["instruction_text"],
                        object_labels=batch["object_labels"],
                    )
                    label_emb = vlm.embed_labels(batch["keypoint_labels"])
                    # Continuous instability diagnostics: log norm stats every 10 steps.
                    should_log_tensor_debug = proposed_step % 10 == 0
                    if should_log_tensor_debug:
                        label_norm_means: List[float] = []
                        pos_norm_means: List[float] = []
                        keypoint_norm_means: List[float] = []
                        for bi, kp_pos in enumerate(batch["keypoint_positions"]):
                            if kp_pos.numel() == 0:
                                continue
                            pos_emb = action_model.pos_mlp(
                                kp_pos.to(device=device, dtype=torch.float32) / action_model.pos_norm_denom
                            )
                            keypoint_token = label_emb[bi].to(device=device, dtype=torch.float32) + pos_emb
                            label_norm_means.append(float(label_emb[bi].norm(dim=-1).mean().item()))
                            pos_norm_means.append(float(pos_emb.norm(dim=-1).mean().item()))
                            keypoint_norm_means.append(float(keypoint_token.norm(dim=-1).mean().item()))
                        noisy_pos_token = action_model.action_in_proj(flow["xt"])
                        dbg = {
                            "step": int(proposed_step),
                            "label_emb_norm_mean": float(np.mean(label_norm_means)) if label_norm_means else 0.0,
                            "pos_emb_norm_mean": float(np.mean(pos_norm_means)) if pos_norm_means else 0.0,
                            "keypoint_token_norm_mean": float(np.mean(keypoint_norm_means))
                            if keypoint_norm_means
                            else 0.0,
                            "context_norm_mean": float(ctx["context"].norm(dim=-1).mean().item()),
                            "context_normed_norm_mean": float(
                                action_model.context_norm(ctx["context"]).norm(dim=-1).mean().item()
                            ),
                            "context_abs_max": float(ctx["context"].abs().max().item()),
                            "noisy_pos_token_norm_mean": float(
                                noisy_pos_token.norm(dim=-1).mean().item()
                            ),
                        }
                        print(
                            "[tensor_debug] "
                            f"step={dbg['step']} "
                            f"label_emb_norm={dbg['label_emb_norm_mean']:.4e} "
                            f"pos_emb_norm={dbg['pos_emb_norm_mean']:.4e} "
                            f"keypoint_token_norm={dbg['keypoint_token_norm_mean']:.4e} "
                            f"C_norm={dbg['context_norm_mean']:.4e} "
                            f"C_normed_norm={dbg['context_normed_norm_mean']:.4e} "
                            f"C_abs_max={dbg['context_abs_max']:.4e} "
                            f"noisy_pos_token_norm={dbg['noisy_pos_token_norm_mean']:.4e}"
                        )
                        if bool(cfg.use_wandb):
                            import wandb

                            wandb.log(
                                {
                                    "debug/label_emb_norm_mean": dbg["label_emb_norm_mean"],
                                    "debug/pos_emb_norm_mean": dbg["pos_emb_norm_mean"],
                                    "debug/keypoint_token_norm_mean": dbg["keypoint_token_norm_mean"],
                                    "debug/context_norm_mean": dbg["context_norm_mean"],
                                    "debug/context_normed_norm_mean": dbg["context_normed_norm_mean"],
                                    "debug/context_abs_max": dbg["context_abs_max"],
                                    "debug/noisy_pos_token_norm_mean": dbg["noisy_pos_token_norm_mean"],
                                },
                                step=int(proposed_step),
                            )
                    pred_v = action_model(
                        label_embeddings=label_emb,
                        keypoint_positions=batch["keypoint_positions"],
                        xt=flow["xt"],
                        t=flow["t"],
                        context=ctx["context"],
                        context_attention_mask=ctx["attention_mask"],
                        collect_debug_stats=bool(should_log_tensor_debug),
                    )
                    per = (pred_v - flow["target_v"]).pow(2).mean(dim=-1)
                    raw_loss = (per * sample_weights).mean()
                    loss = raw_loss / float(accum_steps)

                loss_f = float(raw_loss.detach().cpu().item())
                batch_losses.append(loss_f)
                should_step = (batch_idx % accum_steps == 0) or (batch_idx == len(train_loader))
                bad_batch_threshold = float(getattr(cfg, "bad_batch_loss_threshold", 50.0))
                bad_batch_grad_norm_threshold = float(
                    getattr(cfg, "bad_batch_grad_norm_threshold", 100.0)
                )
                bad_batch_abort_after_step = int(getattr(cfg, "bad_batch_abort_after_step", 300))
                bad_batch_max_retries_per_step = int(
                    getattr(cfg, "bad_batch_max_retries_per_step", 5)
                )
                is_bad_batch = loss_f > bad_batch_threshold
                should_handle_bad_batch = is_bad_batch and (proposed_step > bad_batch_abort_after_step)
                if should_handle_bad_batch:
                    _log_bad_batch_samples(
                        global_step=proposed_step,
                        loss_f=loss_f,
                        batch=batch,
                        flow=flow,
                        bad_batch_jsonl_path=bad_batch_jsonl,
                    )
                    if bool(cfg.use_wandb):
                        import wandb

                        wandb.log(
                            {
                                "debug/bad_batch_step": float(global_step),
                                "debug/bad_batch_loss": float(loss_f),
                                "debug/bad_batch_scene_ids": json.dumps(batch["scene_id"]),
                                "debug/bad_batch_shard_paths": json.dumps(batch["shard_path"]),
                                "debug/bad_batch_datapoint_indices": json.dumps(
                                    batch["datapoint_index"]
                                ),
                                "debug/bad_batch_instruction_variant_indices": json.dumps(
                                    batch["instruction_variant_index"]
                                ),
                                "debug/bad_batch_movement_tokens": json.dumps(batch["movement_token"]),
                                "debug/bad_batch_constraint_types": json.dumps(batch["constraint_type"]),
                            },
                            step=int(proposed_step),
                        )
                    print(
                        f"[bad_batch] post_warmup_skip step={proposed_step} "
                        f"loss={loss_f:.6f} threshold={bad_batch_threshold:.2f} "
                        "action=skip_update_continue"
                    )
                    bad_batch_retries_for_pending_step += 1
                    if bad_batch_retries_for_pending_step > int(bad_batch_max_retries_per_step):
                        print(
                            f"[bad_batch] retry_limit_reached step={proposed_step} "
                            f"retries={bad_batch_retries_for_pending_step} "
                            "action=allow_update"
                        )
                    else:
                        if should_step and (proposed_step % max(1, int(cfg.log_every_steps)) == 0):
                            print(
                                f"[train] epoch={epoch} step={proposed_step} loss={loss_f:.6f} "
                                f"action=bad_batch_skip_update retry={bad_batch_retries_for_pending_step}/"
                                f"{bad_batch_max_retries_per_step}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                global_step = proposed_step
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                gn_lora = 0.0
                gn_lbl = 0.0
                gn_exp = 0.0
                lr_row = {
                    f"lr_{g['group_name']}": float(g["lr"])
                    for g in optimizer.param_groups
                    if "group_name" in g
                }

                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    gn_lora = _grad_norm_tensors(lora_params)
                    gn_lbl = _grad_norm_tensors(label_proj_params)
                    gn_exp = _grad_norm_tensors(expert_params)
                    grad_norm_peak = max(float(gn_lora), float(gn_lbl), float(gn_exp))
                    is_bad_grad_batch = grad_norm_peak > bad_batch_grad_norm_threshold
                    should_handle_bad_grad_batch = bool(
                        is_bad_grad_batch and (proposed_step > bad_batch_abort_after_step)
                    )
                    if should_handle_bad_grad_batch:
                        bad_batch_retries_for_pending_step += 1
                        if bool(cfg.use_wandb):
                            import wandb

                            wandb.log(
                                {
                                    "debug/bad_batch_step": float(proposed_step),
                                    "debug/bad_batch_grad_norm_peak": float(grad_norm_peak),
                                    "debug/bad_batch_grad_norm_lora": float(gn_lora),
                                    "debug/bad_batch_grad_norm_label_proj": float(gn_lbl),
                                    "debug/bad_batch_grad_norm_action_expert": float(gn_exp),
                                },
                                step=int(proposed_step),
                            )
                        print(
                            f"[bad_batch] post_warmup_skip step={proposed_step} "
                            f"grad_norm_peak={grad_norm_peak:.6f} threshold={bad_batch_grad_norm_threshold:.2f} "
                            "action=skip_update_continue"
                        )
                        if bad_batch_retries_for_pending_step > int(bad_batch_max_retries_per_step):
                            print(
                                f"[bad_batch] retry_limit_reached step={proposed_step} "
                                f"retries={bad_batch_retries_for_pending_step} "
                                "action=allow_update"
                            )
                        else:
                            if proposed_step % max(1, int(cfg.log_every_steps)) == 0:
                                print(
                                    f"[train] epoch={epoch} step={proposed_step} loss={loss_f:.6f} "
                                    f"action=bad_grad_skip_update retry={bad_batch_retries_for_pending_step}/"
                                    f"{bad_batch_max_retries_per_step} grad_norm_peak={grad_norm_peak:.6f}"
                                )
                            optimizer.zero_grad(set_to_none=True)
                            global_step = int(global_step) - 1
                            continue
                    torch.nn.utils.clip_grad_norm_(expert_params + lora_params + label_proj_params, 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    scheduler.step()
                    bad_batch_retries_for_pending_step = 0
                if batch_idx == 1:
                    print(
                        f"[train_first_step] epoch={epoch} step={global_step} loss={loss_f:.6f} "
                        f"gn_lora={gn_lora:.4e} gn_label_proj={gn_lbl:.4e} gn_action_expert={gn_exp:.4e} "
                        f"lrs={lr_row}"
                    )
                if should_step and (global_step % max(1, int(cfg.log_every_steps)) == 0):
                    print(
                        f"[train] epoch={epoch} step={global_step} loss={loss_f:.6f} "
                        f"gn_lora={gn_lora:.4e} gn_label_proj={gn_lbl:.4e} gn_action_expert={gn_exp:.4e} "
                        f"opt_step={optimizer_step} accum_steps={accum_steps} lrs={lr_row}"
                    )

                if should_step and (global_step % 500 == 0):
                    fixed200_metrics_step = _run_fixed200_eval(
                        epoch=int(epoch), global_step=int(global_step), source="step_500"
                    )
                    fixed200_mean_l2 = float(
                        fixed200_metrics_step.get("val_mean_l2_error_m", float("inf"))
                    )
                    fixed200_success = float(
                        fixed200_metrics_step.get("val_constraint_satisfaction_rate", 0.0)
                    )
                    if fixed200_mean_l2 < best_fixed200_mean_l2:
                        best_fixed200_mean_l2 = fixed200_mean_l2
                        torch.save(
                            _checkpoint_payload(epoch, best_val),
                            out_dir / "checkpoint_best_fixed200_mean_l2.pt",
                        )
                    if fixed200_success > best_fixed200_success:
                        best_fixed200_success = fixed200_success
                        torch.save(
                            _checkpoint_payload(epoch, best_val),
                            out_dir / "checkpoint_best_fixed200_success_rate.pt",
                        )

                if should_step and (global_step % 1000 == 0):
                    torch.save(
                        _checkpoint_payload(epoch, best_val),
                        out_dir / f"checkpoint_step_{int(global_step):07d}.pt",
                    )

                if should_step and (global_step % metrics_interval == 0):
                    cross_attn_out_norm = float(
                        action_model.last_debug_stats.get("cross_attn_out_norm_mean", 0.0)
                    )
                    train_step_row: Dict[str, Any] = {
                        "kind": "train_step",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "global_step": int(global_step),
                        "optimizer_step": int(optimizer_step),
                        "epoch": int(epoch),
                        "batch_in_epoch": int(batch_idx),
                        "num_train_batches": int(len(train_loader)),
                        "loss": loss_f,
                        "grad_norm_lora": gn_lora,
                        "grad_norm_label_proj": gn_lbl,
                        "grad_norm_action_expert": gn_exp,
                        "debug_cross_attn_out_norm": cross_attn_out_norm,
                        **lr_row,
                    }
                    _append_jsonl(train_metrics_jsonl, train_step_row)
                    _write_progress(
                        {
                            "phase": "train",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "optimizer_step": int(optimizer_step),
                            "batch_in_epoch": int(batch_idx),
                            "num_train_batches": int(len(train_loader)),
                            "last_loss": loss_f,
                            "grad_norm_lora": gn_lora,
                            "grad_norm_label_proj": gn_lbl,
                            "grad_norm_action_expert": gn_exp,
                            "learning_rates": lr_row,
                            "elapsed_seconds": float(time.perf_counter() - t0),
                        }
                    )
                    if bool(cfg.use_wandb):
                        import wandb

                        wb_train: Dict[str, Any] = {
                            "train/loss": loss_f,
                            "train/epoch": float(epoch),
                            "train/batch_in_epoch": float(batch_idx),
                            "train/elapsed_s": float(time.perf_counter() - t0),
                            "train/grad_norm_lora": gn_lora,
                            "train/grad_norm_label_proj": gn_lbl,
                            "train/grad_norm_action_expert": gn_exp,
                            "debug/cross_attn_out_norm": cross_attn_out_norm,
                        }
                        for k, v in lr_row.items():
                            wb_train[f"train/{k}"] = float(v)
                        wandb.log(wb_train, step=int(global_step))
                        pred_log_every = max(1, int(cfg.wandb_prediction_log_every_steps))
                        if global_step % pred_log_every == 0:
                            _wandb_log_fixed_prediction_examples(
                                step=int(global_step),
                                cfg=cfg,
                                region_cfg=region_cfg,
                                vlm=vlm,
                                action_model=action_model,
                                loader=fixed_val_loader,
                                device=device,
                                xyz_mean=xyz_mean_t,
                                xyz_std=xyz_std_t,
                                norm_eps=float(norm_eps_f),
                            )

            heartbeat = {
                "epoch": int(epoch),
                "elapsed_seconds": float(time.perf_counter() - t0),
                "train_loss_mean": float(np.mean(batch_losses)) if batch_losses else 0.0,
            }
            _write_progress(
                {
                    "phase": "epoch_end",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **heartbeat,
                    "global_step": int(global_step),
                    "num_train_batches": int(len(train_loader)),
                }
            )
            _append_jsonl(
                train_metrics_jsonl,
                {
                    "kind": "train_epoch",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "train_loss_mean": heartbeat["train_loss_mean"],
                    "elapsed_seconds": heartbeat["elapsed_seconds"],
                },
            )
            if bool(cfg.use_wandb):
                import wandb

                wandb.log(
                    {
                        "train/epoch_loss_mean": heartbeat["train_loss_mean"],
                        "train/epoch_elapsed_s": heartbeat["elapsed_seconds"],
                    },
                    step=int(global_step),
                )

            should_eval = (epoch % max(1, int(cfg.eval_every_epochs)) == 0) or epoch == 1
            if should_eval:
                val_metrics = evaluate(
                    cfg=cfg,
                    region_cfg=region_cfg,
                    vlm=vlm,
                    action_model=action_model,
                    loader=val_loader,
                    device=device,
                    xyz_mean=xyz_mean_t,
                    xyz_std=xyz_std_t,
                    norm_eps=float(norm_eps_f),
                )
                fixed200_metrics = _run_fixed200_eval(
                    epoch=int(epoch), global_step=int(global_step), source="epoch_eval"
                )
                fixed200_mean_l2 = float(fixed200_metrics.get("val_mean_l2_error_m", float("inf")))
                fixed200_success = float(
                    fixed200_metrics.get("val_constraint_satisfaction_rate", 0.0)
                )
                if fixed200_mean_l2 < best_fixed200_mean_l2:
                    best_fixed200_mean_l2 = fixed200_mean_l2
                    torch.save(
                        _checkpoint_payload(epoch, best_val),
                        out_dir / "checkpoint_best_fixed200_mean_l2.pt",
                    )
                if fixed200_success > best_fixed200_success:
                    best_fixed200_success = fixed200_success
                    torch.save(
                        _checkpoint_payload(epoch, best_val),
                        out_dir / "checkpoint_best_fixed200_success_rate.pt",
                    )
                row = {"epoch": int(epoch), **heartbeat, **val_metrics}
                metrics.append(row)
                mov_bits: List[str] = []
                vbm = val_metrics.get("val_by_movement") or {}
                if isinstance(vbm, dict):
                    for bk, st in sorted(vbm.items()):
                        if isinstance(st, dict) and "mean_l2_m" in st and "n" in st:
                            mov_bits.append(f"{bk}:μ={st['mean_l2_m']:.4f} med={st.get('median_l2_m', 0):.4f} n={st['n']}")
                mov_s = (" | " + " ".join(mov_bits)) if mov_bits else ""
                print(
                    f"[eval] epoch={epoch} mean_l2={row['val_mean_l2_error_m']:.6f} "
                    f"median_l2={row['val_median_l2_error_m']:.6f} "
                    f"constraint_sat={row['val_constraint_satisfaction_rate']:.4f}{mov_s}"
                )
                print(
                    f"[eval_fixed200] epoch={epoch} n={len(fixed200_idxs)} "
                    f"constraint_sat={fixed200_metrics['val_constraint_satisfaction_rate']:.4f} "
                    f"group_l2_mean={fixed200_metrics.get('val_group_l2_mean_to_sample_m', 0.0):.6f} "
                    f"group_l2_std={fixed200_metrics.get('val_group_l2_std_to_sample_m', 0.0):.6f} "
                    f"by_token={json.dumps(fixed200_metrics.get('val_success_by_movement_token', {}), default=str)}"
                )
                _append_jsonl(
                    train_metrics_jsonl,
                    {
                        "kind": "eval",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        **val_metrics,
                    },
                )
                _write_progress(
                    {
                        "phase": "eval",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        **heartbeat,
                        **val_metrics,
                    }
                )
                if bool(cfg.use_wandb):
                    _wandb_log_split_metrics(val_metrics, step=int(global_step), prefix="val")
                if row["val_mean_l2_error_m"] < best_val:
                    best_val = float(row["val_mean_l2_error_m"])
                    torch.save(_checkpoint_payload(epoch, best_val), out_dir / "checkpoint_best.pt")
                (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            if epoch % max(1, int(cfg.checkpoint_every_epochs)) == 0 or epoch == 1:
                torch.save(_checkpoint_payload(epoch, best_val), out_dir / f"checkpoint_{epoch:06d}.pt")

        if (out_dir / "checkpoint_best.pt").exists():
            ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device)
            action_model.load_state_dict(ckpt["model_trainable_state"]["action_expert"])
        test_metrics = evaluate(
            cfg=cfg,
            region_cfg=region_cfg,
            vlm=vlm,
            action_model=action_model,
            loader=test_loader,
            device=device,
            xyz_mean=xyz_mean_t,
            xyz_std=xyz_std_t,
            norm_eps=float(norm_eps_f),
        )
        final = {"best_val_mean_l2_error_m": float(best_val), **{f"test::{k}": v for k, v in test_metrics.items()}}
        (out_dir / "final_test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(
            f"[test] best_val={final['best_val_mean_l2_error_m']:.6f} "
            f"test_l2={final['test::val_mean_l2_error_m']:.6f}"
        )
        if bool(cfg.use_wandb):
            import wandb

            _wandb_log_split_metrics(test_metrics, step=int(global_step), prefix="test")
            wandb.log({"summary/best_val_mean_l2_error_m": float(best_val)}, step=int(global_step))


def main() -> None:
    apply_hf_env()
    parser = argparse.ArgumentParser(description="Train action-expert flow-matching module.")
    parser.add_argument("--config", type=Path, required=True, help="Path to action expert yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cache_root = apply_hf_cache(str(cfg.hf_cache_dir))
    print(f"[hf] HF_HOME={cache_root} HF_HUB_CACHE={cache_root / 'hub'}")
    train(cfg)


if __name__ == "__main__":
    main()

