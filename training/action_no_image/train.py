"""Train image-free CLIP + self-attention flow-matching model."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.losses import (
    RegionConstraintConfig,
    goal_region_contains,
    movement_token_eval_bucket,
)
from training.action_no_image.config import ActionNoImageConfig, load_config
from training.action_no_image.dataset import (
    NoImageActionDataset,
    list_shard_paths_multi,
    load_no_image_samples_from_shard,
    load_no_image_samples_from_shards,
    no_image_collate,
    resolve_shard_path,
    split_sample_indices,
)
from training.action_no_image.viz import (
    compose_sample_comparison_pair,
    pick_fixed_viz_indices,
    render_all_keypoints_with_xyz,
    render_success_region_panel,
)
from training.action_no_image.model import ActionNoImageModel
from training.action_no_image.text_encoder import ClipTextEncoder
from training.action_expert.xyz_normalization import denormalize_xyz_torch, load_xyz_normalization_stats
def _prepare_run_dir(output_root: Path) -> tuple[Path, str]:
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
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _region_cfg(cfg: ActionNoImageConfig) -> RegionConstraintConfig:
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


def _canonical_shard_paths(paths: List[Path]) -> list[str]:
    return sorted(str(p.resolve()) for p in paths)


def _assert_norm_stats_match_shards(
    stats_path: Path,
    shard_paths: List[Path],
    single_scene_id: str,
) -> None:
    try:
        meta = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    saved = meta.get("shard_paths")
    if isinstance(saved, list) and len(saved) > 0:
        want = _canonical_shard_paths(shard_paths)
        got = sorted(str(Path(x).resolve()) for x in saved)
        if want != got:
            raise ValueError(
                f"normalization_stats_path {stats_path} was computed for different shards than "
                f"this run (expected {len(want)} paths, stats lists {len(got)}; paths differ)."
            )
        return
    if len(shard_paths) > 1:
        raise ValueError(
            f"Multi-shard training requires {stats_path} to include a shard_paths list "
            "(run: python -m training.action_no_image.compute_xyz_normalization "
            "--dataset_dir ... --max_shards N --output ...)."
        )
    if not single_scene_id:
        return
    stem = stats_path.stem
    if str(single_scene_id) not in stem:
        raise ValueError(
            f"normalization_stats_path {stats_path} must contain scene_id '{single_scene_id}' "
            "in the filename (e.g. normalization_stats_action_no_image_scene_03546.json) "
            "so global dataset stats are not reused by mistake."
        )


def _batch_to_model_inputs(
    batch: Dict[str, Any],
    clip: ClipTextEncoder,
    device: torch.device,
) -> Dict[str, Any]:
    bsz = len(batch["instruction_text"])
    instr_clip = clip.encode(batch["instruction_text"])
    tool_clip = clip.encode(batch["tool_label"])
    ref_clip = clip.encode(batch["ref_label"])
    sec_texts = [
        batch["secondary_ref_label"][i]
        if bool(batch["has_secondary_ref"][i].item())
        else "none"
        for i in range(bsz)
    ]
    sec_clip = clip.encode(sec_texts)
    tbl0 = batch["table_label"][0]
    table_clip = clip.encode([tbl0] * len(batch["instruction_text"]))
    return {
        "instr_clip": instr_clip,
        "tool_clip": tool_clip,
        "ref_clip": ref_clip,
        "sec_clip": sec_clip,
        "table_clip": table_clip,
        "tool_xyz_norm": batch["tool_xyz_norm"].to(device),
        "ref_xyz_norm": batch["ref_xyz_norm"].to(device),
        "sec_xyz_norm": batch["sec_xyz_norm"].to(device),
        "table_xyz_norm": batch["table_xyz_norm"].to(device),
        "goal_xyz_norm": batch["goal_xyz_norm"].to(device),
        "has_secondary_ref": batch["has_secondary_ref"].to(device),
    }


@contextmanager
def _wandb_session(cfg: ActionNoImageConfig, *, run_id: str, out_dir: Path, scene_id: str):
    if not bool(cfg.use_wandb):
        yield
        return
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("use_wandb is true but wandb is not installed. pip install wandb") from exc
    init_kw: Dict[str, Any] = {
        "project": str(cfg.wandb_project),
        "name": str(cfg.wandb_run_name).strip() or run_id,
        "config": {**asdict(cfg), "run_id": run_id, "output_dir": str(out_dir), "scene_id": scene_id},
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


def _wandb_log_split_metrics(metrics: Dict[str, Any], *, step: int, prefix: str) -> None:
    """Log aggregate + per-token success under `{prefix}/`, per-token L2 under `{prefix}_debug/`."""
    import wandb

    p = str(prefix).strip("/")
    p_dbg = f"{p}_debug"
    flat: Dict[str, Any] = {
        f"{p}/mean_l2_error_m": metrics["val_mean_l2_error_m"],
        f"{p}/l2_std_error_m": metrics.get("val_l2_std_error_m", 0.0),
        f"{p}/success_rate": metrics["val_constraint_satisfaction_rate"],
    }
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
                flat[f"{p_dbg}/movement_token/{t}/mean_l2_error_m"] = stats["mean_l2_error_m"]
            if "l2_std_error_m" in stats:
                flat[f"{p_dbg}/movement_token/{t}/l2_std_error_m"] = stats["l2_std_error_m"]
    wandb.log(flat, step=int(step))


def _sample_flow(goal_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
    bsz = goal_xyz.shape[0]
    x0 = torch.randn_like(goal_xyz)
    t = torch.rand(bsz, device=goal_xyz.device, dtype=torch.float32)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * goal_xyz
    target_v = goal_xyz - x0
    return {"x0": x0, "x1": goal_xyz, "t": t, "xt": xt, "target_v": target_v}


@torch.no_grad()
def _rollout(
    *,
    model: ActionNoImageModel,
    clip: ClipTextEncoder,
    batch_tensors: Dict[str, Any],
    steps: int,
    n_samples: int,
) -> torch.Tensor:
    device = batch_tensors["goal_xyz_norm"].device
    bsz = batch_tensors["goal_xyz_norm"].shape[0]
    dt = 1.0 / max(1, int(steps))
    finals: List[torch.Tensor] = []
    for _ in range(int(n_samples)):
        x = torch.randn(bsz, 3, device=device)
        for i in range(int(steps)):
            t = torch.full((bsz,), float(i) / float(max(1, steps)), device=device)
            pred = model(
                instr_clip=batch_tensors["instr_clip"],
                tool_clip=batch_tensors["tool_clip"],
                ref_clip=batch_tensors["ref_clip"],
                sec_clip=batch_tensors["sec_clip"],
                table_clip=batch_tensors["table_clip"],
                tool_xyz_norm=batch_tensors["tool_xyz_norm"],
                ref_xyz_norm=batch_tensors["ref_xyz_norm"],
                sec_xyz_norm=batch_tensors["sec_xyz_norm"],
                table_xyz_norm=batch_tensors["table_xyz_norm"],
                xt=x,
                t=t,
                has_secondary_ref=batch_tensors["has_secondary_ref"],
            )
            x = x + pred * dt
        finals.append(x)
    return torch.stack(finals, dim=1)


@torch.no_grad()
def _predict_world_one(
    collated: Dict[str, Any],
    *,
    model: ActionNoImageModel,
    clip: ClipTextEncoder,
    device: torch.device,
    cfg: ActionNoImageConfig,
    region_cfg: RegionConstraintConfig,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Single-row prediction in world meters (same aggregation as `evaluate`)."""
    batch = {k: v for k, v in collated.items()}
    m = _batch_to_model_inputs(batch, clip, device)
    samples = _rollout(
        model=model,
        clip=clip,
        batch_tensors=m,
        steps=int(cfg.integration_steps),
        n_samples=int(cfg.inference_samples),
    )
    dataset_goals = batch["dataset_goal_xyz_world"].to(device)
    valid_rows: List[torch.Tensor] = []
    for s in range(samples.shape[1]):
        p = denormalize_xyz_torch(samples[0, s], xyz_mean, xyz_std, float(norm_eps))
        ok = goal_region_contains(
            pred_xyz=p,
            goal_xyz=dataset_goals[0],
            movement_token=batch["movement_token"][0],
            constraint_type=batch["constraint_type"][0],
            constraint_params=batch["constraint_params"][0],
            cfg=region_cfg,
        )
        if ok:
            valid_rows.append(samples[0, s])
    if valid_rows:
        pred_n = torch.stack(valid_rows, dim=0).mean(dim=0)
    else:
        pred_n = samples[0].mean(dim=0)
    return denormalize_xyz_torch(pred_n, xyz_mean, xyz_std, float(norm_eps))


@torch.no_grad()
def evaluate(
    *,
    cfg: ActionNoImageConfig,
    region_cfg: RegionConstraintConfig,
    model: ActionNoImageModel,
    clip: ClipTextEncoder,
    loader: DataLoader[Dict[str, Any]],
    device: torch.device,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> Dict[str, Any]:
    model.eval()
    clip.eval()
    all_l2: List[float] = []
    all_success: List[bool] = []
    all_tokens: List[str] = []
    all_buckets: List[str] = []
    in_region_count = 0
    total_count = 0

    for raw in loader:
        batch = {k: v for k, v in raw.items()}
        goals_norm = batch["goal_xyz_norm"].to(device)
        goals_world = batch["goal_xyz_world"].to(device)
        dataset_goals = batch["dataset_goal_xyz_world"].to(device)
        m = _batch_to_model_inputs(batch, clip, device)
        samples = _rollout(
            model=model,
            clip=clip,
            batch_tensors=m,
            steps=int(cfg.integration_steps),
            n_samples=int(cfg.inference_samples),
        )

        bsz = goals_norm.shape[0]
        for i in range(bsz):
            valid_rows: List[torch.Tensor] = []
            for s in range(samples.shape[1]):
                p = denormalize_xyz_torch(samples[i, s], xyz_mean, xyz_std, float(norm_eps))
                ok = goal_region_contains(
                    pred_xyz=p,
                    goal_xyz=dataset_goals[i],
                    movement_token=batch["movement_token"][i],
                    constraint_type=batch["constraint_type"][i],
                    constraint_params=batch["constraint_params"][i],
                    cfg=region_cfg,
                )
                if ok:
                    valid_rows.append(samples[i, s])
            if valid_rows:
                pred_n = torch.stack(valid_rows, dim=0).mean(dim=0)
                in_region_count += 1
                sample_success = True
            else:
                pred_n = samples[i].mean(dim=0)
                sample_success = False
            pred_w = denormalize_xyz_torch(pred_n, xyz_mean, xyz_std, float(norm_eps))
            err = float(torch.linalg.norm(pred_w - goals_world[i]).item())
            all_l2.append(err)
            all_success.append(bool(sample_success))
            tok = str(batch["movement_token"][i]).strip().lower()
            all_tokens.append(tok)
            all_buckets.append(
                movement_token_eval_bucket(batch["movement_token"][i], batch["constraint_type"][i])
            )
            total_count += 1

    model.train()
    return {
        "val_mean_l2_error_m": float(np.mean(all_l2)) if all_l2 else 0.0,
        "val_l2_std_error_m": float(np.std(np.asarray(all_l2, dtype=np.float64))) if all_l2 else 0.0,
        "val_constraint_satisfaction_rate": float(in_region_count / max(1, total_count)),
        "val_success_by_movement_token": _success_by_token(all_tokens, all_success),
        "val_l2_by_movement_token": _l2_by_token(all_tokens, all_l2),
        "val_by_movement": _by_bucket(all_buckets, all_l2),
    }


def _success_by_token(tokens: List[str], success: List[bool]) -> Dict[str, Any]:
    from collections import defaultdict

    c: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t, s in zip(tokens, success):
        c[t][1] += 1
        if s:
            c[t][0] += 1
    out: Dict[str, Any] = {}
    for t in sorted(c.keys()):
        ok_n, n = c[t]
        out[t] = {"success_rate": float(ok_n / max(1, n)), "n": int(n)}
    return out


def _l2_by_token(tokens: List[str], l2s: List[float]) -> Dict[str, Any]:
    from collections import defaultdict

    d: dict[str, List[float]] = defaultdict(list)
    for t, e in zip(tokens, l2s):
        d[t].append(float(e))
    out: Dict[str, Any] = {}
    for t in sorted(d.keys()):
        arr = np.asarray(d[t], dtype=np.float64)
        out[t] = {"mean_l2_error_m": float(np.mean(arr)), "l2_std_error_m": float(np.std(arr))}
    return out


def _by_bucket(buckets: List[str], l2s: List[float]) -> Dict[str, Any]:
    from collections import defaultdict

    d: dict[str, List[float]] = defaultdict(list)
    for b, e in zip(buckets, l2s):
        d[b].append(float(e))
    out: Dict[str, Any] = {}
    for b in sorted(d.keys()):
        arr = np.asarray(d[b], dtype=np.float64)
        out[b] = {
            "mean_l2_m": float(np.mean(arr)),
            "median_l2_m": float(np.median(arr)),
            "n": int(len(d[b])),
        }
    return out


def train(cfg: ActionNoImageConfig) -> None:
    _seed_everything(int(cfg.seed))
    device = torch.device(cfg.device)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"

    dataset_dir = Path(cfg.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir

    max_shards = int(getattr(cfg, "max_shards", 0) or 0)
    if max_shards > 0:
        shard_paths = list_shard_paths_multi(
            dataset_dir,
            max_shards,
            explode_instruction_variants=bool(cfg.explode_instruction_variants),
        )
    else:
        shard_paths = [resolve_shard_path(dataset_dir, str(cfg.shard_path))]

    scene_ids: List[str] = []
    for p in shard_paths:
        sj = json.loads(p.read_text(encoding="utf-8"))
        scene_ids.append(str(sj.get("scene_id", "")))
    scene_id = scene_ids[0] if len(shard_paths) == 1 else f"multi{len(shard_paths)}"

    stats_path = Path(cfg.normalization_stats_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    if not stats_path.is_file():
        hint = (
            f"  python -m training.action_no_image.compute_xyz_normalization "
            f"--dataset_dir {dataset_dir} --max_shards {max_shards} --output {stats_path}"
            if max_shards > 0
            else (
                f"  python -m training.action_no_image.compute_xyz_normalization "
                f"--shard_path {shard_paths[0]} --output {stats_path}"
            )
        )
        raise FileNotFoundError(f"Missing {stats_path}. Run:\n{hint}")
    _assert_norm_stats_match_shards(
        stats_path,
        shard_paths,
        single_scene_id=scene_ids[0] if len(shard_paths) == 1 else "",
    )

    xyz_mean_np, xyz_std_np, norm_eps_f = load_xyz_normalization_stats(stats_path)
    xyz_mean_t = torch.as_tensor(xyz_mean_np, dtype=torch.float32, device=device)
    xyz_std_t = torch.as_tensor(xyz_std_np, dtype=torch.float32, device=device)

    all_samples = (
        load_no_image_samples_from_shards(
            shard_paths,
            explode_instruction_variants=bool(cfg.explode_instruction_variants),
        )
        if len(shard_paths) > 1
        else load_no_image_samples_from_shard(
            shard_paths[0],
            explode_instruction_variants=bool(cfg.explode_instruction_variants),
        )
    )
    n = len(all_samples)
    splits = split_sample_indices(
        n,
        seed=int(cfg.seed),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
    )
    train_samples = [all_samples[i] for i in splits["train"]]
    val_samples = [all_samples[i] for i in splits["val"]]
    test_samples = [all_samples[i] for i in splits["test"]]

    region_cfg = _region_cfg(cfg)
    table_world = np.asarray(cfg.table_xyz_world, dtype=np.float64).reshape(3)

    out_dir, run_id = _prepare_run_dir(Path(cfg.output_dir))
    print(
        f"[run] output_dir={out_dir} scene_id={scene_id} "
        f"n_shards={len(shard_paths)} shards={[p.name for p in shard_paths]}"
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard_paths": [str(p) for p in shard_paths],
        "scene_ids": scene_ids,
        "scene_id": scene_id,
        "split_indices": {k: v for k, v in splits.items()},
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "config": asdict(cfg),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_ds = NoImageActionDataset(
        train_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
        table_xyz_world=table_world,
        table_label=str(cfg.table_label),
        region_cfg=region_cfg,
        sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
        goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
    )
    val_ds = NoImageActionDataset(
        val_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
        table_xyz_world=table_world,
        table_label=str(cfg.table_label),
        region_cfg=region_cfg,
        sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
        goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
    )
    test_ds = NoImageActionDataset(
        test_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
        table_xyz_world=table_world,
        table_label=str(cfg.table_label),
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
        collate_fn=no_image_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=no_image_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=no_image_collate,
    )

    clip = ClipTextEncoder(
        model_id=str(cfg.clip_model_id),
        device=device,
        cache_dir=str(cfg.hf_cache_dir),
        local_files_only=bool(cfg.local_files_only),
    )
    model = ActionNoImageModel(
        d_clip=int(clip.d_clip),
        d_model=int(cfg.d_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.action_dropout),
        ffn_multiplier=int(cfg.ffn_multiplier),
        pos_norm_denom=float(cfg.pos_norm_denom),
    ).to(device)

    label_proj_params = list(model.instr_proj.parameters()) + list(model.label_proj.parameters())
    action_expert_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("instr_proj") and not n.startswith("label_proj")
    ]
    optimizer = optim.AdamW(
        [
            {"params": action_expert_params, "lr": float(cfg.action_expert_lr), "group_name": "action_expert"},
            {"params": label_proj_params, "lr": float(cfg.label_proj_lr), "group_name": "label_proj"},
        ],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    optimizer_steps_per_epoch = max(1, len(train_loader))
    total_optimizer_steps = max(1, int(cfg.epochs) * int(optimizer_steps_per_epoch))
    warmup = int(cfg.cosine_warmup_steps)

    def lr_mult(step: int) -> float:
        s = max(0, int(step))
        if warmup > 0 and s < warmup:
            return float((s + 1) / warmup)
        prog = float(s - warmup) / float(max(1, total_optimizer_steps - warmup))
        prog = min(max(prog, 0.0), 1.0)
        return float(0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_mult(s))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16

    global_step = 0
    best_val = float("inf")
    metrics_path = out_dir / "train_metrics.jsonl"
    t0 = time.perf_counter()

    def append_jsonl(row: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    fixed_viz_idx = pick_fixed_viz_indices(
        val_samples,
        n=min(max(0, int(cfg.wandb_prediction_num_examples)), len(val_samples)),
        seed=int(cfg.seed),
    )

    with _wandb_session(cfg, run_id=run_id, out_dir=out_dir, scene_id=scene_id):
        clip.eval()
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            epoch_losses: List[float] = []
            for batch_idx, raw in enumerate(train_loader, start=1):
                global_step += 1
                goals = raw["goal_xyz_norm"].to(device)
                flow = _sample_flow(goals)
                m = _batch_to_model_inputs(raw, clip, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=autocast_device, enabled=use_amp, dtype=amp_dtype):
                    pred = model(
                        instr_clip=m["instr_clip"],
                        tool_clip=m["tool_clip"],
                        ref_clip=m["ref_clip"],
                        sec_clip=m["sec_clip"],
                        table_clip=m["table_clip"],
                        tool_xyz_norm=m["tool_xyz_norm"],
                        ref_xyz_norm=m["ref_xyz_norm"],
                        sec_xyz_norm=m["sec_xyz_norm"],
                        table_xyz_norm=m["table_xyz_norm"],
                        xt=flow["xt"],
                        t=flow["t"],
                        has_secondary_ref=m["has_secondary_ref"],
                    )
                    loss = F.mse_loss(pred, flow["target_v"])

                loss_f = float(loss.detach().cpu().item())
                epoch_losses.append(loss_f)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                if global_step % max(1, int(cfg.log_every_steps)) == 0:
                    lrs = {g.get("group_name", "?"): float(g["lr"]) for g in optimizer.param_groups}
                    print(
                        f"[train] epoch={epoch} step={global_step} loss={loss_f:.6f} "
                        f"elapsed={time.perf_counter() - t0:.1f}s lrs={lrs}"
                    )
                if global_step % max(1, int(cfg.metrics_log_every_steps)) == 0:
                    append_jsonl(
                        {
                            "kind": "train_step",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": loss_f,
                        }
                    )
                    if bool(cfg.use_wandb):
                        import wandb

                        lr_payload = {
                            f"train/lr_{g.get('group_name', '?')}": float(g["lr"])
                            for g in optimizer.param_groups
                        }
                        wandb.log(
                            {
                                "train/loss": loss_f,
                                "train/epoch": float(epoch),
                                **lr_payload,
                            },
                            step=int(global_step),
                        )

            print(f"[epoch_end] epoch={epoch} mean_loss={float(np.mean(epoch_losses)) if epoch_losses else 0:.6f}")

            if epoch % max(1, int(cfg.eval_every_epochs)) == 0 or epoch == 1:
                val_m = evaluate(
                    cfg=cfg,
                    region_cfg=region_cfg,
                    model=model,
                    clip=clip,
                    loader=val_loader,
                    device=device,
                    xyz_mean=xyz_mean_t,
                    xyz_std=xyz_std_t,
                    norm_eps=float(norm_eps_f),
                )
                print(
                    f"[eval] epoch={epoch} mean_l2={val_m['val_mean_l2_error_m']:.6f} "
                    f"constraint_sat={val_m['val_constraint_satisfaction_rate']:.4f}"
                )
                append_jsonl({"kind": "eval", "epoch": epoch, **val_m})
                if bool(cfg.use_wandb):
                    import wandb

                    _wandb_log_split_metrics(val_m, step=int(global_step), prefix="val")
                    if fixed_viz_idx:
                        model.eval()
                        clip.eval()
                        pair_imgs: List[Any] = []
                        panel_caps: List[str] = []
                        for vi in fixed_viz_idx:
                            s = val_samples[vi]
                            raw_one = val_ds[vi]
                            collated = no_image_collate([raw_one])
                            pred_w = _predict_world_one(
                                collated,
                                model=model,
                                clip=clip,
                                device=device,
                                cfg=cfg,
                                region_cfg=region_cfg,
                                xyz_mean=xyz_mean_t,
                                xyz_std=xyz_std_t,
                                norm_eps=float(norm_eps_f),
                            )
                            cap = (
                                f"{s.scene_id}|dp={s.datapoint_index}|iv={s.instruction_variant_index}|{s.movement_token}"
                            )
                            pil_all = render_all_keypoints_with_xyz(Path(s.shard_path))
                            fig = render_success_region_panel(
                                Path(s.shard_path),
                                s,
                                pred_w.detach().cpu().numpy(),
                                region_cfg,
                                sampled_goal_xyz_world=raw_one["goal_xyz_world"].detach().cpu().numpy(),
                                depth_eps_m=float(cfg.depth_occlusion_eps_m),
                            )
                            fig.canvas.draw()
                            w, h = fig.canvas.get_width_height()
                            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                            rgba = argb[:, :, [1, 2, 3, 0]]
                            sr_img = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
                            plt.close(fig)
                            pair_imgs.append(
                                compose_sample_comparison_pair(
                                    all_keypoint_image=pil_all,
                                    success_region_image=sr_img,
                                )
                            )
                            panel_caps.append(cap)
                        per_sample_payload: Dict[str, Any] = {}
                        for i in range(len(pair_imgs)):
                            key = f"qualitative/sample_pair_{i:02d}"
                            per_sample_payload[key] = wandb.Image(
                                pair_imgs[i],
                                caption=f"epoch={epoch} | {panel_caps[i]}",
                            )
                        wandb.log(
                            per_sample_payload,
                            step=int(global_step),
                        )
                if val_m["val_mean_l2_error_m"] < best_val:
                    best_val = float(val_m["val_mean_l2_error_m"])
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model": model.state_dict(),
                            "config": asdict(cfg),
                            "best_val_mean_l2_error_m": best_val,
                        },
                        out_dir / "checkpoint_best.pt",
                    )

            if epoch % max(1, int(cfg.checkpoint_every_epochs)) == 0 or epoch == 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                    },
                    out_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                )

        ckpt_best = out_dir / "checkpoint_best.pt"
        if ckpt_best.is_file():
            model.load_state_dict(torch.load(ckpt_best, map_location=device)["model"])
        test_m = evaluate(
            cfg=cfg,
            region_cfg=region_cfg,
            model=model,
            clip=clip,
            loader=test_loader,
            device=device,
            xyz_mean=xyz_mean_t,
            xyz_std=xyz_std_t,
            norm_eps=float(norm_eps_f),
        )
        final = {"best_val_mean_l2_error_m": float(best_val), **{f"test::{k}": v for k, v in test_m.items()}}
        (out_dir / "final_test_metrics.json").write_text(
            json.dumps(final, indent=2, default=str), encoding="utf-8"
        )
        print(f"[test] best_val={best_val:.6f} test_l2={test_m['val_mean_l2_error_m']:.6f}")
        if bool(cfg.use_wandb):
            import wandb

            _wandb_log_split_metrics(test_m, step=int(global_step), prefix="test")


def main() -> None:
    apply_hf_env()
    parser = argparse.ArgumentParser(description="Train image-free action expert.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cache_root = apply_hf_cache(str(cfg.hf_cache_dir))
    print(f"[hf] HF_HOME={cache_root}")
    train(cfg)


if __name__ == "__main__":
    main()
