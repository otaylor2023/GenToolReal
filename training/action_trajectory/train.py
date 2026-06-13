"""Train waypoint-trajectory CLIP + self-attention flow-matching model."""

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

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.config import ActionTrajectoryConfig, load_config
from training.action_trajectory.dataset import (
    WaypointTrajectoryDataset,
    WaypointTrajectorySample,
    load_waypoint_samples,
    load_waypoint_samples_from_shards,
    split_sample_indices,
    waypoint_collate,
)
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.text_encoder import ClipTextEncoder
from training.action_trajectory.viz import (
    compose_gt_pred_pair,
    pick_fixed_viz_indices,
    pick_fixed_viz_indices_by_movement,
    render_reactive_rollout_video_for_sample,
    render_trajectory_panel,
    sample_with_waypoints,
)


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


def _assemble_shard_paths(cfg: ActionTrajectoryConfig) -> List[Path]:
    """Resolve shard paths from multi-dir or single-dir config."""
    dataset_dirs = [str(d).strip() for d in getattr(cfg, "dataset_dirs", []) if str(d).strip()]
    if dataset_dirs:
        caps = [int(x) for x in getattr(cfg, "max_shards_per_dir", [])]
        shard_paths: List[Path] = []
        for i, rel_dir in enumerate(dataset_dirs):
            dataset_dir = Path(rel_dir)
            if not dataset_dir.is_absolute():
                dataset_dir = (REPO_ROOT / dataset_dir).resolve()
            shards = sorted(dataset_dir.glob("*_shard.json"))
            if not shards:
                raise FileNotFoundError(f"No shard files in {dataset_dir}")
            cap = int(caps[i]) if i < len(caps) else 0
            if cap > 0:
                shards = shards[:cap]
            shard_paths.extend(shards)
        if not shard_paths:
            raise FileNotFoundError("No shard files resolved from dataset_dirs")
        return shard_paths

    dataset_dir = Path(cfg.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()

    max_shards = int(getattr(cfg, "max_shards", 0) or 0)
    if max_shards > 0:
        shard_paths = sorted(dataset_dir.glob("*_shard.json"))[:max_shards]
        if not shard_paths:
            raise FileNotFoundError(f"No shard files in {dataset_dir}")
        return shard_paths
    return [_resolve_shard_path(dataset_dir, str(cfg.shard_path))]


def _resolve_shard_path(dataset_dir: Path, shard_path: str) -> Path:
    if str(shard_path).strip():
        p = Path(shard_path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"shard_path not found: {p}")
        return p
    shards = sorted(dataset_dir.glob("*_shard.json"))
    if not shards:
        raise FileNotFoundError(f"No shard files in {dataset_dir}")
    return shards[0]


def _canonical_shard_paths(paths: List[Path]) -> list[str]:
    return sorted(str(p.resolve()) for p in paths)


def _assert_norm_stats_match_shards(stats_path: Path, shard_paths: List[Path]) -> None:
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
                f"normalization_stats_path {stats_path} was computed for different shards "
                f"than this run (expected {len(want)} paths, stats lists {len(got)})."
            )


@contextmanager
def _wandb_session(cfg: ActionTrajectoryConfig, *, run_id: str, out_dir: Path, scene_id: str):
    if not bool(cfg.use_wandb):
        yield
        return
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("use_wandb is true but wandb is not installed. pip install wandb") from exc
    explicit_name = str(cfg.wandb_run_name).strip()
    prefix = str(getattr(cfg, "wandb_run_prefix", "")).strip()
    if explicit_name:
        wandb_name = explicit_name
    elif prefix:
        idx = run_id.split("_", 1)[1] if "_" in run_id else run_id
        wandb_name = f"{prefix}_{idx}"
    else:
        wandb_name = run_id
    init_kw: Dict[str, Any] = {
        "project": str(cfg.wandb_project),
        "name": wandb_name,
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
    import wandb

    p = str(prefix).strip("/")
    flat: Dict[str, Any] = {
        f"{p}/trajectory_mse": metrics["val_trajectory_mse"],
        f"{p}/trajectory_rmse": metrics["val_trajectory_rmse"],
        f"{p}/contact_mean_l2_m": metrics["val_contact_mean_l2_m"],
        f"{p}/contact_max_l2_m": metrics.get("val_contact_max_l2_m", 0.0),
        f"{p}/normal_cosine_mean": metrics["val_normal_cosine_mean"],
        f"{p}/surface_dir_cosine_mean": metrics["val_surface_dir_cosine_mean"],
        f"{p}/success_rate": metrics.get("val_success_rate", 0.0),
        f"{p}/position_success_rate": metrics.get("val_position_success_rate", 0.0),
    }
    by_tok = metrics.get("val_by_movement_token") or {}
    if isinstance(by_tok, dict):
        for tok, stats in by_tok.items():
            if not isinstance(stats, dict):
                continue
            t = str(tok).replace("/", "_")
            for key in (
                "contact_mean_l2_m",
                "trajectory_rmse",
                "success_rate",
                "position_success_rate",
                "n",
            ):
                if key in stats:
                    flat[f"{p}/movement_token/{t}/{key}"] = stats[key]
    wandb.log(flat, step=int(step))


def _sample_flow_waypoints(x1: torch.Tensor) -> Dict[str, torch.Tensor]:
    x0 = torch.randn_like(x1)
    bsz = x1.shape[0]
    t = torch.rand(bsz, device=x1.device, dtype=torch.float32)
    xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    target_v = x1 - x0
    return {"x0": x0, "x1": x1, "t": t, "xt": xt, "target_v": target_v}


@torch.no_grad()
def _rollout(
    *,
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    steps: int,
    n_samples: int,
) -> torch.Tensor:
    device = batch_tensors["waypoints_norm"].device
    bsz = batch_tensors["waypoints_norm"].shape[0]
    dt = 1.0 / max(1, int(steps))
    finals: List[torch.Tensor] = []
    for _ in range(int(n_samples)):
        x = torch.randn(bsz, ActionTrajectoryModel.ACTION_DIM, device=device)
        for i in range(int(steps)):
            t = torch.full((bsz,), float(i) / float(max(1, steps)), device=device)
            pred = model(
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
            x = x + pred * dt
        finals.append(x)
    return torch.stack(finals, dim=1)


def _batch_to_model_inputs(
    batch: Dict[str, Any],
    clip: ClipTextEncoder,
    device: torch.device,
) -> Dict[str, Any]:
    bsz = len(batch["instruction_text"])
    instr_clip = clip.encode(batch["instruction_text"])
    tool_clip = clip.encode(batch["tool_label"])
    material_texts = [
        batch["material_label"][i] if bool(batch["has_material"][i].item()) else "none"
        for i in range(bsz)
    ]
    material_clip = clip.encode(material_texts)
    destination_texts = [
        batch["destination_label"][i] if bool(batch["has_destination"][i].item()) else "none"
        for i in range(bsz)
    ]
    destination_clip = clip.encode(destination_texts)
    tbl0 = batch["table_label"][0]
    table_clip = clip.encode([tbl0] * bsz)
    return {
        "instr_clip": instr_clip,
        "tool_clip": tool_clip,
        "material_clip": material_clip,
        "destination_clip": destination_clip,
        "table_clip": table_clip,
        "tool_contact_xyz_norm": batch["tool_contact_xyz_norm"].to(device),
        "tool_normal": batch["tool_normal"].to(device),
        "tool_surface_dir": batch["tool_surface_dir"].to(device),
        "material_xyz_norm": batch["material_xyz_norm"].to(device),
        "destination_xyz_norm": batch["destination_xyz_norm"].to(device),
        "table_xyz_norm": batch["table_xyz_norm"].to(device),
        "waypoints_norm": batch["waypoints_norm"].to(device),
        "has_material": batch["has_material"].to(device),
        "has_destination": batch["has_destination"].to(device),
    }


SUCCESS_POSITION_TOL_M = 0.03
SUCCESS_ANGLE_TOL_DEG = 20.0


def _trajectory_metrics_one(
    pred_wp: torch.Tensor,
    gt_wp: torch.Tensor,
    *,
    pos_tol_m: float = SUCCESS_POSITION_TOL_M,
    angle_tol_deg: float = SUCCESS_ANGLE_TOL_DEG,
) -> Dict[str, float]:
    """Compare flattened waypoint tensors (world space, xyz + dirs).

    Success requires every waypoint to satisfy:
      * contact position within `pos_tol_m` of GT,
      * normal angle within `angle_tol_deg` of GT,
      * surface_dir angle within `angle_tol_deg` of GT.
    """
    n_wp = int(pred_wp.numel() // ActionTrajectoryModel.WAYPOINT_DIM)
    pred = pred_wp.reshape(n_wp, ActionTrajectoryModel.WAYPOINT_DIM).float()
    gt = gt_wp.reshape(n_wp, ActionTrajectoryModel.WAYPOINT_DIM).float()
    mse = float(F.mse_loss(pred, gt).item())
    rmse = float(mse**0.5)
    contact_err = torch.linalg.norm(pred[:, 0:3] - gt[:, 0:3], dim=-1)
    contact_l2 = float(contact_err.mean().item())
    contact_max = float(contact_err.max().item())
    pred_n = F.normalize(pred[:, 3:6], dim=-1, eps=1e-6)
    gt_n = F.normalize(gt[:, 3:6], dim=-1, eps=1e-6)
    normal_cos_per_wp = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
    normal_cos = float(normal_cos_per_wp.mean().item())
    pred_sd = F.normalize(pred[:, 6:9], dim=-1, eps=1e-6)
    gt_sd = F.normalize(gt[:, 6:9], dim=-1, eps=1e-6)
    sd_cos_per_wp = (pred_sd * gt_sd).sum(dim=-1).clamp(-1.0, 1.0)
    sd_cos = float(sd_cos_per_wp.mean().item())

    cos_thresh = float(np.cos(np.deg2rad(float(angle_tol_deg))))
    position_ok = bool((contact_err <= float(pos_tol_m)).all().item())
    normal_ok = bool((normal_cos_per_wp >= cos_thresh).all().item())
    sd_ok = bool((sd_cos_per_wp >= cos_thresh).all().item())
    success = bool(position_ok and normal_ok and sd_ok)
    position_success = bool(position_ok)
    return {
        "trajectory_mse": mse,
        "trajectory_rmse": rmse,
        "contact_mean_l2_m": contact_l2,
        "contact_max_l2_m": contact_max,
        "normal_cosine_mean": normal_cos,
        "surface_dir_cosine_mean": sd_cos,
        "success": float(success),
        "position_success": float(position_success),
    }


def _aggregate_trajectory_metrics(rows: List[Dict[str, float]], tokens: List[str]) -> Dict[str, Any]:
    from collections import defaultdict

    if not rows:
        return {
            "val_trajectory_mse": 0.0,
            "val_trajectory_rmse": 0.0,
            "val_contact_mean_l2_m": 0.0,
            "val_contact_max_l2_m": 0.0,
            "val_normal_cosine_mean": 0.0,
            "val_surface_dir_cosine_mean": 0.0,
            "val_success_rate": 0.0,
            "val_position_success_rate": 0.0,
            "val_success_position_tol_m": SUCCESS_POSITION_TOL_M,
            "val_success_angle_tol_deg": SUCCESS_ANGLE_TOL_DEG,
            "val_by_movement_token": {},
        }
    keys = (
        "trajectory_mse",
        "trajectory_rmse",
        "contact_mean_l2_m",
        "contact_max_l2_m",
        "normal_cosine_mean",
        "surface_dir_cosine_mean",
        "success",
        "position_success",
    )
    arr = np.asarray([[r[k] for k in keys] for r in rows], dtype=np.float64)
    by_tok: dict[str, list[Dict[str, float]]] = defaultdict(list)
    for t, r in zip(tokens, rows):
        by_tok[str(t)].append(r)
    tok_out: Dict[str, Any] = {}
    for t in sorted(by_tok.keys()):
        sub = by_tok[t]
        tok_out[t] = {
            "trajectory_rmse": float(np.mean([x["trajectory_rmse"] for x in sub])),
            "contact_mean_l2_m": float(np.mean([x["contact_mean_l2_m"] for x in sub])),
            "contact_max_l2_m": float(np.mean([x["contact_max_l2_m"] for x in sub])),
            "normal_cosine_mean": float(np.mean([x["normal_cosine_mean"] for x in sub])),
            "surface_dir_cosine_mean": float(np.mean([x["surface_dir_cosine_mean"] for x in sub])),
            "success_rate": float(np.mean([x["success"] for x in sub])),
            "position_success_rate": float(np.mean([x["position_success"] for x in sub])),
            "n": int(len(sub)),
        }
    return {
        "val_trajectory_mse": float(np.mean(arr[:, 0])),
        "val_trajectory_rmse": float(np.mean(arr[:, 1])),
        "val_contact_mean_l2_m": float(np.mean(arr[:, 2])),
        "val_contact_max_l2_m": float(np.mean(arr[:, 3])),
        "val_normal_cosine_mean": float(np.mean(arr[:, 4])),
        "val_surface_dir_cosine_mean": float(np.mean(arr[:, 5])),
        "val_success_rate": float(np.mean(arr[:, 6])),
        "val_position_success_rate": float(np.mean(arr[:, 7])),
        "val_success_position_tol_m": SUCCESS_POSITION_TOL_M,
        "val_success_angle_tol_deg": SUCCESS_ANGLE_TOL_DEG,
        "val_by_movement_token": tok_out,
    }


@torch.no_grad()
def _predict_waypoints_one(
    collated: Dict[str, Any],
    *,
    model: ActionTrajectoryModel,
    clip: ClipTextEncoder,
    device: torch.device,
    cfg: ActionTrajectoryConfig,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    batch = {k: v for k, v in collated.items()}
    m = _batch_to_model_inputs(batch, clip, device)
    samples = _rollout(
        model=model,
        batch_tensors=m,
        steps=int(cfg.integration_steps),
        n_samples=int(cfg.inference_samples),
    )
    pred_norm = samples[0].mean(dim=0)
    contact, normal, surface_dir = ActionTrajectoryModel.postprocess_waypoints(
        pred_norm.unsqueeze(0),
        xyz_mean,
        xyz_std,
        norm_eps,
    )
    wp = torch.cat([contact, normal, surface_dir], dim=-1).reshape(-1)
    return wp


@torch.no_grad()
def evaluate(
    *,
    cfg: ActionTrajectoryConfig,
    model: ActionTrajectoryModel,
    clip: ClipTextEncoder,
    loader: DataLoader[Dict[str, Any]],
    device: torch.device,
    xyz_mean: torch.Tensor,
    xyz_std: torch.Tensor,
    norm_eps: float,
) -> Dict[str, Any]:
    model.eval()
    clip.eval()
    rows: List[Dict[str, float]] = []
    tokens: List[str] = []

    for raw in loader:
        batch = {k: v for k, v in raw.items()}
        gt_world = batch["waypoints_world"].to(device)
        m = _batch_to_model_inputs(batch, clip, device)
        samples = _rollout(
            model=model,
            batch_tensors=m,
            steps=int(cfg.integration_steps),
            n_samples=int(cfg.inference_samples),
        )
        bsz = gt_world.shape[0]
        for i in range(bsz):
            pred_norm = samples[i].mean(dim=0)
            contact, normal, surface_dir = ActionTrajectoryModel.postprocess_waypoints(
                pred_norm.unsqueeze(0),
                xyz_mean,
                xyz_std,
                norm_eps,
            )
            pred_wp = torch.cat([contact, normal, surface_dir], dim=-1).reshape(-1)
            row = _trajectory_metrics_one(pred_wp, gt_world[i])
            rows.append(row)
            tokens.append(str(batch["movement_token"][i]))

    model.train()
    return _aggregate_trajectory_metrics(rows, tokens)


def train(cfg: ActionTrajectoryConfig) -> None:
    _seed_everything(int(cfg.seed))
    device = torch.device(cfg.device)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"

    shard_paths = _assemble_shard_paths(cfg)

    scene_ids: List[str] = []
    for p in shard_paths:
        sj = json.loads(p.read_text(encoding="utf-8"))
        scene_ids.append(str(sj.get("scene_id") or sj.get("shard_id", "")))
    scene_id = scene_ids[0] if len(shard_paths) == 1 else f"multi{len(shard_paths)}"

    stats_path = Path(cfg.normalization_stats_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    if not stats_path.is_file():
        hint = (
            f"  python -m training.action_trajectory.compute_xyz_normalization "
            f"--shard_path {shard_paths[0]} --output {stats_path} --table_xyz "
            f"{cfg.table_xyz_world[0]} {cfg.table_xyz_world[1]} {cfg.table_xyz_world[2]}"
        )
        raise FileNotFoundError(f"Missing {stats_path}. Run:\n{hint}")
    _assert_norm_stats_match_shards(stats_path, shard_paths)

    xyz_mean_np, xyz_std_np, norm_eps_f = load_xyz_normalization_stats(stats_path)
    xyz_mean_t = torch.as_tensor(xyz_mean_np, dtype=torch.float32, device=device)
    xyz_std_t = torch.as_tensor(xyz_std_np, dtype=torch.float32, device=device)

    all_samples = (
        load_waypoint_samples_from_shards(shard_paths)
        if len(shard_paths) > 1
        else load_waypoint_samples(shard_paths[0])
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

    out_dir, run_id = _prepare_run_dir(Path(cfg.output_dir))
    print(
        f"[run] output_dir={out_dir} scene_id={scene_id} "
        f"n_shards={len(shard_paths)} n_samples={n}"
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

    train_ds = WaypointTrajectoryDataset(
        train_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
    )
    val_ds = WaypointTrajectoryDataset(
        val_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
    )
    test_ds = WaypointTrajectoryDataset(
        test_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
    )
    viz_ds = WaypointTrajectoryDataset(
        all_samples,
        xyz_mean=xyz_mean_np,
        xyz_std=xyz_std_np,
        norm_eps=float(norm_eps_f),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=waypoint_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=waypoint_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=waypoint_collate,
    )

    clip = ClipTextEncoder(
        model_id=str(cfg.clip_model_id),
        device=device,
        cache_dir=str(cfg.hf_cache_dir),
        local_files_only=bool(cfg.local_files_only),
    )
    model = ActionTrajectoryModel(
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
    best_val_rmse = float("inf")
    best_val_success = float("-inf")
    metrics_path = out_dir / "train_metrics.jsonl"
    t0 = time.perf_counter()

    def append_jsonl(row: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    requested_viz = max(0, int(cfg.wandb_prediction_num_examples))
    if requested_viz >= 10:
        fixed_viz_idx = pick_fixed_viz_indices_by_movement(
            all_samples,
            per_movement=2,
            seed=int(cfg.seed),
        )
        fixed_viz_samples = all_samples
        fixed_viz_ds = viz_ds
    else:
        fixed_viz_idx = pick_fixed_viz_indices(
            val_samples,
            n=min(requested_viz, len(val_samples)),
            seed=int(cfg.seed),
        )
        fixed_viz_samples = val_samples
        fixed_viz_ds = val_ds

    with _wandb_session(cfg, run_id=run_id, out_dir=out_dir, scene_id=scene_id):
        clip.eval()
        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
            epoch_losses: List[float] = []
            for _batch_idx, raw in enumerate(train_loader, start=1):
                global_step += 1
                targets = raw["waypoints_norm"].to(device)
                flow = _sample_flow_waypoints(targets)
                m = _batch_to_model_inputs(raw, clip, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=autocast_device, enabled=use_amp, dtype=amp_dtype):
                    pred = model(
                        instr_clip=m["instr_clip"],
                        tool_clip=m["tool_clip"],
                        material_clip=m["material_clip"],
                        destination_clip=m["destination_clip"],
                        table_clip=m["table_clip"],
                        tool_contact_xyz_norm=m["tool_contact_xyz_norm"],
                        tool_normal=m["tool_normal"],
                        tool_surface_dir=m["tool_surface_dir"],
                        material_xyz_norm=m["material_xyz_norm"],
                        destination_xyz_norm=m["destination_xyz_norm"],
                        table_xyz_norm=m["table_xyz_norm"],
                        xt=flow["xt"],
                        t=flow["t"],
                        has_material=m["has_material"],
                        has_destination=m["has_destination"],
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
                    model=model,
                    clip=clip,
                    loader=val_loader,
                    device=device,
                    xyz_mean=xyz_mean_t,
                    xyz_std=xyz_std_t,
                    norm_eps=float(norm_eps_f),
                )
                print(
                    f"[eval] epoch={epoch} success={val_m['val_success_rate']:.3f} "
                    f"pos_success={val_m['val_position_success_rate']:.3f} "
                    f"contact_l2={val_m['val_contact_mean_l2_m']:.4f} "
                    f"traj_rmse={val_m['val_trajectory_rmse']:.4f} "
                    f"normal_cos={val_m['val_normal_cosine_mean']:.3f}"
                )
                append_jsonl({"kind": "eval", "epoch": epoch, **val_m})
                if bool(cfg.use_wandb):
                    import wandb

                    _wandb_log_split_metrics(val_m, step=int(global_step), prefix="val")
                    if fixed_viz_idx:
                        model.eval()
                        clip.eval()
                        viz_payload: Dict[str, Any] = {}
                        rollout_video_dir = out_dir / "wandb_reactive_rollout_videos"
                        for i, vi in enumerate(fixed_viz_idx):
                            s: WaypointTrajectorySample = fixed_viz_samples[vi]
                            raw_one = fixed_viz_ds[vi]
                            collated = waypoint_collate([raw_one])
                            pred_wp = _predict_waypoints_one(
                                collated,
                                model=model,
                                clip=clip,
                                device=device,
                                cfg=cfg,
                                xyz_mean=xyz_mean_t,
                                xyz_std=xyz_std_t,
                                norm_eps=float(norm_eps_f),
                            )
                            gt_wp_t = raw_one["waypoints_world"].to(device)
                            row = _trajectory_metrics_one(pred_wp, gt_wp_t)
                            pred_sample = sample_with_waypoints(s, pred_wp.detach().cpu().numpy())
                            mtoken = str(s.movement_token)
                            gt_img = render_trajectory_panel(s, mtoken)
                            pred_img = render_trajectory_panel(pred_sample, mtoken)
                            pair = compose_gt_pred_pair(gt_img, pred_img)
                            ok = bool(row["success"] > 0.5)
                            cap = (
                                f"epoch={epoch} | {mtoken} dp={s.datapoint_index} | "
                                f"success={'YES' if ok else 'no'} | "
                                f"contact_l2={row['contact_mean_l2_m']:.3f}m | "
                                f'"{s.instruction[:60]}"'
                            )
                            viz_payload[f"qualitative/gt_pred_pair_{i:02d}"] = wandb.Image(pair, caption=cap)
                            video_path = render_reactive_rollout_video_for_sample(
                                s,
                                out_dir=rollout_video_dir / f"epoch_{epoch:04d}",
                                chunk_size=5,
                                fps=10.0,
                            )
                            if video_path is not None:
                                viz_payload[
                                    f"qualitative/reactive_rollout_video_{i:02d}"
                                ] = wandb.Video(str(video_path), fps=10, format="mp4", caption=cap)
                        wandb.log(viz_payload, step=int(global_step))

                cur_success = float(val_m["val_success_rate"])
                cur_rmse = float(val_m["val_trajectory_rmse"])
                is_better = (cur_success > best_val_success) or (
                    cur_success == best_val_success and cur_rmse < best_val_rmse
                )
                if is_better:
                    best_val_success = cur_success
                    best_val_rmse = cur_rmse
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model": model.state_dict(),
                            "config": asdict(cfg),
                            "best_val_success_rate": best_val_success,
                            "best_val_trajectory_rmse": best_val_rmse,
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
            model=model,
            clip=clip,
            loader=test_loader,
            device=device,
            xyz_mean=xyz_mean_t,
            xyz_std=xyz_std_t,
            norm_eps=float(norm_eps_f),
        )
        final = {
            "best_val_success_rate": float(best_val_success),
            "best_val_trajectory_rmse": float(best_val_rmse),
            **{f"test::{k}": v for k, v in test_m.items()},
        }
        (out_dir / "final_test_metrics.json").write_text(
            json.dumps(final, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"[test] best_val_success={best_val_success:.3f} best_val_rmse={best_val_rmse:.4f} "
            f"test_success={test_m['val_success_rate']:.3f} "
            f"test_contact_l2={test_m['val_contact_mean_l2_m']:.4f}"
        )
        if bool(cfg.use_wandb):
            import wandb

            _wandb_log_split_metrics(test_m, step=int(global_step), prefix="test")


def main() -> None:
    apply_hf_env()
    parser = argparse.ArgumentParser(description="Train waypoint trajectory action model.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cache_root = apply_hf_cache(str(cfg.hf_cache_dir))
    print(f"[hf] HF_HOME={cache_root}")
    train(cfg)


if __name__ == "__main__":
    main()
