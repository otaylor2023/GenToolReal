"""Behavior cloning trainer for tool-position prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.bc.losses import RegionLossConfig, canonical_mse_loss, region_aware_loss
from training.bc.model import ToolPositionBCRegressor
from training.bc.tool_position_bc_data import (
    ToolPositionBCDataset,
    load_episode_samples_from_shards,
    split_shards_85_10_5,
)


@dataclass
class BCConfig:
    dataset_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    head_lr: float
    backbone_lr: float
    warmstart_mse_epochs: int
    eval_every_epochs: int
    checkpoint_every_epochs: int
    log_every_steps: int
    device: str
    seed: int = 7
    num_workers: int = 4
    use_amp: bool = True
    image_size: int = 224
    run_tag: str = ""
    train_fraction: float = 0.85
    val_fraction: float = 0.10
    use_real_qwen: bool = True
    vl_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    qwen_local_files_only: bool = False
    qwen_forward_chunk_size: int = 0
    unfreeze_last_n_layers: int = 8
    enable_gradient_checkpointing: bool = True
    mlp_hidden_dims: List[int] | Tuple[int, ...] = (1024, 512, 256)
    mlp_dropout: float = 0.1
    task_prompt_template: str = (
        "You are a robot tool-positioning policy. Your goal is to output the absolute "
        "3D target position (x, y, z) in meters where the tool should move, based on "
        "the instruction. Movement is defined using the specified tool keypoint. You "
        "are given the scene image and a set of scene keypoints with coordinates to "
        "help you understand where objects are located. Return only the target position."
    )
    world_scale_prompt_template: str = (
        "Coordinate system: canonical tabletop frame in meters. Do not assume a fixed absolute "
        "tabletop z value; infer scene geometry from the provided keypoints and coordinates. "
        "Axis directions are fixed: +X points to the right relative to the camera, +Y points away "
        "from the camera, and +Z points upward. All scene keypoint coordinates and the tool "
        "keypoint position are provided in this same frame."
    )
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


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    image = batch["image"].to(device=device, dtype=torch.float32)
    goal = batch["goal_xyz_world"].to(device=device, dtype=torch.float32)
    ref = batch["reference_xyz_world"].to(device=device, dtype=torch.float32)
    ref2 = batch["secondary_reference_xyz_world"].to(device=device, dtype=torch.float32)
    has_ref2 = batch["has_secondary_reference"]
    if torch.is_tensor(has_ref2):
        has_ref2 = has_ref2.tolist()
    return {
        "obs": {
            "image": image,
            "text_context": list(batch["text_context"]),
        },
        "goal_xyz_world": goal,
        "movement_token": list(batch["movement_token"]),
        "constraint_type": list(batch["constraint_type"]),
        "reference_xyz_world": ref,
        "secondary_reference_xyz_world": ref2,
        "has_secondary_reference": [bool(x) for x in has_ref2],
        "constraint_params": list(batch["constraint_params"]),
    }


def _bc_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate to keep variable-keys metadata as plain Python lists."""
    collated = {
        "image": default_collate([b["image"] for b in batch]),
        "text_context": [b["text_context"] for b in batch],
        "goal_xyz_world": default_collate([b["goal_xyz_world"] for b in batch]),
        "movement_token": [b["movement_token"] for b in batch],
        "constraint_type": [b["constraint_type"] for b in batch],
        "reference_xyz_world": default_collate([b["reference_xyz_world"] for b in batch]),
        "secondary_reference_xyz_world": default_collate(
            [b["secondary_reference_xyz_world"] for b in batch]
        ),
        "has_secondary_reference": [b["has_secondary_reference"] for b in batch],
        "constraint_params": [b["constraint_params"] for b in batch],
    }
    return collated


@torch.no_grad()
def evaluate(
    model: ToolPositionBCRegressor,
    loader: DataLoader[Dict[str, Any]],
    device: torch.device,
    region_cfg: RegionLossConfig,
) -> Dict[str, float]:
    def _constraint_bucket(token: str, ctype: str) -> str:
        token_l = str(token or "").strip().lower()
        ctype_l = str(ctype or "").strip().lower()
        if token_l == "exact_above" or "exact_above" in ctype_l:
            return "exact_above"
        if token_l in {"left", "right", "in_front", "behind"} or "directional" in ctype_l:
            return "directional"
        if token_l == "between" or "between" in ctype_l:
            return "between"
        if token_l in {"near", "next_to"}:
            return "near_next_to"
        if token_l in {"over", "above"} or "above_v0" in ctype_l:
            return "over_above"
        return "other"

    model.eval()
    all_l2: List[float] = []
    all_mse: List[float] = []
    all_region: List[float] = []
    counts: Dict[str, int] = {}
    sat: Dict[str, int] = {}
    grouped_l2: Dict[str, List[float]] = {
        "exact_above": [],
        "directional": [],
        "between": [],
        "near_next_to": [],
        "over_above": [],
    }

    for raw in loader:
        batch = _batch_to_device(raw, device)
        pred = model(batch["obs"])
        goal = batch["goal_xyz_world"]
        all_l2.extend(torch.linalg.norm(pred - goal, dim=-1).detach().cpu().tolist())
        all_mse.append(float(canonical_mse_loss(pred, goal).detach().cpu().item()))
        region = region_aware_loss(
            pred,
            goal,
            batch["movement_token"],
            batch["constraint_type"],
            batch["reference_xyz_world"],
            batch["secondary_reference_xyz_world"],
            batch["has_secondary_reference"],
            batch["constraint_params"],
            region_cfg,
        )
        all_region.append(float(region.detach().cpu().item()))
        for token, ctype in zip(batch["movement_token"], batch["constraint_type"]):
            key = str(ctype or token or "unknown")
            counts[key] = counts.get(key, 0) + 1
        per_sample_l2 = torch.linalg.norm(pred - goal, dim=-1)
        ok = per_sample_l2 <= 0.02
        for idx, (token, ctype) in enumerate(zip(batch["movement_token"], batch["constraint_type"])):
            key = str(ctype or token or "unknown")
            sat[key] = sat.get(key, 0) + int(ok[idx].item())
            bucket = _constraint_bucket(token, ctype)
            if bucket in grouped_l2:
                grouped_l2[bucket].append(float(per_sample_l2[idx].detach().cpu().item()))

    out: Dict[str, float] = {
        "val_mean_l2_error_m": float(np.mean(all_l2)) if all_l2 else 0.0,
        "val_mse_loss": float(np.mean(all_mse)) if all_mse else 0.0,
        "val_region_loss": float(np.mean(all_region)) if all_region else 0.0,
    }
    for key in sorted(counts.keys()):
        out[f"val_sat_rate::{key}"] = float(sat.get(key, 0) / max(1, counts[key]))
    for bucket in ["exact_above", "directional", "between", "near_next_to", "over_above"]:
        vals = grouped_l2[bucket]
        out[f"val_l2::{bucket}"] = float(np.mean(vals)) if vals else 0.0
    model.train()
    return out


def train(cfg: BCConfig) -> None:
    _seed_everything(int(cfg.seed))
    device = torch.device(cfg.device)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    output_root = Path(cfg.output_dir)
    out_dir, run_id = _prepare_run_dir(output_root, cfg.run_tag)
    print(f"[run] output_dir={out_dir}")

    splits = split_shards_85_10_5(
        Path(cfg.dataset_dir),
        seed=int(cfg.seed),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
    )
    train_samples = load_episode_samples_from_shards(
        splits["train"],
        task_prompt=cfg.task_prompt_template,
        world_scale_prompt=cfg.world_scale_prompt_template,
    )
    val_samples = load_episode_samples_from_shards(
        splits["val"],
        task_prompt=cfg.task_prompt_template,
        world_scale_prompt=cfg.world_scale_prompt_template,
    )
    test_samples = load_episode_samples_from_shards(
        splits["test"],
        task_prompt=cfg.task_prompt_template,
        world_scale_prompt=cfg.world_scale_prompt_template,
    )

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
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_path = out_dir / "progress.json"

    train_ds = ToolPositionBCDataset(train_samples, image_size=(cfg.image_size, cfg.image_size))
    val_ds = ToolPositionBCDataset(val_samples, image_size=(cfg.image_size, cfg.image_size))
    test_ds = ToolPositionBCDataset(test_samples, image_size=(cfg.image_size, cfg.image_size))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_bc_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=_bc_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers) // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=_bc_collate,
    )

    model = ToolPositionBCRegressor(
        use_real_qwen=bool(cfg.use_real_qwen),
        vl_model_id=str(cfg.vl_model_id),
        device=device,
        hf_cache_dir=str(cfg.hf_cache_dir),
        qwen_local_files_only=bool(cfg.qwen_local_files_only),
        qwen_forward_chunk_size=int(cfg.qwen_forward_chunk_size),
        unfreeze_last_n_layers=int(cfg.unfreeze_last_n_layers),
        enable_gradient_checkpointing=bool(cfg.enable_gradient_checkpointing),
        hidden_dims=tuple(int(x) for x in cfg.mlp_hidden_dims),
        dropout=float(cfg.mlp_dropout),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    any_bf16 = any(p.dtype == torch.bfloat16 for p in trainable_params)
    amp_dtype = torch.bfloat16 if any_bf16 else torch.float16
    head_params = [p for p in model.xyz_head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone_trainable_parameters() if p.requires_grad]
    param_groups: List[Dict[str, Any]] = []
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(cfg.head_lr),
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(cfg.backbone_lr),
                "weight_decay": float(cfg.weight_decay),
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer")
    optimizer = optim.AdamW(param_groups)
    print(
        "[init] "
        f"device={device} use_amp={use_amp} amp_dtype={amp_dtype} "
        f"gradient_checkpointing={bool(cfg.enable_gradient_checkpointing)} "
        f"splits(shards)={metadata['split_counts_shards']} "
        f"splits(samples)={metadata['split_counts_samples']} "
        f"batches(train/val/test)=({len(train_loader)}/{len(val_loader)}/{len(test_loader)}) "
        f"params(trainable/total)=({trainable_param_count}/{total_params}) "
        f"param_groups={len(param_groups)}"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not any_bf16))
    region_cfg = RegionLossConfig(
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

    metrics: List[Dict[str, Any]] = []
    best_val = float("inf")
    t0 = time.perf_counter()

    def _checkpoint_payload(epoch: int, best_val_metric: float) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "run_id": run_id,
            "config": asdict(cfg),
            "region_loss_config": asdict(region_cfg),
            "model_trainable_state": {
                "xyz_head": model.xyz_head.state_dict(),
            },
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "best_val_mean_l2_error_m": float(best_val_metric),
            "frozen_backbone": {
                "use_real_qwen": bool(cfg.use_real_qwen),
                "vl_model_id": str(cfg.vl_model_id),
                "hf_cache_dir": str(cfg.hf_cache_dir),
            },
        }

    global_step = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        batch_losses: List[float] = []
        phase = "mse" if epoch <= int(cfg.warmstart_mse_epochs) else "region"
        print(
            f"[epoch_start] epoch={epoch}/{cfg.epochs} phase={phase} "
            f"num_train_batches={len(train_loader)}"
        )

        for batch_idx, raw in enumerate(train_loader, start=1):
            global_step += 1
            batch = _batch_to_device(raw, device)
            goal = batch["goal_xyz_world"]
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(batch["obs"])
                if phase == "mse":
                    loss = canonical_mse_loss(pred, goal)
                else:
                    loss = region_aware_loss(
                        pred,
                        goal,
                        batch["movement_token"],
                        batch["constraint_type"],
                        batch["reference_xyz_world"],
                        batch["secondary_reference_xyz_world"],
                        batch["has_secondary_reference"],
                        batch["constraint_params"],
                        region_cfg,
                    )
            optimizer.zero_grad(set_to_none=True)
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
            batch_losses.append(float(loss.detach().cpu().item()))
            if batch_idx == 1:
                print(
                    f"[train_first_step] epoch={epoch} step={global_step} "
                    f"phase={phase} loss={batch_losses[-1]:.6f}"
                )

            if global_step % max(1, int(cfg.log_every_steps)) == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} phase={phase} "
                    f"loss={batch_losses[-1]:.6f}"
                )

        heartbeat = {
            "epoch": int(epoch),
            "phase": phase,
            "elapsed_seconds": float(time.perf_counter() - t0),
            "train_loss_mean": float(np.mean(batch_losses)) if batch_losses else 0.0,
        }
        progress_path.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")

        should_eval = (epoch % max(1, int(cfg.eval_every_epochs)) == 0) or epoch == 1
        if should_eval:
            val_metrics = evaluate(model, val_loader, device, region_cfg)
            row = {"epoch": int(epoch), "phase": phase, **heartbeat, **val_metrics}
            metrics.append(row)
            print(
                f"[eval] epoch={epoch} phase={phase} "
                f"val_l2={row['val_mean_l2_error_m']:.6f} "
                f"val_mse={row['val_mse_loss']:.6f} "
                f"val_region={row['val_region_loss']:.6f} "
                f"exact_above={row['val_l2::exact_above']:.6f} "
                f"directional={row['val_l2::directional']:.6f} "
                f"between={row['val_l2::between']:.6f} "
                f"near_next_to={row['val_l2::near_next_to']:.6f} "
                f"over_above={row['val_l2::over_above']:.6f}"
            )
            if row["val_mean_l2_error_m"] < best_val:
                best_val = float(row["val_mean_l2_error_m"])
                torch.save(_checkpoint_payload(epoch, best_val), out_dir / "checkpoint_best.pt")
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if epoch % max(1, int(cfg.checkpoint_every_epochs)) == 0 or epoch == 1:
            torch.save(_checkpoint_payload(epoch, best_val), out_dir / f"checkpoint_{epoch:06d}.pt")

    if (out_dir / "checkpoint_best.pt").exists():
        ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device)
        model.xyz_head.load_state_dict(ckpt["model_trainable_state"]["xyz_head"])
    test_metrics = evaluate(model, test_loader, device, region_cfg)
    final = {
        "best_val_mean_l2_error_m": float(best_val),
        **{f"test::{k}": v for k, v in test_metrics.items()},
    }
    (out_dir / "final_test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(
        f"[test] best_val={final['best_val_mean_l2_error_m']:.6f} "
        f"test_l2={final['test::val_mean_l2_error_m']:.6f}"
    )


def _load_config(path: Path) -> BCConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BCConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tool-position behavior cloning module.")
    parser.add_argument("--config", type=Path, required=True, help="Path to tool_position_bc.yaml")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

