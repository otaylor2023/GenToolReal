"""Offline evaluation entrypoint for action-expert checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.config import load_config
from training.action_expert.dataset import (
    ActionExpertDataset,
    action_expert_collate,
    load_action_samples_from_shards,
    split_shards_85_10_5,
)
from training.action_expert.losses import RegionConstraintConfig
from training.action_expert.model import ActionExpertModel
from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.train_action_expert import _region_cfg, evaluate
from training.action_expert.vlm import PaliGemmaContextEncoder
from training.action_expert.xyz_normalization import load_xyz_normalization_stats


def main() -> None:
    apply_hf_env()
    parser = argparse.ArgumentParser(description="Evaluate action expert checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_hf_cache(str(cfg.hf_cache_dir))
    region_cfg: RegionConstraintConfig = _region_cfg(cfg)
    device = torch.device(cfg.device)
    stats_path = Path(cfg.normalization_stats_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    xyz_mean_np, xyz_std_np, norm_eps_f = load_xyz_normalization_stats(stats_path)
    xyz_mean_t = torch.as_tensor(xyz_mean_np, dtype=torch.float32, device=device)
    xyz_std_t = torch.as_tensor(xyz_std_np, dtype=torch.float32, device=device)
    splits = split_shards_85_10_5(
        Path(cfg.dataset_dir),
        seed=int(cfg.seed),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
    )
    test_samples = load_action_samples_from_shards(
        splits["test"],
        max_keypoints=int(cfg.max_keypoints),
        region_cfg=region_cfg,
        explode_instruction_variants=bool(cfg.explode_instruction_variants),
    )
    test_loader = DataLoader(
        ActionExpertDataset(
            test_samples,
            image_size=(cfg.image_size, cfg.image_size),
            xyz_mean=xyz_mean_np,
            xyz_std=xyz_std_np,
            norm_eps=float(norm_eps_f),
            region_cfg=region_cfg,
            sample_goal_in_constraint_region=bool(cfg.sample_goal_in_constraint_region),
            goal_rejection_sample_max_attempts=int(cfg.goal_rejection_sample_max_attempts),
        ),
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
    action_model = ActionExpertModel(
        d_model=int(vlm.d_model),
        num_heads=int(cfg.num_heads),
        num_layers=int(cfg.num_action_expert_layers),
        dropout=float(cfg.action_dropout),
        ffn_multiplier=int(cfg.ffn_multiplier),
        pos_norm_denom=float(cfg.pos_norm_denom),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    action_model.load_state_dict(ckpt["model_trainable_state"]["action_expert"])
    vlm.load_state_dict(ckpt["model_trainable_state"]["vlm_lora"], strict=False)

    metrics = evaluate(
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

