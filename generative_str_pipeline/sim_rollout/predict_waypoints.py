"""Run the trained action_trajectory VLA to predict 6x9 waypoints for one scene.

Loads a scene (tool pose + material/destination/table xyz + instruction) from a
procedural shard datapoint, runs flow-matching inference, and writes a
waypoints JSON consumable by ``build_brush_trajectory`` (predicted waypoints +
the tool home pose used to seed ``start_pose``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.config import load_config
from training.action_trajectory.dataset import (
    WaypointTrajectoryDataset,
    load_waypoint_samples,
    waypoint_collate,
)
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.text_encoder import ClipTextEncoder
from training.action_trajectory.train import _batch_to_model_inputs, _rollout


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict VLA waypoints for one scene.")
    parser.add_argument("--config", type=str, default="training/cfg/action_trajectory_brush_sweep_diverse.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/runs/action_trajectory/run_0007/checkpoint_epoch_0030.pt",
    )
    parser.add_argument("--shard_path", type=str, default="training/datasets/dataset_0009_brush_sweep_diverse/shards/brush_sweep_diverse_0000_shard.json")
    parser.add_argument("--datapoint_index", type=int, default=0)
    parser.add_argument("--instruction", type=str, default="", help="Optional instruction override")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the flow-matching noise for reproducible / comparable samples.",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    apply_hf_env()
    cfg = load_config(_resolve(args.config))
    apply_hf_cache(str(cfg.hf_cache_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    mean_np, std_np, norm_eps = load_xyz_normalization_stats(_resolve(cfg.normalization_stats_path))
    xyz_mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    xyz_std = torch.tensor(std_np, dtype=torch.float32, device=device)

    # Build scene sample from a shard datapoint
    samples = load_waypoint_samples(_resolve(args.shard_path))
    sample = next(s for s in samples if int(s.datapoint_index) == int(args.datapoint_index))
    ds = WaypointTrajectoryDataset(
        [sample], xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
    )
    item = ds[0]
    if args.instruction.strip():
        item["instruction_text"] = args.instruction.strip()
    collated = waypoint_collate([item])

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
    ckpt = torch.load(_resolve(args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    clip.eval()

    m = _batch_to_model_inputs(collated, clip, device)
    samples_out = _rollout(
        model=model,
        batch_tensors=m,
        steps=int(cfg.integration_steps),
        n_samples=int(cfg.inference_samples),
    )
    pred_norm = samples_out[0].mean(dim=0)
    contact, normal, surface_dir = ActionTrajectoryModel.postprocess_waypoints(
        pred_norm.unsqueeze(0), xyz_mean, xyz_std, float(norm_eps)
    )
    waypoints = torch.cat([contact, normal, surface_dir], dim=-1).reshape(-1, 9).cpu().numpy()

    out: Dict[str, Any] = {
        "instruction": item["instruction_text"],
        "waypoints": waypoints.tolist(),
        "tool_contact_xyz_world": np.asarray(sample.tool_contact_xyz_world, dtype=float).tolist(),
        "tool_current_normal": np.asarray(sample.tool_current_normal, dtype=float).tolist(),
        "tool_current_surface_dir": np.asarray(sample.tool_current_surface_dir, dtype=float).tolist(),
        "material_xyz_world": np.asarray(sample.material_xyz_world, dtype=float).tolist(),
        "destination_xyz_world": np.asarray(sample.destination_xyz_world, dtype=float).tolist(),
        "table_xyz_world": np.asarray(sample.table_xyz_world, dtype=float).tolist(),
        "_meta": {
            "checkpoint": str(_resolve(args.checkpoint)),
            "datapoint_index": int(args.datapoint_index),
            "inference_samples": int(cfg.inference_samples),
            "integration_steps": int(cfg.integration_steps),
            "seed": args.seed,
        },
    }
    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  instruction: {item['instruction_text']}")
    print(f"  contact[0]: {waypoints[0, 0:3]}")
    print(f"  contact[-1]: {waypoints[5, 0:3]}")


if __name__ == "__main__":
    main()
