"""Run the brush VLA on a single real-robot first frame and visualize the plan.

Builds one scene from explicit tool/material/destination world poses (real-robot
frame), shifts them into the model's table-centered training frame, converts the
brush root pose into the contact frame the model conditions on, runs flow-matching
inference for the 15-waypoint plan, and renders the trajectory panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_str_pipeline.sim_rollout.waypoint_to_pose import matrix_from_quat_xyzw
from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.config import load_config
from training.action_trajectory.dataset import (
    WaypointTrajectoryDataset,
    WaypointTrajectorySample,
    waypoint_collate,
)
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.text_encoder import ClipTextEncoder
from training.action_trajectory.train import _batch_to_model_inputs, _rollout


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _contact_frame_direct(xyz, quat_xyzw):
    """Contact frame read directly from a pose whose rotation columns are
    [surface_dir, normal x surface_dir, normal] (the dataset convention):
    surface_dir = R.x_hat, normal = R.z_hat. The provided real-robot brush pose
    is already in this near-flat convention (normal ~ +z)."""
    R = matrix_from_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    contact = np.asarray(xyz, dtype=np.float64).reshape(3)
    surface_dir = R[:, 0]
    normal = R[:, 2]
    return (
        contact.astype(np.float32),
        normal.astype(np.float32),
        surface_dir.astype(np.float32),
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/cfg/action_trajectory_brush_sweep_reactive.yaml")
    parser.add_argument("--checkpoint", type=str, default="training/runs/action_trajectory/run_0015/checkpoint_best.pt")
    parser.add_argument("--control_frame", type=str, default="generative_str_pipeline/assets/object_control_points/blue_brush.json")
    parser.add_argument("--instruction", type=str, default="Sweep the cube to the goal with the brush")
    parser.add_argument("--y_shift", type=float, default=0.8)
    parser.add_argument("--table_z", type=float, default=0.53)
    # Raw real-robot-frame poses (before y_shift).
    parser.add_argument("--tool_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--tool_quat_xyzw", type=float, nargs=4, required=True)
    parser.add_argument("--material_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--destination_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    apply_hf_env()
    cfg = load_config(_resolve(args.config))
    apply_hf_cache(str(cfg.hf_cache_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    shift = np.array([0.0, float(args.y_shift), 0.0], dtype=np.float64)
    tool_xyz = np.asarray(args.tool_xyz, dtype=np.float64) + shift
    material_xyz = np.asarray(args.material_xyz, dtype=np.float64) + shift
    destination_xyz = np.asarray(args.destination_xyz, dtype=np.float64) + shift

    contact, normal, surface_dir = _contact_frame_direct(tool_xyz, args.tool_quat_xyzw)
    print(f"[frame] tool_contact={contact}  normal={normal}  surface_dir={surface_dir}")
    print(f"[frame] material={material_xyz}  destination={destination_xyz}")

    mean_np, std_np, norm_eps = load_xyz_normalization_stats(_resolve(cfg.normalization_stats_path))
    xyz_mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    xyz_std = torch.tensor(std_np, dtype=torch.float32, device=device)

    sample = WaypointTrajectorySample(
        scene_id="firstframe",
        shard_path="",
        datapoint_index=0,
        movement_token="stroke_sweep",
        instruction=str(args.instruction),
        tool_label="the brush",
        tool_contact_xyz_world=contact,
        tool_current_normal=normal,
        tool_current_surface_dir=surface_dir,
        material_label="the cube",
        material_xyz_world=material_xyz.astype(np.float32),
        material_normal=None,
        has_material=True,
        destination_label="the goal",
        destination_xyz_world=destination_xyz.astype(np.float32),
        destination_normal=None,
        has_destination=True,
        table_label="table surface center",
        table_xyz_world=np.array([0.0, 0.0, float(args.table_z)], dtype=np.float32),
        table_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        waypoints=np.zeros((ActionTrajectoryModel.NUM_WAYPOINTS, 9), dtype=np.float32),
    )

    ds = WaypointTrajectoryDataset(
        [sample], xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
    )
    collated = waypoint_collate([ds[0]])

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
    c, nrm, sd = ActionTrajectoryModel.postprocess_waypoints(
        pred_norm.unsqueeze(0), xyz_mean, xyz_std, float(norm_eps)
    )
    waypoints = torch.cat([c, nrm, sd], dim=-1).reshape(-1, 9).cpu().numpy()
    print(f"[pred] waypoint[0] contact={waypoints[0,0:3]}")
    print(f"[pred] waypoint[-1] contact={waypoints[-1,0:3]}")

    from dataclasses import replace as _replace

    pred_sample = _replace(sample, waypoints=waypoints.astype(np.float32))

    from generative_str_pipeline.visualize_brush_trajectories import render_datapoint

    img = render_datapoint(pred_sample, "stroke_sweep")
    out_png = _resolve(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"Wrote {out_png}")

    if args.out_json.strip():
        out_json = _resolve(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "instruction": str(args.instruction),
                    "y_shift": float(args.y_shift),
                    "tool_contact_xyz_world": contact.tolist(),
                    "tool_current_normal": normal.tolist(),
                    "tool_current_surface_dir": surface_dir.tolist(),
                    "material_xyz_world": material_xyz.tolist(),
                    "destination_xyz_world": destination_xyz.tolist(),
                    "table_xyz_world": [0.0, 0.0, float(args.table_z)],
                    "waypoints": waypoints.tolist(),
                    "checkpoint": str(_resolve(args.checkpoint)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
