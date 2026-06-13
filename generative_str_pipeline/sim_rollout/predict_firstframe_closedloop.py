"""Model-in-the-loop closed-loop (receding-horizon) rollout from one real frame.

Starting from an explicit first-frame scene (brush contact + cube + goal in the
robot frame, shifted into the model's table-centered frame), this runs the brush
VLA in a receding-horizon loop: predict 15 waypoints, execute the first
``chunk`` analytically (cube pushed by the same analytic model the data was
trained on), re-observe, replan -- until the cube reaches the goal. The per-
generation plans + executed object motion are rendered with the standard
verification renderer (side-by-side executed | full plan) and saved under
training/verification/.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_str_pipeline.build_dataset_0011_brush_sweep_reactive import (
    ReactiveGenConfig,
    _execute_chunk,
)
from generative_str_pipeline.render_reactive_rollout_viz import (
    GenerationMeta,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    render_executed_video,
)
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
    R = matrix_from_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    return (
        np.asarray(xyz, dtype=np.float64).reshape(3).astype(np.float32),
        R[:, 2].astype(np.float32),  # normal
        R[:, 0].astype(np.float32),  # surface_dir
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/cfg/action_trajectory_brush_sweep_reactive.yaml")
    parser.add_argument("--checkpoint", type=str, default="training/runs/action_trajectory/run_0015/checkpoint_best.pt")
    parser.add_argument("--instruction", type=str, default="Sweep the cube to the goal with the brush")
    parser.add_argument("--y_shift", type=float, default=0.8)
    parser.add_argument("--table_z", type=float, default=0.53)
    parser.add_argument("--tool_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--tool_quat_xyzw", type=float, nargs=4, required=True)
    parser.add_argument("--material_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--destination_xyz", type=float, nargs=3, required=True)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--max_replans", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_subdir", type=str, default="firstframe_brush_closedloop")
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
    material_xyz = (np.asarray(args.material_xyz, dtype=np.float64) + shift).astype(np.float32)
    destination_xyz = (np.asarray(args.destination_xyz, dtype=np.float64) + shift).astype(np.float32)
    table_z = float(args.table_z)

    contact, normal, surface_dir = _contact_frame_direct(tool_xyz, args.tool_quat_xyzw)

    mean_np, std_np, norm_eps = load_xyz_normalization_stats(_resolve(cfg.normalization_stats_path))
    xyz_mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    xyz_std = torch.tensor(std_np, dtype=torch.float32, device=device)

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

    def _make_sample(tc, tn, tsd, mat) -> WaypointTrajectorySample:
        return WaypointTrajectorySample(
            scene_id="firstframe",
            shard_path="",
            datapoint_index=0,
            movement_token="stroke_sweep",
            instruction=str(args.instruction),
            tool_label="the brush",
            tool_contact_xyz_world=np.asarray(tc, dtype=np.float32),
            tool_current_normal=np.asarray(tn, dtype=np.float32),
            tool_current_surface_dir=np.asarray(tsd, dtype=np.float32),
            material_label="the cube",
            material_xyz_world=np.asarray(mat, dtype=np.float32),
            material_normal=None,
            has_material=True,
            destination_label="the goal",
            destination_xyz_world=destination_xyz,
            destination_normal=None,
            has_destination=True,
            table_label="table surface center",
            table_xyz_world=np.array([0.0, 0.0, table_z], dtype=np.float32),
            table_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            waypoints=np.zeros((ActionTrajectoryModel.NUM_WAYPOINTS, 9), dtype=np.float32),
        )

    def _predict(sample: WaypointTrajectorySample) -> np.ndarray:
        ds = WaypointTrajectoryDataset(
            [sample], xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
        )
        collated = waypoint_collate([ds[0]])
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
        return torch.cat([c, nrm, sd], dim=-1).reshape(-1, 9).cpu().numpy().astype(np.float32)

    gcfg = ReactiveGenConfig(table_xyz_world=[0.0, 0.0, table_z])
    surface_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    chunk = max(1, int(args.chunk))

    brush = contact.copy()
    brush_n = normal.copy()
    brush_sd = surface_dir.copy()
    obj = material_xyz.copy()
    in_contact = False

    generations: list[tuple[WaypointTrajectorySample, GenerationMeta]] = []
    for gen in range(int(args.max_replans)):
        sample = _make_sample(brush, brush_n, brush_sd, obj)
        target = _predict(sample)
        sample = dc_replace(sample, waypoints=target)

        new_brush, new_obj, new_contact, material_trace = _execute_chunk(
            gcfg,
            target,
            brush_xyz=brush,
            object_xyz=obj,
            in_contact=in_contact,
            destination_xyz=destination_xyz,
            surface_normal=surface_normal,
            chunk=chunk,
        )

        generations.append(
            (
                sample,
                GenerationMeta(
                    scene_index=0,
                    window_index=gen,
                    rollout_step=gen * chunk,
                    material_after_xyz=np.asarray(new_obj, dtype=np.float64).reshape(3),
                    material_trace_xyz=np.asarray(material_trace, dtype=np.float64).reshape(-1, 3),
                ),
            )
        )

        reached = float(np.linalg.norm(new_obj[:2] - destination_xyz[:2])) <= gcfg.goal_region_radius_m
        print(
            f"[gen {gen}] brush=({brush[0]:+.3f},{brush[1]:+.3f},{brush[2]:+.3f}) "
            f"obj=({obj[0]:+.3f},{obj[1]:+.3f}) -> after=({new_obj[0]:+.3f},{new_obj[1]:+.3f}) "
            f"contact={new_contact} dist_to_goal={float(np.linalg.norm(new_obj[:2]-destination_xyz[:2])):.3f}"
        )
        if reached:
            print(f"[done] cube delivered to goal at gen {gen}")
            break

        brush = new_brush[0:3].copy()
        brush_n = new_brush[3:6].copy()
        brush_sd = new_brush[6:9].copy()
        obj = np.asarray(new_obj, dtype=np.float32)
        in_contact = bool(new_contact)

    out_root = _resolve(str(Path("training/verification") / args.output_subdir))
    executed_dir = out_root / "executed"
    out_path = executed_dir / "firstframe_executed.mp4"

    import pyrender

    renderer = pyrender.OffscreenRenderer(viewport_width=RENDER_WIDTH, viewport_height=RENDER_HEIGHT)
    try:
        render_executed_video(
            generations, out_path, renderer=renderer, chunk_size=chunk, fps=float(args.fps)
        )
    finally:
        renderer.delete()
    print(f"Wrote {out_path} ({len(generations)} generations)")


if __name__ == "__main__":
    main()
