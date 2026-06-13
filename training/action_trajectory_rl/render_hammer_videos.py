"""Render IsaacGym hammer-a-nail rollout videos driven by the pretrained VLA.

This is an EVAL/VIDEO-ONLY driver -- it does NOT run GRPO optimization. It loads
a completed action-trajectory checkpoint (e.g. run_0023, pretrained on
``dataset_0014_hammer_nail_reactive``), samples a few hammer-a-nail scenes on the
fixed sim table, and rolls the model out closed-loop (receding horizon, exactly
like the dataset) in the new z-locked "nail" IsaacGym env, capturing video.

Run from the repo root in the policy_exec conda env (IsaacGym + transformers):

    export LD_LIBRARY_PATH=/home/ubuntu/miniconda3/envs/policy_exec/lib:$LD_LIBRARY_PATH
    /home/ubuntu/miniconda3/envs/policy_exec/bin/python \\
        -m training.action_trajectory_rl.render_hammer_videos \\
        --num_scenes 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
POLICY_EXEC_ROOT = REPO_ROOT / "policy_exec"
if str(POLICY_EXEC_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_EXEC_ROOT))

# IsaacGym must be imported before torch in processes that use it.
try:  # pragma: no cover - depends on IsaacGym installation.
    from isaacgym import gymapi as _isaacgym_preload  # noqa: F401
except Exception:
    _isaacgym_preload = None

import torch

from generative_str_pipeline.sim_rollout.sample_hammer_scenes import (
    HAMMER_NAIL_POST_HEIGHT_M,
    HAMMER_NAIL_POST_SIZE,
    sample_hammer_scenes,
)
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.dataset import (
    WaypointTrajectoryDataset,
    waypoint_collate,
)
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.train import _batch_to_model_inputs
from training.action_trajectory_rl.closed_loop_rollout import run_closed_loop_rollout
from training.action_trajectory_rl.flow_grpo import (
    FlowSampleResult,
    sample_with_logprobs,
)
from training.action_trajectory_rl.train_grpo import (
    _load_models,
    _resolve,
    _scene_to_sample,
)

# Claw-hammer assets (object registered in dextoolbench.objects).
HAMMER_OBJECT_NAME = "claw_hammer"
HAMMER_TOOL_OBJ = "policy_exec/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.obj"
HAMMER_CONTROL_FRAME = "closed_loop/closed_loop/assets/control_frames/claw_hammer.json"
# Narrow centered table (same surface height the scenes are sampled on).
HAMMER_TABLE_URDF = "urdf/table_narrow.urdf"


def main() -> None:
    p = argparse.ArgumentParser(description="Render VLA hammer-a-nail sim videos.")
    p.add_argument(
        "--config",
        type=str,
        default="training/cfg/action_trajectory_hammer_nail_reactive_10epoch.yaml",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="training/runs/action_trajectory/run_0023/checkpoint_best.pt",
    )
    p.add_argument(
        "--normalization_stats_path",
        type=str,
        default="training/cfg/normalization_stats_action_trajectory_hammer_nail_reactive.json",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="training/runs/action_trajectory/run_0023/isaacgym_hammer_videos",
    )
    p.add_argument("--policy_exec_config", type=str, default="policy_exec/pretrained_policy/config.yaml")
    p.add_argument("--policy_exec_checkpoint", type=str, default="policy_exec/pretrained_policy/model.pth")
    p.add_argument("--num_scenes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--integration_steps", type=int, default=30)
    p.add_argument("--sde_sigma", type=float, default=0.0)
    p.add_argument("--steps_per_segment", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=5)
    p.add_argument("--max_replans", type=int, default=30)
    p.add_argument("--max_steps_per_chunk", type=int, default=200)
    p.add_argument("--max_total_steps", type=int, default=3000)
    p.add_argument("--max_stall_replans", type=int, default=8)
    p.add_argument("--episode_length", type=int, default=3000)
    p.add_argument("--capture_interval", type=int, default=4)
    p.add_argument("--video_fps", type=int, default=15)
    p.add_argument("--nail_sink_max_m", type=float, default=0.05)
    p.add_argument("--nail_z_damping", type=float, default=0.8)
    args = p.parse_args()

    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_np, std_np, norm_eps = load_xyz_normalization_stats(
        _resolve(args.normalization_stats_path)
    )
    cfg, clip, model, _ref = _load_models(
        _resolve(args.config), _resolve(args.checkpoint), device
    )
    model.eval()
    xyz_mean = torch.tensor(mean_np, device=device, dtype=torch.float32)
    xyz_std = torch.tensor(std_np, device=device, dtype=torch.float32)

    n = int(args.num_scenes)
    scenes = sample_hammer_scenes(n, seed=int(args.seed))
    print(f"[hammer] sampled {len(scenes)} scenes; table_z={scenes[0]['table_xyz_world'][2]}")
    for i, s in enumerate(scenes):
        print(
            f"  scene {i}: nail_head_z0={s['material_xyz_world'][2]:.4f} "
            f"target_z={s['destination_xyz_world'][2]:.4f} "
            f"home={np.round(s['tool_contact_xyz_world'], 3).tolist()}"
        )

    def _plan_wp(scenes_list: List[Dict[str, Any]]) -> torch.Tensor:
        samples = [_scene_to_sample(sc, i) for i, sc in enumerate(scenes_list)]
        ds = WaypointTrajectoryDataset(
            samples, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
        )
        coll = waypoint_collate([ds[i] for i in range(len(samples))])
        bt = _batch_to_model_inputs(coll, clip, device)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) + 12345)
        with torch.no_grad():
            fs: FlowSampleResult = sample_with_logprobs(
                model,
                bt,
                steps=int(args.integration_steps),
                sigma=float(args.sde_sigma),
                generator=gen,
            )
        c, nrm, sd = ActionTrajectoryModel.postprocess_waypoints(
            fs.final, xyz_mean, xyz_std, float(norm_eps)
        )
        return torch.cat([c, nrm, sd], dim=-1).view(
            -1, ActionTrajectoryModel.NUM_WAYPOINTS, 9
        )

    wp_world = _plan_wp(scenes)

    from dextoolbench.vec_rollout import VectorizedSimRollout

    sim_runner = VectorizedSimRollout(
        num_envs=n,
        config_path=_resolve(args.policy_exec_config),
        checkpoint_path=_resolve(args.policy_exec_checkpoint),
        object_name=HAMMER_OBJECT_NAME,
        headless=True,
        episode_length=int(args.episode_length),
        table_urdf=HAMMER_TABLE_URDF,
        material_size=list(HAMMER_NAIL_POST_SIZE),
        material_density=200.0,
        nail_zlock=True,
        nail_sink_max_m=float(args.nail_sink_max_m),
        nail_z_damping=float(args.nail_z_damping),
        record_video=True,
        video_num_envs=n,
        video_env_indices=list(range(n)),
    )

    # Re-frame each shown env's camera tight on its (centered) nail so the strike
    # and the post sinking are clearly visible (the default framing targets the
    # off-center brush workspace).
    from isaacgym import gymapi

    env = sim_runner.env
    for ei, handle in zip(sim_runner.video_env_indices, sim_runner.camera_handles):
        nail_xy = np.asarray(scenes[ei]["material_xyz_world"], dtype=np.float64)[:2]
        table_z = float(scenes[ei]["table_xyz_world"][2])
        target = gymapi.Vec3(float(nail_xy[0]), float(nail_xy[1]), table_z + 0.07)
        cam_pos = target + gymapi.Vec3(0.30, -0.42, 0.26)
        env.gym.set_camera_location(handle, env.envs[ei], cam_pos, target)

    print("[hammer] running closed-loop rollout (capture on)...")
    result = None
    try:
        result = run_closed_loop_rollout(
            sim_runner,
            scenes=scenes,
            wp_world=wp_world,
            model_resample_fn=_plan_wp,
            control_frame_path=_resolve(HAMMER_CONTROL_FRAME),
            tool_obj_path=_resolve(HAMMER_TOOL_OBJ),
            task="hammer",
            steps_per_segment=int(args.steps_per_segment),
            chunk_size=int(args.chunk_size),
            max_replans=int(args.max_replans),
            max_steps_per_chunk=int(args.max_steps_per_chunk),
            max_total_steps=int(args.max_total_steps),
            max_stall_replans=int(args.max_stall_replans),
            nail_post_height_m=float(HAMMER_NAIL_POST_HEIGHT_M),
            material_z_offset=-float(HAMMER_NAIL_POST_HEIGHT_M),
            capture=True,
            capture_interval=int(args.capture_interval),
        )
    except Exception as exc:  # pragma: no cover
        import traceback

        print(f"[hammer][warn] rollout raised: {exc}")
        traceback.print_exc()

    if result is None or not result.frames_by_env:
        print("[hammer][error] no frames captured; nothing to save.")
        return

    import imageio

    start = result.ball_start_xyz
    final = result.ball_final_xyz
    saved: List[str] = []
    for ei in sorted(result.frames_by_env.keys()):
        frames = result.frames_by_env.get(ei) or []
        if not frames:
            continue
        vid_path = out_dir / f"hammer_scene{ei:02d}.mp4"
        imageio.mimsave(str(vid_path), frames, fps=int(args.video_fps))
        sink_cm = float(start[ei, 2] - final[ei, 2]) * 100.0
        saved.append(str(vid_path))
        print(
            f"[hammer] saved {vid_path} ({len(frames)} frames) "
            f"nail_sink~{sink_cm:.1f}cm track={float(result.tracking_frac[ei]):.2f}"
        )

    print("\n=== HAMMER VIDEO SUMMARY ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"saved {len(saved)} videos under {out_dir}")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    main()
