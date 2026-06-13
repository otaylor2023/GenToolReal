"""Diagnostic: trace one RL scene through the VLA + sim-batch conversion to
check sweep direction vs. material/destination marker placement.

Run (no IsaacGym needed):
  conda run -n policy_exec python -m training.action_trajectory_rl.diag_directions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_str_pipeline.sim_rollout.sample_rl_scenes import sample_rl_scenes
from training.action_expert.hf_env import apply_hf_cache, apply_hf_env
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.config import load_config
from training.action_trajectory.dataset import (
    WaypointTrajectoryDataset,
    waypoint_collate,
)
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.text_encoder import ClipTextEncoder
from training.action_trajectory.train import _batch_to_model_inputs, _rollout
from training.action_trajectory_rl.sim_convert import waypoints_tensor_to_sim_batch
from training.action_trajectory_rl.train_grpo import _scene_to_sample


def main() -> None:
    cfg_path = REPO_ROOT / "training/cfg/action_trajectory_brush_sweep_diverse.yaml"
    ckpt_path = REPO_ROOT / "training/runs/action_trajectory/run_0007/checkpoint_epoch_0030.pt"
    control_frame = REPO_ROOT / "generative_str_pipeline/assets/object_control_points/blue_brush.json"

    apply_hf_env()
    cfg = load_config(cfg_path)
    apply_hf_cache(str(cfg.hf_cache_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_np, std_np, norm_eps = load_xyz_normalization_stats(
        REPO_ROOT / "training/cfg/normalization_stats_action_trajectory_brush_sweep_diverse.json"
    )
    xyz_mean = torch.tensor(mean_np, dtype=torch.float32, device=device)
    xyz_std = torch.tensor(std_np, dtype=torch.float32, device=device)

    n_scenes = 3
    scenes = sample_rl_scenes(n_scenes, seed=7)
    samples = [_scene_to_sample(sc, i) for i, sc in enumerate(scenes)]
    ds = WaypointTrajectoryDataset(samples, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps))
    collated = waypoint_collate([ds[i] for i in range(n_scenes)])

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
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    clip.eval()

    m = _batch_to_model_inputs(collated, clip, device)
    with torch.no_grad():
        out = _rollout(model=model, batch_tensors=m, steps=int(cfg.integration_steps), n_samples=1)
    # out shape: [B, n_samples, 54] -> mean over the samples axis (dim=1)
    pred_norm = out.mean(dim=1)
    contact, normal, sdir = ActionTrajectoryModel.postprocess_waypoints(
        pred_norm, xyz_mean, xyz_std, float(norm_eps)
    )
    wp_world = torch.cat([contact, normal, sdir], dim=-1).view(-1, 6, 9)

    import os
    align = os.environ.get("DIAG_ALIGN", "0") == "1"
    print(f"\n##### align_to_canonical={align} #####")
    goals_b, start_b, mat_b, dest_b, T = waypoints_tensor_to_sim_batch(
        wp_world, scenes, control_frame_path=control_frame, steps_per_segment=8,
        align_to_canonical=align,
    )

    # Load brush control frame + edge corners to verify executed orientation.
    import json as _json

    from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
        load_control_frame,
        matrix_from_quat_xyzw,
        waypoint_to_object_pose,
    )

    cf_json = _json.loads(control_frame.read_text(encoding="utf-8"))
    rect = cf_json["corners_rectangle"]
    fl = np.asarray(rect["front_left"], dtype=np.float64)
    fr = np.asarray(rect["front_right"], dtype=np.float64)
    bl = np.asarray(rect["back_left"], dtype=np.float64)
    br = np.asarray(rect["back_right"], dtype=np.float64)
    front_center_obj = 0.5 * (fl + fr)
    back_center_obj = 0.5 * (bl + br)
    T_oc = load_control_frame(control_frame)

    np.set_printoptions(precision=3, suppress=True)
    for i, sc in enumerate(scenes):
        mat = np.asarray(sc["material_xyz_world"])
        dest = np.asarray(sc["destination_xyz_world"])
        wp = wp_world[i].cpu().numpy()
        print(f"\n=== scene {i}: {sc['instruction']!r}")
        print(f"  INPUT material(ball)={mat}  destination(goal)={dest}")
        print(f"  model touchdown wp[2]={wp[2,:3]}  end wp[5]={wp[5,:3]}")
        sweep_vec = wp[5, :3] - wp[2, :3]
        md_vec = dest - mat
        cos = float(np.dot(sweep_vec[:2], md_vec[:2]) / (np.linalg.norm(sweep_vec[:2]) * np.linalg.norm(md_vec[:2]) + 1e-9))
        print(f"  sweep(wp2->wp5) xy vs (material->destination) xy cos={cos:+.3f}  "
              f"({'ALIGNED' if cos > 0.3 else 'REVERSED/OFF'})")

        # --- surface_dir orientation check ---
        sds = wp[:, 6:9]
        sd_unit = sds / (np.linalg.norm(sds, axis=1, keepdims=True) + 1e-9)
        # consistency across the 6 waypoints (should be ~constant orientation)
        pair_cos = sd_unit @ sd_unit[2]
        print(f"  model surface_dir[2]={sd_unit[2]}  (front->back orientation)")
        print(f"  surface_dir consistency across wps (cos vs wp2)={np.round(pair_cos,2)}")
        sd_vs_motion = float(np.dot(sd_unit[2][:2], md_vec[:2]) /
                             (np.linalg.norm(sd_unit[2][:2]) * np.linalg.norm(md_vec[:2]) + 1e-9))
        print(f"  surface_dir vs motion(material->dest) cos={sd_vs_motion:+.2f} "
              f"(expect NEGATIVE: front leads, surface_dir points back)")
        # executed brush orientation: transform actual brush edges by the pose
        xyz_w, quat_w = waypoint_to_object_pose(wp[2, 0:3], wp[2, 3:6], wp[2, 6:9], T_oc)
        R = matrix_from_quat_xyzw(quat_w)
        front_w = R @ front_center_obj + xyz_w
        back_w = R @ back_center_obj + xyz_w
        exec_fb = back_w - front_w
        exec_fb_u = exec_fb / (np.linalg.norm(exec_fb) + 1e-9)
        cos_exec = float(np.dot(exec_fb_u, sd_unit[2]))
        print(f"  EXECUTED brush (back-front) world dir={exec_fb_u}  "
              f"cos vs predicted surface_dir={cos_exec:+.3f}  "
              f"({'MATCH' if cos_exec > 0.9 else 'MISMATCH!!'})")
        g = np.asarray(goals_b[i])
        ball = np.asarray(mat_b[i])
        goal = np.asarray(dest_b[i])
        print(f"  sim brush spawn={g[0,:3]}  final={g[-1,:3]}")
        print(f"  sim ball(mat_b)={ball}  goal(dest_b)={goal}")
        # touchdown = sim goal closest to the ball in xy
        d_to_ball = np.linalg.norm(g[:, :2] - ball[:2], axis=1)
        td = int(np.argmin(d_to_ball))
        print(f"  brush touchdown goal idx={td}/{len(g)} at {g[td,:3]} "
              f"(dist to ball={d_to_ball[td]*100:.1f}cm)")
        sweep = g[-1, :2] - g[td, :2]
        ball_to_goal = goal[:2] - ball[:2]
        cos2 = float(np.dot(sweep, ball_to_goal) / (np.linalg.norm(sweep) * np.linalg.norm(ball_to_goal) + 1e-9))
        print(f"  sim sweep(touchdown->final) vs (ball->goal) cos={cos2:+.3f}  "
              f"({'ALIGNED' if cos2 > 0.3 else 'REVERSED/OFF'})  "
              f"ball-goal dist={np.linalg.norm(ball_to_goal)*100:.1f}cm")


if __name__ == "__main__":
    main()
