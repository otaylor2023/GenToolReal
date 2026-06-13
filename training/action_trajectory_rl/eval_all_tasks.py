"""Evaluate all four reactive tool tasks for an action-trajectory checkpoint.

This is an eval-only companion to ``train_grpo.py``. It samples an equal number
of scenes per task, predicts 15-waypoint trajectories with the flow model, runs
closed-loop IsaacGym rollouts through the existing single-task worker path, and
writes per-task success/reward metrics to JSON. It does not update the model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POLICY_EXEC_ROOT = REPO_ROOT / "policy_exec"
if str(POLICY_EXEC_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_EXEC_ROOT))

from generative_str_pipeline.sim_rollout.sample_flip_scenes import sample_flip_scenes
from generative_str_pipeline.sim_rollout.sample_hammer_scenes import (
    HAMMER_NAIL_POST_HEIGHT_M,
    sample_hammer_scenes,
)
from generative_str_pipeline.sim_rollout.sample_pour_scenes import sample_pour_scenes
from generative_str_pipeline.sim_rollout.sample_rl_scenes import sample_rl_scenes
from generative_str_pipeline.sim_workspace import (
    WIDE_TABLE_X_MAX_M,
    WIDE_TABLE_X_MIN_M,
    WIDE_TABLE_Y_MAX_M,
    WIDE_TABLE_Y_MIN_M,
)
from training.action_expert.xyz_normalization import load_xyz_normalization_stats
from training.action_trajectory.dataset import WaypointTrajectoryDataset, waypoint_collate
from training.action_trajectory.model import ActionTrajectoryModel
from training.action_trajectory.train import _batch_to_model_inputs
from training.action_trajectory_rl.flow_grpo import sample_with_logprobs
from training.action_trajectory_rl.reward import (
    FlipRewardConfig,
    HammerRewardConfig,
    PourRewardConfig,
    RewardConfig,
    compute_combined_reward,
    compute_flip_reward,
    compute_hammer_reward,
    compute_pour_reward,
    flip_reward_breakdown,
    hammer_reward_breakdown,
    pour_reward_breakdown,
    reward_breakdown,
)
from training.action_trajectory_rl.train_grpo import (
    GrpoTrainConfig,
    _build_all_task_specs,
    _load_models,
    _resolve,
    _scene_to_sample,
)


TASK_ORDER = ["brush_sweep", "spatula_flip", "spoon_pour", "hammer_nail"]


def _sample_task_scenes(task_name: str, n: int, seed: int) -> List[Dict[str, Any]]:
    if task_name == "brush_sweep":
        scenes = sample_rl_scenes(n, seed=seed)
    elif task_name == "spatula_flip":
        scenes = sample_flip_scenes(n, seed=seed)
    elif task_name == "spoon_pour":
        scenes = sample_pour_scenes(n, seed=seed)
    elif task_name == "hammer_nail":
        scenes = sample_hammer_scenes(n, seed=seed)
    else:
        raise ValueError(task_name)
    for sc in scenes:
        sc["grpo_task_type"] = task_name
    return scenes


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _predict_waypoints(
    *,
    model,
    clip,
    scenes: List[Dict[str, Any]],
    mean_np: np.ndarray,
    std_np: np.ndarray,
    norm_eps: float,
    device: torch.device,
    integration_steps: int,
    seed: int,
) -> torch.Tensor:
    samples = [_scene_to_sample(sc, i) for i, sc in enumerate(scenes)]
    ds = WaypointTrajectoryDataset(samples, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps))
    collated = waypoint_collate([ds[i] for i in range(len(samples))])
    batch_tensors = _batch_to_model_inputs(collated, clip, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    with torch.no_grad():
        fs = sample_with_logprobs(
            model,
            batch_tensors,
            steps=int(integration_steps),
            sigma=0.0,
            generator=gen,
        )
    xyz_mean = torch.tensor(mean_np, device=device, dtype=torch.float32)
    xyz_std = torch.tensor(std_np, device=device, dtype=torch.float32)
    contact, normal, sdir = ActionTrajectoryModel.postprocess_waypoints(
        fs.final, xyz_mean, xyz_std, float(norm_eps)
    )
    return torch.cat([contact, normal, sdir], dim=-1).view(
        -1, ActionTrajectoryModel.NUM_WAYPOINTS, 9
    )


def _score_task(task_name: str, data, scenes: List[Dict[str, Any]], tcfg: GrpoTrainConfig):
    dest_np = np.array([s["destination_xyz_world"] for s in scenes], dtype=np.float64)
    if task_name == "spatula_flip":
        cfg = FlipRewardConfig(
            w_flip=float(tcfg.reward_w_flip),
            inverted_dot_max=float(tcfg.flip_inverted_dot_max),
            settle_tol_m=float(tcfg.flip_settle_tol_m),
            material_half_z_m=float(tcfg.flip_material_half_z_m),
            partial_scale=float(tcfg.flip_partial_scale),
            table_x_min_m=float(WIDE_TABLE_X_MIN_M),
            table_x_max_m=float(WIDE_TABLE_X_MAX_M),
            table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
            table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
            off_table_drop_m=float(tcfg.off_table_drop_m),
            off_table_penalty=float(tcfg.off_table_penalty),
        )
        table_z = np.array([s["table_xyz_world"][2] for s in scenes], dtype=np.float64)
        rewards = compute_flip_reward(
            object_quat_final=data["object_quat_final"],
            object_xyz_final=data["object_xyz_final"],
            table_z=table_z,
            cfg=cfg,
        )
        breakdown = flip_reward_breakdown(
            object_quat_final=data["object_quat_final"],
            object_xyz_final=data["object_xyz_final"],
            table_z=table_z,
            cfg=cfg,
        )
        success = breakdown["flip_success"]
    elif task_name == "spoon_pour":
        cfg = PourRewardConfig(
            w_pour=float(tcfg.reward_w_pour),
            goal_region_half_x_m=float(tcfg.goal_region_radius_m),
            goal_region_half_y_m=float(tcfg.goal_region_radius_m),
            material_radius_m=float(tcfg.material_radius_m),
            settle_tol_m=float(tcfg.pour_settle_tol_m),
            material_half_z_m=float(tcfg.pour_material_half_z_m),
            partial_scale=float(tcfg.pour_partial_scale),
            table_x_min_m=float(WIDE_TABLE_X_MIN_M),
            table_x_max_m=float(WIDE_TABLE_X_MAX_M),
            table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
            table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
            off_table_drop_m=float(tcfg.off_table_drop_m),
            off_table_penalty=float(tcfg.off_table_penalty),
        )
        table_z = np.array([s["table_xyz_world"][2] for s in scenes], dtype=np.float64)
        rewards = compute_pour_reward(
            material_start_xyz=data["ball_start_xyz"],
            material_final_xyz=data["ball_final_xyz"],
            destination_xyz=dest_np,
            table_z=table_z,
            cfg=cfg,
        )
        breakdown = pour_reward_breakdown(
            material_start_xyz=data["ball_start_xyz"],
            material_final_xyz=data["ball_final_xyz"],
            destination_xyz=dest_np,
            table_z=table_z,
            cfg=cfg,
        )
        success = breakdown["pour_success"]
    elif task_name == "hammer_nail":
        cfg = HammerRewardConfig(
            w_hammer=float(tcfg.reward_w_hammer),
            target_tol_m=float(tcfg.hammer_target_tol_m),
            partial_scale=float(tcfg.hammer_partial_scale),
        )
        half_post = 0.5 * float(HAMMER_NAIL_POST_HEIGHT_M)
        head_start = np.asarray(data["ball_start_xyz"], dtype=np.float64).copy()
        head_final = np.asarray(data["ball_final_xyz"], dtype=np.float64).copy()
        head_start[:, 2] += half_post
        head_final[:, 2] += half_post
        rewards = compute_hammer_reward(
            head_start_xyz=head_start,
            head_final_xyz=head_final,
            target_xyz=dest_np,
            cfg=cfg,
        )
        breakdown = hammer_reward_breakdown(
            head_start_xyz=head_start,
            head_final_xyz=head_final,
            target_xyz=dest_np,
            cfg=cfg,
        )
        success = breakdown["hammer_success"]
    else:
        cfg = RewardConfig(
            w_track=float(tcfg.reward_w_track),
            w_ball=float(tcfg.reward_w_ball),
            ball_success_radius_m=float(tcfg.ball_success_radius_m),
            goal_region_half_x_m=float(tcfg.goal_region_radius_m),
            goal_region_half_y_m=float(tcfg.goal_region_radius_m),
            ball_radius_m=float(tcfg.material_radius_m),
            table_x_min_m=float(WIDE_TABLE_X_MIN_M),
            table_x_max_m=float(WIDE_TABLE_X_MAX_M),
            table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
            table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
            off_table_drop_m=float(tcfg.off_table_drop_m),
            off_table_penalty=float(tcfg.off_table_penalty),
        )
        rewards = compute_combined_reward(
            tracking_frac=data["tracking_frac"],
            ball_start_xyz=data["ball_start_xyz"],
            ball_final_xyz=data["ball_final_xyz"],
            destination_xyz=dest_np,
            cfg=cfg,
        )
        breakdown = reward_breakdown(
            tracking_frac=data["tracking_frac"],
            ball_start_xyz=data["ball_start_xyz"],
            ball_final_xyz=data["ball_final_xyz"],
            destination_xyz=dest_np,
            cfg=cfg,
        )
        success = breakdown["ball_in_region"]
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(success)),
        "tracking_mean": float(np.mean(data["tracking_frac"])),
        "num_envs": int(len(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/cfg/action_trajectory_all_tasks_grpo_from_10epoch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    t0 = time.time()
    raw = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8"))
    tcfg = GrpoTrainConfig(**{k: v for k, v in raw.items() if hasattr(GrpoTrainConfig, k)})
    tcfg.checkpoint = str(args.checkpoint)
    tcfg.num_scenes = int(args.scenes_per_task) * len(TASK_ORDER)
    tcfg.group_size = 1
    tcfg.reward_rollout_repeats = 1
    tcfg.sde_sigma = 0.0
    tcfg.use_wandb = False
    tcfg.headless = True

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / f"{out_path.stem}_workers"
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_np, std_np, norm_eps = load_xyz_normalization_stats(_resolve(tcfg.normalization_stats_path))
    _cfg, clip, model, _ref_model = _load_models(_resolve(tcfg.config), _resolve(args.checkpoint), device)
    model.eval()

    rng = np.random.default_rng(int(args.seed))
    scenes_by_task: Dict[str, List[Dict[str, Any]]] = {}
    wp_by_task: Dict[str, torch.Tensor] = {}
    for task_name in TASK_ORDER:
        scenes = _sample_task_scenes(
            task_name, int(args.scenes_per_task), seed=int(rng.integers(0, 2**31))
        )
        scenes_by_task[task_name] = scenes
        wp_by_task[task_name] = _predict_waypoints(
            model=model,
            clip=clip,
            scenes=scenes,
            mean_np=mean_np,
            std_np=std_np,
            norm_eps=float(norm_eps),
            device=device,
            integration_steps=int(tcfg.integration_steps),
            seed=int(rng.integers(0, 2**31)),
        )

    metrics = {}
    for task_name in TASK_ORDER:
        spec_path = work_dir / f"{task_name}.json"
        out_npz = work_dir / f"{task_name}.npz"
        log_path = work_dir / f"{task_name}.log"
        spec = {
            "tcfg": asdict(tcfg),
            "task_name": task_name,
            "scenes": scenes_by_task[task_name],
            "wp_world": wp_by_task[task_name].detach().cpu().numpy().astype(np.float32).tolist(),
            "model_checkpoint": str(_resolve(args.checkpoint)),
            "seed": int(rng.integers(0, 2**31)),
        }
        spec_path.write_text(json.dumps(spec, default=_json_default), encoding="utf-8")
        env = os.environ.copy()
        py_path = [str(REPO_ROOT), str(POLICY_EXEC_ROOT)]
        if env.get("PYTHONPATH"):
            py_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_path)
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "training.action_trajectory_rl.train_grpo",
                "--rollout_worker_spec",
                str(spec_path),
                "--rollout_worker_out",
                str(out_npz),
            ],
            cwd=str(REPO_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            code = proc.wait()
            log_f.close()
            if code != 0:
                raise RuntimeError(f"{task_name} worker exited {code}; see {log_path}")
            data = np.load(out_npz)
            metrics[task_name] = _score_task(task_name, data, scenes_by_task[task_name], tcfg)
        finally:
            if proc.poll() is None:
                proc.terminate()
            if not log_f.closed:
                log_f.close()

    result = {
        "label": str(args.label),
        "checkpoint": str(_resolve(args.checkpoint)),
        "config": str(_resolve(args.config)),
        "scenes_per_task": int(args.scenes_per_task),
        "seed": int(args.seed),
        "elapsed_sec": float(time.time() - t0),
        "tasks": metrics,
        "macro_success": float(np.mean([m["success_rate"] for m in metrics.values()])),
        "macro_reward": float(np.mean([m["reward_mean"] for m in metrics.values()])),
        "macro_tracking": float(np.mean([m["tracking_mean"] for m in metrics.values()])),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
