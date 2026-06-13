"""GRPO fine-tuning for the flow-matching action-trajectory VLA using IsaacGym rollouts.

Quick start (VLA update only, no IsaacGym):
  python -m training.action_trajectory_rl.train_grpo --skip_sim --num_iterations 10

Full loop (requires policy_exec conda env with IsaacGym + transformers in same env,
or extend with a subprocess sim worker):
  PYTHONPATH=policy_exec:$PYTHONPATH python -m training.action_trajectory_rl.train_grpo \\
    --num_scenes 8 --group_size 4

Vectorized sim smoke (4 envs):
  cd policy_exec && python -m dextoolbench.grpo_sim_smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# IsaacGym must be imported before torch in processes that use it. This import is
# optional so `--skip_sim` still works in non-Isaac training environments.
POLICY_EXEC_ROOT = REPO_ROOT / "policy_exec"
if str(POLICY_EXEC_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_EXEC_ROOT))
try:  # pragma: no cover - depends on IsaacGym installation.
    from isaacgym import gymapi as _isaacgym_preload  # noqa: F401
except Exception:
    _isaacgym_preload = None

import torch
import torch.optim as optim
import yaml

from generative_str_pipeline.sim_rollout.sample_rl_scenes import sample_rl_scenes
from generative_str_pipeline.sim_rollout.sample_flip_scenes import (
    FLIP_MATERIAL_SIZE,
    PAN_LAYERS,
    PAN_RADIUS_M,
    PAN_SEGMENTS,
    PAN_WALL_HEIGHT_M,
    PAN_WALL_THICKNESS_M,
    PAN_XYZ,
    sample_flip_scenes,
)
from generative_str_pipeline.sim_rollout.sample_pour_scenes import (
    POUR_MATERIAL_SIZE,
    POUR_PAN_XYZ,
    sample_pour_scenes,
)
from generative_str_pipeline.sim_rollout.sample_hammer_scenes import (
    HAMMER_NAIL_POST_HEIGHT_M,
    HAMMER_NAIL_POST_SIZE,
    sample_hammer_scenes,
)
from generative_str_pipeline.sim_workspace import (
    TABLE_HALF_X_M,
    TABLE_HALF_Y_M,
    WIDE_TABLE_X_MAX_M,
    WIDE_TABLE_X_MIN_M,
    WIDE_TABLE_Y_MAX_M,
    WIDE_TABLE_Y_MIN_M,
)

# Sim-only wide table: brush head rests on the +x extension while the handle
# stays over the original region (see table_wide_brush.urdf / sim_workspace).
WIDE_TABLE_URDF = "urdf/table_wide_brush.urdf"
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
from training.action_trajectory.train import _batch_to_model_inputs
from training.action_trajectory_rl.flow_grpo import (
    FlowSampleResult,
    grpo_policy_loss,
    kl_ref_loss,
    recompute_logprob_sum,
    sample_with_logprobs,
)
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
    group_relative_advantages,
    hammer_reward_breakdown,
    pour_reward_breakdown,
    reward_breakdown,
)
from training.action_trajectory_rl.closed_loop_rollout import (
    ClosedLoopGenRecord,
    run_closed_loop_rollout,
)
from training.action_trajectory_rl.sim_convert import waypoints_tensor_to_sim_batch
from generative_str_pipeline.sim_workspace import (
    WIDE_TABLE_X_MAX_M,
    WIDE_TABLE_X_MIN_M,
    WIDE_TABLE_Y_MAX_M,
    WIDE_TABLE_Y_MIN_M,
)


@dataclass
class GrpoTrainConfig:
    config: str = "training/cfg/action_trajectory_brush_sweep_diverse.yaml"
    checkpoint: str = "training/runs/action_trajectory/run_0007/checkpoint_epoch_0030.pt"
    normalization_stats_path: str = (
        "training/cfg/normalization_stats_action_trajectory_brush_sweep_diverse.json"
    )
    policy_exec_config: str = "policy_exec/pretrained_policy/config.yaml"
    policy_exec_checkpoint: str = "policy_exec/pretrained_policy/model.pth"
    control_frame: str = (
        "generative_str_pipeline/assets/object_control_points/blue_brush.json"
    )
    # Task selector: "brush_sweep" (default), "spatula_flip", "spoon_pour", "hammer_nail", or "all_tasks".
    task_type: str = "brush_sweep"
    # Manipulated tool object (NAME_TO_OBJECT key) and its collision mesh used to
    # rest the tool flat at spawn. Empty -> brush defaults.
    object_name: str = "blue_brush"
    tool_obj_path: str = ""
    # Flip reward (used when task_type == "spatula_flip").
    reward_w_flip: float = 1.0
    flip_inverted_dot_max: float = -0.5
    flip_settle_tol_m: float = 0.04
    flip_material_half_z_m: float = 0.006
    flip_partial_scale: float = 0.5
    # Pour reward (used when task_type == "spoon_pour"): material poured into the
    # goal region and settled on the table.
    reward_w_pour: float = 1.0
    pour_settle_tol_m: float = 0.03
    pour_material_half_z_m: float = 0.009
    pour_partial_scale: float = 0.8
    # Hammer reward (used when task_type == "hammer_nail" or "all_tasks").
    reward_w_hammer: float = 1.0
    hammer_target_tol_m: float = 0.004
    hammer_partial_scale: float = 0.8
    output_dir: str = "training/runs/action_trajectory_grpo"
    num_iterations: int = 200
    num_scenes: int = 8
    group_size: int = 4
    integration_steps: int = 30
    sde_sigma: float = 0.02
    grpo_clip: float = 0.2
    kl_coef: float = 0.01
    inner_epochs: int = 2
    lr: float = 5e-5
    steps_per_segment: int = 1
    lift_z: float = 0.0
    z_offset: float = 0.0
    max_rollout_steps: int = 600
    use_closed_loop: bool = True
    closed_loop_chunk_size: int = 2
    closed_loop_max_replans: int = 15
    closed_loop_max_steps_per_chunk: int = 300
    # Total sim-step budget across all replans of one closed-loop rollout. Kept
    # separate from ``max_rollout_steps`` (the env episode length) so many short
    # replans can fit without inflating the per-env no-progress timeout. Defaults
    # to ``max_rollout_steps`` when <= 0.
    closed_loop_max_total_steps: int = 0
    # Stop an env after this many consecutive replans that reach none of their
    # chunk sub-goals (waypoint stayed unreachable).
    closed_loop_max_stall_replans: int = 3
    # Off-table drop threshold (m below the table top) for the failure check.
    off_table_drop_m: float = 0.08
    # Roll each sampled trajectory this many times in sim and average the reward,
    # so the GRPO advantage reflects true trajectory quality rather than sim
    # physics noise (a lucky single roll).
    reward_rollout_repeats: int = 4
    seed: int = 0
    reward_w_track: float = 1.0
    reward_w_ball: float = 0.5
    ball_success_radius_m: float = 0.06
    # Goal-region (blue patch) half-size and ball radius for the in-region test.
    goal_region_radius_m: float = 0.05
    material_radius_m: float = 0.02
    # Penalty applied to the ball term when the ball falls off the table.
    off_table_penalty: float = 1.0
    skip_sim: bool = False
    headless: bool = True
    use_wandb: bool = False
    wandb_project: str = "generative-str-action-expert"
    wandb_entity: str = ""
    wandb_group: str = "grpo_sweep"
    wandb_run_name: str = ""
    wandb_run_prefix: str = ""
    wandb_tags: Optional[List[str]] = None
    wandb_notes: str = ""
    video_every: int = 10
    video_capture_interval: int = 4
    video_fps: int = 15
    video_num_envs: int = 4


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _render_trajectory_images(
    *,
    shown_envs: List[int],
    env_to_sample: Dict[int, int],
    scenes: List[Dict[str, Any]],
    wp_world_np: np.ndarray,
    out_dir: Path,
    it: int,
    generations_by_env: Optional[Dict[int, List[ClosedLoopGenRecord]]] = None,
) -> Dict[int, Path]:
    """Render the dataset-style trajectory viz for the shown envs in a subprocess.

    ``env_to_sample`` maps each shown sim-env index to the per-sample index used
    to look up its scene and predicted waypoints (a sample may be rolled out
    multiple times in sim). pyrender/EGL runs in a separate process so its GL
    context never collides with IsaacGym. Returns {env_idx: png_path}.
    """
    import subprocess
    import tempfile

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    paths: Dict[int, Path] = {}
    for e in shown_envs:
        s = env_to_sample[e]
        png = out_dir / f"iter_{it:05d}_env{e:02d}_traj.png"
        paths[e] = png
        item: Dict[str, Any] = {
            "out_path": str(png),
            "movement_token": str(scenes[s].get("movement_token", "stroke_sweep")),
            "scene": scenes[s],
            "waypoints": wp_world_np[s].astype(float).tolist(),
        }
        if generations_by_env is not None and e in generations_by_env:
            item["generations"] = [
                {
                    "material_xyz": g.material_xyz,
                    "path_contacts": g.path_contacts,
                    "path_normals": g.path_normals,
                    "path_surface_dirs": g.path_surface_dirs,
                }
                for g in generations_by_env[e]
            ]
        items.append(item)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump({"items": items}, f)
        spec_path = f.name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "generative_str_pipeline.sim_rollout.render_rollout_trajectory",
            "--input",
            spec_path,
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        timeout=300,
    )
    return paths


def _next_run_dir(base: Path) -> Path:
    """Return base/run_00XX for the next unused index (matches other trainers)."""
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for d in base.glob("run_*"):
        if d.is_dir():
            try:
                existing.append(int(d.name.split("_")[-1]))
            except ValueError:
                continue
    idx = (max(existing) + 1) if existing else 0
    run_dir = base / f"run_{idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _scene_to_sample(scene: Dict[str, Any], idx: int) -> WaypointTrajectorySample:
    return WaypointTrajectorySample(
        scene_id="rl_grpo",
        shard_path="",
        datapoint_index=int(idx),
        movement_token=str(scene.get("movement_token", "stroke_sweep")),
        instruction=str(scene["instruction"]),
        tool_label=str(scene.get("tool_label", "the brush")),
        tool_contact_xyz_world=np.asarray(scene["tool_contact_xyz_world"], dtype=np.float32),
        tool_current_normal=np.asarray(scene["tool_current_normal"], dtype=np.float32),
        tool_current_surface_dir=np.asarray(
            scene["tool_current_surface_dir"], dtype=np.float32
        ),
        material_label=scene.get("material_label"),
        material_xyz_world=np.asarray(scene["material_xyz_world"], dtype=np.float32),
        material_normal=None,
        has_material=True,
        destination_label=scene.get("destination_label"),
        destination_xyz_world=np.asarray(scene["destination_xyz_world"], dtype=np.float32),
        destination_normal=None,
        has_destination=True,
        table_label=str(scene.get("table_label", "table surface center")),
        table_xyz_world=np.asarray(scene["table_xyz_world"], dtype=np.float32),
        table_normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        waypoints=np.zeros((ActionTrajectoryModel.NUM_WAYPOINTS, 9), dtype=np.float32),
    )


def _load_models(cfg_path: Path, ckpt_path: Path, device: torch.device):
    apply_hf_env()
    cfg = load_config(cfg_path)
    apply_hf_cache(str(cfg.hf_cache_dir))
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
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    model.train()
    return cfg, clip, model, ref_model


def _build_all_task_specs(tcfg: GrpoTrainConfig, envs_per_task: int) -> Dict[str, Dict[str, Any]]:
    """Task-specific sim settings for mixed-task GRPO rollout workers."""
    pan_envs = max(1, int(envs_per_task))
    return {
        "brush_sweep": {
            "sim_task": "sweep",
            "object_name": "blue_brush",
            "control_frame": "generative_str_pipeline/assets/object_control_points/blue_brush.json",
            "tool_obj_path": "",
            "runner_kw": {},
        },
        "spatula_flip": {
            "sim_task": "flip",
            "object_name": "flat_spatula",
            "control_frame": "generative_str_pipeline/assets/object_control_points/flat_spatula.json",
            "tool_obj_path": "policy_exec/assets/urdf/dextoolbench/spatula/flat_spatula/flat_spatula.obj",
            "runner_kw": {
                "material_size": list(FLIP_MATERIAL_SIZE),
                "add_pan": True,
                "pan_xyz_batch": [list(PAN_XYZ)] * pan_envs,
                "pan_radius_m": float(PAN_RADIUS_M),
                "pan_wall_height_m": float(PAN_WALL_HEIGHT_M),
                "pan_wall_thickness_m": float(PAN_WALL_THICKNESS_M),
                "pan_segments": int(PAN_SEGMENTS),
                "pan_layers": int(PAN_LAYERS),
            },
        },
        "spoon_pour": {
            "sim_task": "pour",
            "object_name": "spoon_spatula",
            "control_frame": "generative_str_pipeline/assets/object_control_points/spoon_spatula.json",
            "tool_obj_path": "policy_exec/assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.obj",
            "runner_kw": {
                "material_size": list(POUR_MATERIAL_SIZE),
                "add_pan": True,
                "pan_xyz_batch": [list(POUR_PAN_XYZ)] * pan_envs,
                "pan_radius_m": float(PAN_RADIUS_M),
                "pan_wall_height_m": float(PAN_WALL_HEIGHT_M),
                "pan_wall_thickness_m": float(PAN_WALL_THICKNESS_M),
                "pan_segments": int(PAN_SEGMENTS),
                "pan_layers": int(PAN_LAYERS),
            },
        },
        "hammer_nail": {
            "sim_task": "hammer",
            "object_name": "mallet_hammer",
            "control_frame": "closed_loop/closed_loop/assets/control_frames/mallet_hammer.json",
            "tool_obj_path": "policy_exec/assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.obj",
            "runner_kw": {
                "material_size": list(HAMMER_NAIL_POST_SIZE),
                "nail_zlock": True,
                "nail_sink_max_m": 0.08,
                "nail_z_damping": 0.5,
            },
        },
    }


def _run_rollout_worker(spec_path: Path, out_path: Path) -> None:
    """Single-task IsaacGym rollout worker used by the all_tasks trainer path."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    tcfg = GrpoTrainConfig(**{k: v for k, v in spec["tcfg"].items() if hasattr(GrpoTrainConfig, k)})
    task_name = str(spec["task_name"])
    scenes = [dict(s) for s in spec["scenes"]]
    wp_world = torch.as_tensor(np.asarray(spec["wp_world"], dtype=np.float32), dtype=torch.float32)
    envs_per_task = len(scenes)
    task_specs = _build_all_task_specs(tcfg, envs_per_task)
    task_spec = task_specs[task_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_np, std_np, norm_eps = load_xyz_normalization_stats(_resolve(tcfg.normalization_stats_path))
    _cfg, clip, model, _ref_model = _load_models(_resolve(tcfg.config), _resolve(spec["model_checkpoint"]), device)
    model.eval()
    xyz_mean = torch.tensor(mean_np, device=device, dtype=torch.float32)
    xyz_std = torch.tensor(std_np, device=device, dtype=torch.float32)
    rng = np.random.default_rng(int(spec.get("seed", 0)))

    def _resample_wp(scenes_list: List[Dict[str, Any]]) -> torch.Tensor:
        samples_local = [_scene_to_sample(sc, i) for i, sc in enumerate(scenes_list)]
        ds_local = WaypointTrajectoryDataset(
            samples_local, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
        )
        coll_local = waypoint_collate([ds_local[i] for i in range(len(samples_local))])
        bt_local = _batch_to_model_inputs(coll_local, clip, device)
        gen_local = torch.Generator(device=device)
        gen_local.manual_seed(int(rng.integers(0, 2**31)))
        with torch.no_grad():
            fs_local = sample_with_logprobs(
                model,
                bt_local,
                steps=int(tcfg.integration_steps),
                sigma=float(tcfg.sde_sigma),
                generator=gen_local,
            )
        c, nrm, sd = ActionTrajectoryModel.postprocess_waypoints(
            fs_local.final, xyz_mean, xyz_std, float(norm_eps)
        )
        return torch.cat([c, nrm, sd], dim=-1).view(
            -1, ActionTrajectoryModel.NUM_WAYPOINTS, 9
        )

    policy_exec_root = REPO_ROOT / "policy_exec"
    if str(policy_exec_root) not in sys.path:
        sys.path.insert(0, str(policy_exec_root))
    from dextoolbench.vec_rollout import VectorizedSimRollout

    runner = VectorizedSimRollout(
        num_envs=envs_per_task,
        config_path=_resolve(tcfg.policy_exec_config),
        checkpoint_path=_resolve(tcfg.policy_exec_checkpoint),
        object_name=str(task_spec["object_name"]),
        headless=bool(tcfg.headless),
        lift_z=float(tcfg.lift_z),
        z_offset=float(tcfg.z_offset),
        episode_length=int(tcfg.max_rollout_steps),
        table_urdf=WIDE_TABLE_URDF,
        material_radius=float(tcfg.material_radius_m),
        goal_region_radius=float(tcfg.goal_region_radius_m),
        record_video=False,
        video_num_envs=0,
        **task_spec["runner_kw"],
    )
    cl_out = run_closed_loop_rollout(
        runner,
        scenes=scenes,
        wp_world=wp_world,
        model_resample_fn=_resample_wp,
        control_frame_path=_resolve(str(task_spec["control_frame"])),
        tool_obj_path=(
            _resolve(str(task_spec["tool_obj_path"]))
            if str(task_spec["tool_obj_path"]).strip()
            else None
        ),
        task=str(task_spec["sim_task"]),
        flip_inverted_dot_max=float(tcfg.flip_inverted_dot_max),
        flip_settle_tol_m=float(tcfg.flip_settle_tol_m),
        flip_material_half_z_m=float(tcfg.flip_material_half_z_m),
        pour_settle_tol_m=float(tcfg.pour_settle_tol_m),
        pour_material_half_z_m=float(tcfg.pour_material_half_z_m),
        hammer_target_tol_m=float(tcfg.hammer_target_tol_m),
        nail_post_height_m=float(HAMMER_NAIL_POST_HEIGHT_M) if task_name == "hammer_nail" else 0.0,
        material_z_offset=(0.5 * float(HAMMER_NAIL_POST_HEIGHT_M) if task_name == "hammer_nail" else 0.0),
        steps_per_segment=int(tcfg.steps_per_segment),
        chunk_size=int(tcfg.closed_loop_chunk_size),
        max_replans=int(tcfg.closed_loop_max_replans),
        max_steps_per_chunk=int(tcfg.closed_loop_max_steps_per_chunk),
        max_total_steps=int(
            tcfg.closed_loop_max_total_steps
            if tcfg.closed_loop_max_total_steps > 0
            else tcfg.max_rollout_steps
        ),
        goal_region_half_m=float(tcfg.goal_region_radius_m),
        ball_radius_m=float(tcfg.material_radius_m),
        table_x_min_m=float(WIDE_TABLE_X_MIN_M),
        table_x_max_m=float(WIDE_TABLE_X_MAX_M),
        table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
        table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
        off_table_drop_m=float(tcfg.off_table_drop_m),
        max_stall_replans=int(tcfg.closed_loop_max_stall_replans),
        capture=False,
        capture_interval=int(tcfg.video_capture_interval),
        record_generations=False,
    )
    np.savez_compressed(
        out_path,
        tracking_frac=cl_out.tracking_frac,
        ball_start_xyz=cl_out.ball_start_xyz,
        ball_final_xyz=cl_out.ball_final_xyz,
        episode_lengths=cl_out.episode_lengths,
        material_displacement_m=cl_out.material_displacement_m,
        num_replans=cl_out.num_replans if cl_out.num_replans is not None else np.zeros(envs_per_task, dtype=np.int32),
        has_object=np.asarray([cl_out.object_quat_final is not None], dtype=np.bool_),
        object_quat_final=(
            cl_out.object_quat_final
            if cl_out.object_quat_final is not None
            else np.zeros((envs_per_task, 4), dtype=np.float32)
        ),
        object_xyz_final=(
            cl_out.object_xyz_final
            if cl_out.object_xyz_final is not None
            else np.zeros((envs_per_task, 3), dtype=np.float32)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO train action-trajectory VLA.")
    parser.add_argument("--config", type=str, default="training/cfg/action_trajectory_grpo.yaml")
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--num_scenes", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--skip_sim", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--rollout_worker_spec", type=str, default="")
    parser.add_argument("--rollout_worker_out", type=str, default="")
    args = parser.parse_args()

    if str(args.rollout_worker_spec).strip():
        if not str(args.rollout_worker_out).strip():
            raise ValueError("--rollout_worker_out is required with --rollout_worker_spec")
        _run_rollout_worker(_resolve(args.rollout_worker_spec), _resolve(args.rollout_worker_out))
        return

    grpo_cfg_path = _resolve(args.config)
    grpo_raw = yaml.safe_load(grpo_cfg_path.read_text(encoding="utf-8"))
    tcfg = GrpoTrainConfig(**{k: v for k, v in grpo_raw.items() if hasattr(GrpoTrainConfig, k)})
    if args.num_iterations is not None:
        tcfg.num_iterations = int(args.num_iterations)
    if args.num_scenes is not None:
        tcfg.num_scenes = int(args.num_scenes)
    if args.group_size is not None:
        tcfg.group_size = int(args.group_size)
    if args.skip_sim:
        tcfg.skip_sim = True
    if args.use_wandb:
        tcfg.use_wandb = True

    is_flip = str(tcfg.task_type) == "spatula_flip"
    is_pour = str(tcfg.task_type) == "spoon_pour"
    is_hammer = str(tcfg.task_type) == "hammer_nail"
    is_all_tasks = str(tcfg.task_type) == "all_tasks"

    # Per-run directory (run_00XX) shared by checkpoints, metrics, and the wandb name.
    out_dir = _next_run_dir(_resolve(tcfg.output_dir))
    _explicit_name = str(tcfg.wandb_run_name).strip()
    _prefix = str(tcfg.wandb_run_prefix).strip()
    if _explicit_name:
        run_name = _explicit_name
    elif _prefix:
        _idx = out_dir.name.split("_", 1)[1] if "_" in out_dir.name else out_dir.name
        run_name = f"{_prefix}_{_idx}"
    else:
        run_name = out_dir.name

    wandb_run = None
    if bool(tcfg.use_wandb):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "use_wandb is true but wandb is not installed. pip install wandb"
            ) from exc
        # Pull WANDB_API_KEY (+ HF token) from <repo>/.env if not already set.
        apply_hf_env()
        # Tag the run with the group name (in addition to setting wandb group).
        tags = [str(t) for t in (tcfg.wandb_tags or []) if str(t).strip()]
        if str(tcfg.wandb_group).strip():
            tags.append(str(tcfg.wandb_group).strip())
        init_kw: Dict[str, Any] = {
            "project": str(tcfg.wandb_project),
            "name": run_name,
            "config": grpo_raw,
        }
        if str(tcfg.wandb_entity).strip():
            init_kw["entity"] = str(tcfg.wandb_entity).strip()
        if str(tcfg.wandb_group).strip():
            init_kw["group"] = str(tcfg.wandb_group).strip()
        if tags:
            init_kw["tags"] = sorted(set(tags))
        if str(tcfg.wandb_notes).strip():
            init_kw["notes"] = str(tcfg.wandb_notes).strip()
        wandb_run = wandb.init(**init_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_np, std_np, norm_eps = load_xyz_normalization_stats(
        _resolve(tcfg.normalization_stats_path)
    )

    cfg, clip, model, ref_model = _load_models(
        _resolve(tcfg.config), _resolve(tcfg.checkpoint), device
    )
    optimizer = optim.AdamW(model.parameters(), lr=float(tcfg.lr), weight_decay=1e-5)

    metrics_path = out_dir / "metrics.jsonl"
    video_dir = out_dir / "videos"
    record_video = bool(tcfg.use_wandb) and int(tcfg.video_every) > 0

    batch_size = int(tcfg.num_scenes) * int(tcfg.group_size)
    repeats = max(1, int(tcfg.reward_rollout_repeats))
    sim_envs = batch_size * repeats
    sim_runner = None
    if not tcfg.skip_sim and not is_all_tasks:
        # Import here so skip_sim can run without IsaacGym.
        policy_exec_root = REPO_ROOT / "policy_exec"
        if str(policy_exec_root) not in sys.path:
            sys.path.insert(0, str(policy_exec_root))
        from dextoolbench.vec_rollout import VectorizedSimRollout

        # Each of the batch_size distinct VLA samples is rolled out `repeats`
        # times; sim env index = sample * repeats + r. Show repeat 0 of scenes
        # strided by group_size so video tiles are distinct scenes.
        flip_runner_kw: Dict[str, Any] = {}
        if is_flip:
            flip_runner_kw = {
                "material_size": list(FLIP_MATERIAL_SIZE),
                "add_pan": True,
                "pan_xyz_batch": [list(PAN_XYZ)] * sim_envs,
                "pan_radius_m": float(PAN_RADIUS_M),
                "pan_wall_height_m": float(PAN_WALL_HEIGHT_M),
                "pan_wall_thickness_m": float(PAN_WALL_THICKNESS_M),
                "pan_segments": int(PAN_SEGMENTS),
                "pan_layers": int(PAN_LAYERS),
            }
        elif is_pour:
            # Spoon scoop-and-pour: pan positions are updated per scene at reset.
            flip_runner_kw = {
                "material_size": list(POUR_MATERIAL_SIZE),
                "add_pan": True,
                "pan_xyz_batch": [list(POUR_PAN_XYZ)] * sim_envs,
                "pan_radius_m": float(PAN_RADIUS_M),
                "pan_wall_height_m": float(PAN_WALL_HEIGHT_M),
                "pan_wall_thickness_m": float(PAN_WALL_THICKNESS_M),
                "pan_segments": int(PAN_SEGMENTS),
                "pan_layers": int(PAN_LAYERS),
            }
        elif is_hammer:
            flip_runner_kw = {
                "material_size": list(HAMMER_NAIL_POST_SIZE),
                "nail_zlock": True,
                "nail_sink_max_m": 0.08,
                "nail_z_damping": 0.5,
            }
        sim_runner = None if is_all_tasks else VectorizedSimRollout(
            num_envs=sim_envs,
            config_path=_resolve(tcfg.policy_exec_config),
            checkpoint_path=_resolve(tcfg.policy_exec_checkpoint),
            object_name=str(tcfg.object_name),
            headless=bool(tcfg.headless),
            lift_z=float(tcfg.lift_z),
            z_offset=float(tcfg.z_offset),
            episode_length=int(tcfg.max_rollout_steps),
            table_urdf=WIDE_TABLE_URDF,
            material_radius=float(tcfg.material_radius_m),
            goal_region_radius=float(tcfg.goal_region_radius_m),
            record_video=record_video,
            video_num_envs=int(tcfg.video_num_envs),
            video_env_indices=[
                ((i * int(tcfg.group_size)) % batch_size) * repeats
                for i in range(int(tcfg.video_num_envs))
            ],
            **flip_runner_kw,
        )

    envs_per_task = max(1, (int(tcfg.num_scenes) // 4) * int(tcfg.group_size) * repeats)
    all_task_specs = _build_all_task_specs(tcfg, envs_per_task)
    worker_dir = out_dir / "rollout_workers"
    if is_all_tasks and not tcfg.skip_sim:
        worker_dir.mkdir(parents=True, exist_ok=True)
        if int(tcfg.num_scenes) % 4 != 0:
            raise ValueError("all_tasks requires num_scenes divisible by 4")

    reward_cfg = RewardConfig(
        w_track=float(tcfg.reward_w_track),
        w_ball=float(tcfg.reward_w_ball),
        ball_success_radius_m=float(tcfg.ball_success_radius_m),
        goal_region_half_x_m=float(tcfg.goal_region_radius_m),
        goal_region_half_y_m=float(tcfg.goal_region_radius_m),
        ball_radius_m=float(tcfg.material_radius_m),
        table_half_x_m=float(TABLE_HALF_X_M),
        table_half_y_m=float(TABLE_HALF_Y_M),
        table_x_min_m=float(WIDE_TABLE_X_MIN_M),
        table_x_max_m=float(WIDE_TABLE_X_MAX_M),
        table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
        table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
        off_table_penalty=float(tcfg.off_table_penalty),
    )
    flip_reward_cfg = FlipRewardConfig(
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
    pour_reward_cfg = PourRewardConfig(
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
    hammer_reward_cfg = HammerRewardConfig(
        w_hammer=float(tcfg.reward_w_hammer),
        target_tol_m=float(tcfg.hammer_target_tol_m),
        partial_scale=float(tcfg.hammer_partial_scale),
    )
    tool_obj_path = _resolve(tcfg.tool_obj_path) if str(tcfg.tool_obj_path).strip() else None
    dt = 1.0 / max(1, int(tcfg.integration_steps))
    rng = np.random.default_rng(int(tcfg.seed))

    for it in range(int(tcfg.num_iterations)):
        t0 = time.time()
        do_video = record_video and (it % int(tcfg.video_every) == 0)
        breakdown = None
        if is_all_tasks:
            task_order = ["brush_sweep", "spatula_flip", "spoon_pour", "hammer_nail"]
            if int(tcfg.num_scenes) % len(task_order) != 0:
                raise ValueError("all_tasks requires num_scenes divisible by 4")
            per_task = int(tcfg.num_scenes) // len(task_order)
            scenes = []
            for task_name in task_order:
                seed_i = int(rng.integers(0, 2**31))
                if task_name == "brush_sweep":
                    part = sample_rl_scenes(per_task, seed=seed_i)
                elif task_name == "spatula_flip":
                    part = sample_flip_scenes(per_task, seed=seed_i)
                elif task_name == "spoon_pour":
                    part = sample_pour_scenes(per_task, seed=seed_i)
                elif task_name == "hammer_nail":
                    part = sample_hammer_scenes(per_task, seed=seed_i)
                else:
                    raise RuntimeError(task_name)
                for sc in part:
                    sc["grpo_task_type"] = task_name
                scenes.extend(part)
        elif is_flip:
            scenes = sample_flip_scenes(
                int(tcfg.num_scenes),
                seed=int(rng.integers(0, 2**31)),
            )
            for sc in scenes:
                sc["grpo_task_type"] = "spatula_flip"
        elif is_pour:
            scenes = sample_pour_scenes(
                int(tcfg.num_scenes),
                seed=int(rng.integers(0, 2**31)),
            )
            for sc in scenes:
                sc["grpo_task_type"] = "spoon_pour"
        elif is_hammer:
            scenes = sample_hammer_scenes(
                int(tcfg.num_scenes),
                seed=int(rng.integers(0, 2**31)),
            )
            for sc in scenes:
                sc["grpo_task_type"] = "hammer_nail"
        else:
            scenes = sample_rl_scenes(
                int(tcfg.num_scenes),
                seed=int(rng.integers(0, 2**31)),
            )
            for sc in scenes:
                sc["grpo_task_type"] = "brush_sweep"
        expanded_scenes: List[Dict[str, Any]] = []
        for sc in scenes:
            for _ in range(int(tcfg.group_size)):
                expanded_scenes.append(sc)

        samples = [
            _scene_to_sample(sc, i) for i, sc in enumerate(expanded_scenes)
        ]
        ds = WaypointTrajectoryDataset(
            samples, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
        )
        batch_items = [ds[i] for i in range(len(samples))]
        collated = waypoint_collate(batch_items)
        batch_tensors = _batch_to_model_inputs(collated, clip, device)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(rng.integers(0, 2**31)))
        with torch.no_grad():
            flow_sample: FlowSampleResult = sample_with_logprobs(
                model,
                batch_tensors,
                steps=int(tcfg.integration_steps),
                sigma=float(tcfg.sde_sigma),
                generator=gen,
            )

        xyz_mean = torch.tensor(mean_np, device=device, dtype=torch.float32)
        xyz_std = torch.tensor(std_np, device=device, dtype=torch.float32)
        contact, normal, sdir = ActionTrajectoryModel.postprocess_waypoints(
            flow_sample.final,
            xyz_mean,
            xyz_std,
            float(norm_eps),
        )
        wp_world = torch.cat([contact, normal, sdir], dim=-1).view(
            -1, ActionTrajectoryModel.NUM_WAYPOINTS, 9
        )

        rewards = np.zeros(batch_size, dtype=np.float32)
        sim_out = None
        generations_by_env: Optional[Dict[int, List[ClosedLoopGenRecord]]] = None
        _extra_flow_samples: List[FlowSampleResult] = []
        if is_all_tasks and not tcfg.skip_sim:
            # Tile scenes for reward averaging (sim env = sample*repeats + r).
            tiled_scenes: List[Dict[str, Any]] = []
            tiled_wp_list: List[torch.Tensor] = []
            tiled_task_labels: List[str] = []
            for s_idx, sc in enumerate(expanded_scenes):
                task_name = str(sc.get("grpo_task_type", "brush_sweep"))
                for _ in range(repeats):
                    tiled_scenes.append(sc)
                    tiled_wp_list.append(wp_world[s_idx])
                    tiled_task_labels.append(task_name)
            tiled_wp = torch.stack(tiled_wp_list, dim=0)
            sim_batch_size = len(tiled_scenes)
            rewards_env = np.zeros(sim_batch_size, dtype=np.float32)
            breakdown_env = {
                "track": np.zeros(sim_batch_size, dtype=np.float32),
                "ball": np.zeros(sim_batch_size, dtype=np.float32),
                "total": np.zeros(sim_batch_size, dtype=np.float32),
                "ball_dist_final_m": np.zeros(sim_batch_size, dtype=np.float32),
                "ball_in_region": np.zeros(sim_batch_size, dtype=np.float32),
            }
            tracking_all = np.zeros(sim_batch_size, dtype=np.float32)
            disp_all = np.zeros(sim_batch_size, dtype=np.float32)

            worker_ckpt = worker_dir / f"iter_{it:05d}_model.pt"
            torch.save({"model": model.state_dict()}, worker_ckpt)

            def _json_default(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

            worker_jobs = []
            for task_name in all_task_specs:
                idxs = [i for i, t in enumerate(tiled_task_labels) if t == task_name]
                if not idxs:
                    continue
                sub_scenes = [tiled_scenes[i] for i in idxs]
                idx_t = torch.as_tensor(idxs, device=tiled_wp.device, dtype=torch.long)
                sub_wp = tiled_wp.index_select(0, idx_t)
                spec_path = worker_dir / f"iter_{it:05d}_{task_name}.json"
                out_path = worker_dir / f"iter_{it:05d}_{task_name}.npz"
                log_path = worker_dir / f"iter_{it:05d}_{task_name}.log"
                payload = {
                    "tcfg": asdict(tcfg),
                    "task_name": task_name,
                    "scenes": sub_scenes,
                    "wp_world": sub_wp.detach().cpu().numpy().astype(np.float32).tolist(),
                    "model_checkpoint": str(worker_ckpt),
                    "seed": int(rng.integers(0, 2**31)),
                }
                spec_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
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
                        str(out_path),
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                worker_jobs.append((task_name, idxs, out_path, log_path, proc, log_f))

            worker_results = []
            try:
                for task_name, idxs, out_path, log_path, proc, log_f in worker_jobs:
                    code = proc.wait()
                    log_f.close()
                    if code != 0:
                        raise RuntimeError(
                            f"all_tasks rollout worker {task_name} exited {code}; see {log_path}"
                        )
                    if not out_path.is_file():
                        raise FileNotFoundError(
                            f"all_tasks rollout worker {task_name} did not write {out_path}; see {log_path}"
                        )
                    worker_results.append((task_name, idxs, np.load(out_path)))
            finally:
                for _task_name, _idxs, _out_path, _log_path, proc, log_f in worker_jobs:
                    if proc.poll() is None:
                        proc.terminate()
                    if not log_f.closed:
                        log_f.close()

            class _WorkerRolloutResult:
                pass

            for task_name, idxs, data in worker_results:
                cl_out = _WorkerRolloutResult()
                cl_out.tracking_frac = data["tracking_frac"]
                cl_out.ball_start_xyz = data["ball_start_xyz"]
                cl_out.ball_final_xyz = data["ball_final_xyz"]
                cl_out.episode_lengths = data["episode_lengths"]
                cl_out.material_displacement_m = data["material_displacement_m"]
                cl_out.num_replans = data["num_replans"]
                has_object = bool(data["has_object"][0])
                cl_out.object_quat_final = data["object_quat_final"] if has_object else None
                cl_out.object_xyz_final = data["object_xyz_final"] if has_object else None
                sub_scenes = [tiled_scenes[i] for i in idxs]
                dest_np = np.array([s["destination_xyz_world"] for s in sub_scenes], dtype=np.float64)
                if task_name == "spatula_flip":
                    table_z_arr = np.array([s["table_xyz_world"][2] for s in sub_scenes], dtype=np.float64)
                    r_sub = compute_flip_reward(object_quat_final=cl_out.object_quat_final, object_xyz_final=cl_out.object_xyz_final, table_z=table_z_arr, cfg=flip_reward_cfg)
                    fb = flip_reward_breakdown(object_quat_final=cl_out.object_quat_final, object_xyz_final=cl_out.object_xyz_final, table_z=table_z_arr, cfg=flip_reward_cfg)
                    b_sub = {"track": np.zeros_like(fb["flip"]), "ball": fb["flip"], "total": fb["total"], "ball_dist_final_m": np.zeros_like(fb["flip"]), "ball_in_region": fb["flip_success"]}
                elif task_name == "spoon_pour":
                    table_z_arr = np.array([s["table_xyz_world"][2] for s in sub_scenes], dtype=np.float64)
                    r_sub = compute_pour_reward(material_start_xyz=cl_out.ball_start_xyz, material_final_xyz=cl_out.ball_final_xyz, destination_xyz=dest_np, table_z=table_z_arr, cfg=pour_reward_cfg)
                    pb = pour_reward_breakdown(material_start_xyz=cl_out.ball_start_xyz, material_final_xyz=cl_out.ball_final_xyz, destination_xyz=dest_np, table_z=table_z_arr, cfg=pour_reward_cfg)
                    b_sub = {"track": np.zeros_like(pb["pour"]), "ball": pb["pour"], "total": pb["total"], "ball_dist_final_m": pb["material_dist_final_m"], "ball_in_region": pb["pour_success"]}
                elif task_name == "hammer_nail":
                    half_post = 0.5 * float(HAMMER_NAIL_POST_HEIGHT_M)
                    head_start = np.asarray(cl_out.ball_start_xyz, dtype=np.float64).copy(); head_start[:, 2] += half_post
                    head_final = np.asarray(cl_out.ball_final_xyz, dtype=np.float64).copy(); head_final[:, 2] += half_post
                    r_sub = compute_hammer_reward(head_start_xyz=head_start, head_final_xyz=head_final, target_xyz=dest_np, cfg=hammer_reward_cfg)
                    hb = hammer_reward_breakdown(head_start_xyz=head_start, head_final_xyz=head_final, target_xyz=dest_np, cfg=hammer_reward_cfg)
                    b_sub = {"track": np.zeros_like(hb["hammer"]), "ball": hb["hammer"], "total": hb["total"], "ball_dist_final_m": (1.0 - hb["sink_frac"]).astype(np.float32), "ball_in_region": hb["hammer_success"]}
                else:
                    r_sub = compute_combined_reward(tracking_frac=cl_out.tracking_frac, ball_start_xyz=cl_out.ball_start_xyz, ball_final_xyz=cl_out.ball_final_xyz, destination_xyz=dest_np, cfg=reward_cfg)
                    b_sub = reward_breakdown(tracking_frac=cl_out.tracking_frac, ball_start_xyz=cl_out.ball_start_xyz, ball_final_xyz=cl_out.ball_final_xyz, destination_xyz=dest_np, cfg=reward_cfg)
                for local_i, global_i in enumerate(idxs):
                    rewards_env[global_i] = r_sub[local_i]
                    tracking_all[global_i] = cl_out.tracking_frac[local_i]
                    disp_all[global_i] = cl_out.material_displacement_m[local_i]
                    for k in breakdown_env:
                        breakdown_env[k][global_i] = b_sub[k][local_i]
            class _SimProxy:
                tracking_frac = tracking_all
                material_displacement_m = disp_all
                frames_by_env = None
            sim_out = _SimProxy()
            if repeats > 1:
                rewards = rewards_env.reshape(batch_size, repeats).mean(axis=1)
                breakdown = {k: v.reshape(batch_size, repeats).mean(axis=1) for k, v in breakdown_env.items()}
            else:
                rewards = rewards_env
                breakdown = breakdown_env
            rewards = rewards.astype(np.float32)
        elif sim_runner is not None:
            # Tile scenes for reward averaging (sim env = sample*repeats + r).
            tiled_scenes: List[Dict[str, Any]] = []
            tiled_wp_list: List[torch.Tensor] = []
            for s_idx, sc in enumerate(expanded_scenes):
                for _ in range(repeats):
                    tiled_scenes.append(sc)
                    tiled_wp_list.append(wp_world[s_idx])
            tiled_wp = torch.stack(tiled_wp_list, dim=0)
            sim_batch_size = len(tiled_scenes)

            def _resample_wp(scenes_list: List[Dict[str, Any]]) -> torch.Tensor:
                samples_local = [
                    _scene_to_sample(sc, i) for i, sc in enumerate(scenes_list)
                ]
                ds_local = WaypointTrajectoryDataset(
                    samples_local, xyz_mean=mean_np, xyz_std=std_np, norm_eps=float(norm_eps)
                )
                coll_local = waypoint_collate([ds_local[i] for i in range(len(samples_local))])
                bt_local = _batch_to_model_inputs(coll_local, clip, device)
                gen_local = torch.Generator(device=device)
                gen_local.manual_seed(int(rng.integers(0, 2**31)))
                with torch.no_grad():
                    fs_local: FlowSampleResult = sample_with_logprobs(
                        model,
                        bt_local,
                        steps=int(tcfg.integration_steps),
                        sigma=float(tcfg.sde_sigma),
                        generator=gen_local,
                    )
                _extra_flow_samples.append(fs_local)
                c, nrm, sd = ActionTrajectoryModel.postprocess_waypoints(
                    fs_local.final, xyz_mean, xyz_std, float(norm_eps)
                )
                return torch.cat([c, nrm, sd], dim=-1).view(
                    -1, ActionTrajectoryModel.NUM_WAYPOINTS, 9
                )

            if bool(tcfg.use_closed_loop):
                cl_out = run_closed_loop_rollout(
                    sim_runner,
                    scenes=tiled_scenes,
                    wp_world=tiled_wp,
                    model_resample_fn=_resample_wp,
                    control_frame_path=_resolve(tcfg.control_frame),
                    tool_obj_path=tool_obj_path,
                    task=("flip" if is_flip else ("pour" if is_pour else ("hammer" if is_hammer else "sweep"))),
                    flip_inverted_dot_max=float(tcfg.flip_inverted_dot_max),
                    flip_settle_tol_m=float(tcfg.flip_settle_tol_m),
                    flip_material_half_z_m=float(tcfg.flip_material_half_z_m),
                    pour_settle_tol_m=float(tcfg.pour_settle_tol_m),
                    pour_material_half_z_m=float(tcfg.pour_material_half_z_m),
                    steps_per_segment=int(tcfg.steps_per_segment),
                    chunk_size=int(tcfg.closed_loop_chunk_size),
                    max_replans=int(tcfg.closed_loop_max_replans),
                    max_steps_per_chunk=int(tcfg.closed_loop_max_steps_per_chunk),
                    max_total_steps=int(
                        tcfg.closed_loop_max_total_steps
                        if tcfg.closed_loop_max_total_steps > 0
                        else tcfg.max_rollout_steps
                    ),
                    goal_region_half_m=float(tcfg.goal_region_radius_m),
                    ball_radius_m=float(tcfg.material_radius_m),
                    table_x_min_m=float(WIDE_TABLE_X_MIN_M),
                    table_x_max_m=float(WIDE_TABLE_X_MAX_M),
                    table_y_min_m=float(WIDE_TABLE_Y_MIN_M),
                    table_y_max_m=float(WIDE_TABLE_Y_MAX_M),
                    off_table_drop_m=float(tcfg.off_table_drop_m),
                    max_stall_replans=int(tcfg.closed_loop_max_stall_replans),
                    capture=do_video,
                    capture_interval=int(tcfg.video_capture_interval),
                    record_generations=do_video,
                )
                generations_by_env = cl_out.generations_by_env
                dest_np = np.array(
                    [s["destination_xyz_world"] for s in tiled_scenes], dtype=np.float64
                )
                if is_flip:
                    table_z_arr = np.array(
                        [s["table_xyz_world"][2] for s in tiled_scenes],
                        dtype=np.float64,
                    )
                    rewards_env = compute_flip_reward(
                        object_quat_final=cl_out.object_quat_final,
                        object_xyz_final=cl_out.object_xyz_final,
                        table_z=table_z_arr,
                        cfg=flip_reward_cfg,
                    )
                    fb = flip_reward_breakdown(
                        object_quat_final=cl_out.object_quat_final,
                        object_xyz_final=cl_out.object_xyz_final,
                        table_z=table_z_arr,
                        cfg=flip_reward_cfg,
                    )
                    # Map flip components onto the shared logging schema so the
                    # downstream metrics/captions need no special-casing.
                    zeros = np.zeros_like(fb["flip"])
                    breakdown_env = {
                        "track": zeros.copy(),
                        "ball": fb["flip"],
                        "total": fb["total"],
                        "ball_dist_final_m": zeros.copy(),
                        "ball_in_region": fb["flip_success"],
                    }
                elif is_pour:
                    table_z_arr = np.array(
                        [s["table_xyz_world"][2] for s in tiled_scenes],
                        dtype=np.float64,
                    )
                    rewards_env = compute_pour_reward(
                        material_start_xyz=cl_out.ball_start_xyz,
                        material_final_xyz=cl_out.ball_final_xyz,
                        destination_xyz=dest_np,
                        table_z=table_z_arr,
                        cfg=pour_reward_cfg,
                    )
                    pb = pour_reward_breakdown(
                        material_start_xyz=cl_out.ball_start_xyz,
                        material_final_xyz=cl_out.ball_final_xyz,
                        destination_xyz=dest_np,
                        table_z=table_z_arr,
                        cfg=pour_reward_cfg,
                    )
                    zeros = np.zeros_like(pb["pour"])
                    breakdown_env = {
                        "track": zeros.copy(),
                        "ball": pb["pour"],
                        "total": pb["total"],
                        "ball_dist_final_m": pb["material_dist_final_m"],
                        "ball_in_region": pb["pour_success"],
                    }
                elif is_hammer:
                    half_post = 0.5 * float(HAMMER_NAIL_POST_HEIGHT_M)
                    head_start = np.asarray(cl_out.ball_start_xyz, dtype=np.float64).copy()
                    head_final = np.asarray(cl_out.ball_final_xyz, dtype=np.float64).copy()
                    head_start[:, 2] += half_post
                    head_final[:, 2] += half_post
                    rewards_env = compute_hammer_reward(
                        head_start_xyz=head_start,
                        head_final_xyz=head_final,
                        target_xyz=dest_np,
                        cfg=hammer_reward_cfg,
                    )
                    hb = hammer_reward_breakdown(
                        head_start_xyz=head_start,
                        head_final_xyz=head_final,
                        target_xyz=dest_np,
                        cfg=hammer_reward_cfg,
                    )
                    zeros = np.zeros_like(hb["hammer"])
                    breakdown_env = {
                        "track": zeros.copy(),
                        "ball": hb["hammer"],
                        "total": hb["total"],
                        "ball_dist_final_m": (1.0 - hb["sink_frac"]).astype(np.float32),
                        "ball_in_region": hb["hammer_success"],
                    }
                else:
                    rewards_env = compute_combined_reward(
                        tracking_frac=cl_out.tracking_frac,
                        ball_start_xyz=cl_out.ball_start_xyz,
                        ball_final_xyz=cl_out.ball_final_xyz,
                        destination_xyz=dest_np,
                        cfg=reward_cfg,
                    )
                    breakdown_env = reward_breakdown(
                        tracking_frac=cl_out.tracking_frac,
                        ball_start_xyz=cl_out.ball_start_xyz,
                        ball_final_xyz=cl_out.ball_final_xyz,
                        destination_xyz=dest_np,
                        cfg=reward_cfg,
                    )
                class _SimProxy:
                    tracking_frac = cl_out.tracking_frac
                    material_displacement_m = cl_out.material_displacement_m
                    frames_by_env = cl_out.frames_by_env

                sim_out = _SimProxy()
            else:
                goals_b, start_b, mat_b, dest_b, _ = waypoints_tensor_to_sim_batch(
                    tiled_wp,
                    tiled_scenes,
                    control_frame_path=_resolve(tcfg.control_frame),
                    steps_per_segment=int(tcfg.steps_per_segment),
                )
                sim_runner.apply_scene_batch(goals_b, start_b, mat_b, dest_b)
                open_out = sim_runner.roll_until_done(
                    max_steps=int(tcfg.max_rollout_steps),
                    capture=do_video,
                    capture_interval=int(tcfg.video_capture_interval),
                )
                sim_out = open_out
                dest_np = dest_b.cpu().numpy()
                rewards_env = compute_combined_reward(
                    tracking_frac=open_out.tracking_frac,
                    ball_start_xyz=open_out.ball_start_xyz,
                    ball_final_xyz=open_out.ball_final_xyz,
                    destination_xyz=dest_np,
                    cfg=reward_cfg,
                )
                breakdown_env = reward_breakdown(
                    tracking_frac=open_out.tracking_frac,
                    ball_start_xyz=open_out.ball_start_xyz,
                    ball_final_xyz=open_out.ball_final_xyz,
                    destination_xyz=dest_np,
                    cfg=reward_cfg,
                )
            # Average the `repeats` rolls of each sample into one denoised reward.
            if repeats > 1:
                rewards = rewards_env.reshape(batch_size, repeats).mean(axis=1)
                breakdown = {
                    k: v.reshape(batch_size, repeats).mean(axis=1)
                    for k, v in breakdown_env.items()
                }
            else:
                rewards = rewards_env
                breakdown = breakdown_env
            rewards = rewards.astype(np.float32)
        else:
            # Proxy reward from predicted sweep geometry when sim is skipped.
            mat = np.array([s["material_xyz_world"] for s in expanded_scenes])
            dest = np.array([s["destination_xyz_world"] for s in expanded_scenes])
            wp = wp_world.detach().cpu().numpy()
            touch = wp[:, 2, :3]
            d_touch = np.linalg.norm(touch - mat, axis=1)
            d_end = np.linalg.norm(wp[:, 5, :3] - dest, axis=1)
            rewards = np.exp(-3.0 * d_touch) + np.exp(-3.0 * d_end)

        advantages = group_relative_advantages(
            rewards, group_size=int(tcfg.group_size)
        )
        adv_t = torch.tensor(advantages, device=device, dtype=torch.float32)

        # Policy gradient on the initial plan; closed-loop reward reflects replans.
        _ = _extra_flow_samples
        old_logprob = flow_sample.logprob_sum.detach()
        policy_loss_val = 0.0
        kl_val = 0.0
        for _inner in range(int(tcfg.inner_epochs)):
            optimizer.zero_grad()
            new_logprob = recompute_logprob_sum(
                model,
                batch_tensors,
                flow_sample,
                sigma=float(tcfg.sde_sigma),
                dt=dt,
            )
            p_loss = grpo_policy_loss(
                new_logprob=new_logprob,
                old_logprob=old_logprob,
                advantage=adv_t,
                clip_ratio=float(tcfg.grpo_clip),
            )
            k_loss = kl_ref_loss(model, ref_model, batch_tensors, flow_sample)
            loss = p_loss + float(tcfg.kl_coef) * k_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_loss_val = float(p_loss.item())
            kl_val = float(k_loss.item())

        # Mean within-group reward spread: if ~0, rollouts in each scene are
        # identical and GRPO advantages collapse (nothing to learn from).
        _grp = rewards.reshape(int(tcfg.num_scenes), int(tcfg.group_size))
        within_group_std_mean = float(_grp.std(axis=1).mean())

        row = {
            "iteration": it,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "within_group_std_mean": within_group_std_mean,
            "adv_std": float(advantages.std()),
            "policy_loss": policy_loss_val,
            "kl_loss": kl_val,
            "elapsed_sec": time.time() - t0,
            "tracking_mean": float(sim_out.tracking_frac.mean())
            if sim_out is not None
            else 0.0,
            "material_disp_mean_cm": float(sim_out.material_displacement_m.mean() * 100)
            if sim_out is not None
            else 0.0,
        }
        # Break down the reward into its trajectory-tracking and ball-placement
        # parts, and log the absolute ball-to-goal distance, every iteration.
        if breakdown is not None:
            row["reward_track_mean"] = float(np.mean(breakdown["track"]))
            row["reward_ball_mean"] = float(np.mean(breakdown["ball"]))
            row["ball_dist_final_mean_cm"] = float(
                np.mean(breakdown["ball_dist_final_m"]) * 100.0
            )
            row["ball_dist_final_min_cm"] = float(
                np.min(breakdown["ball_dist_final_m"]) * 100.0
            )
            row["ball_in_region_frac"] = float(np.mean(breakdown["ball_in_region"]))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if wandb_run is not None:
            import wandb

            log_payload: Dict[str, Any] = {
                k: v for k, v in row.items() if k != "iteration"
            }
            # Log the iteration count as its own graph (vs the wandb step).
            log_payload["iteration"] = int(it)
            # Qualitative: one video per shown env (distinct scenes), with the
            # dataset-style trajectory render concatenated side-by-side (left =
            # sim rollout, right = predicted trajectory viz).
            if (
                do_video
                and sim_out is not None
                and sim_out.frames_by_env
                and breakdown is not None
            ):
                try:
                    import imageio
                    from PIL import Image

                    video_dir.mkdir(parents=True, exist_ok=True)
                    shown_envs = sorted(sim_out.frames_by_env.keys())
                    # Sim env index -> per-sample index (samples are tiled by
                    # `repeats` for averaging: sim env = sample * repeats + r).
                    env_to_sample = {e: e // repeats for e in shown_envs}
                    # Render the trajectory viz (no fake surfaces) for each env.
                    viz_scenes = tiled_scenes if repeats > 1 else expanded_scenes
                    viz_wp = (
                        tiled_wp.detach().cpu().numpy()
                        if repeats > 1
                        else wp_world.detach().cpu().numpy()
                    )
                    viz_env_to_sample = (
                        {e: e for e in shown_envs}
                        if repeats > 1
                        else env_to_sample
                    )
                    traj_paths = _render_trajectory_images(
                        shown_envs=shown_envs,
                        env_to_sample=viz_env_to_sample,
                        scenes=viz_scenes,
                        wp_world_np=viz_wp,
                        out_dir=video_dir,
                        it=it,
                        generations_by_env=generations_by_env,
                    )
                    for e in shown_envs:
                        frames = sim_out.frames_by_env.get(e) or []
                        if not frames:
                            continue
                        fh = frames[0].shape[0]
                        traj_arr = None
                        png = traj_paths.get(e)
                        if png is not None and png.exists():
                            timg = Image.open(png).convert("RGB")
                            tw = int(round(timg.width * fh / timg.height))
                            traj_arr = np.asarray(
                                timg.resize((tw, fh), Image.BILINEAR)
                            )
                        combined = []
                        for fr in frames:
                            if traj_arr is not None:
                                combined.append(
                                    np.concatenate([fr, traj_arr], axis=1)
                                )
                            else:
                                combined.append(fr)
                        vid_path = video_dir / f"iter_{it:05d}_env{e:02d}.mp4"
                        imageio.mimsave(
                            str(vid_path), combined, fps=int(tcfg.video_fps)
                        )
                        s = env_to_sample[e]
                        prompt = str(
                            expanded_scenes[s].get("instruction", "")
                        ).strip()
                        in_frac = float(breakdown["ball_in_region"][s])
                        caption = (
                            f'prompt: "{prompt}"\n'
                            f"iter {it} | env{e} scene{s // int(tcfg.group_size)} | "
                            f"reward {float(breakdown['total'][s]):.3f} "
                            f"(traj_follow {float(breakdown['track'][s]):.3f} + "
                            f"ball {float(breakdown['ball'][s]):.3f}, "
                            f"ball_dist {float(breakdown['ball_dist_final_m'][s]) * 100:.1f}cm, "
                            f"in_region {in_frac * 100:.0f}% of {repeats}) "
                            f"| loss {policy_loss_val:.4f} | kl {kl_val:.4f}"
                        )
                        log_payload[f"rollout/env{e}"] = wandb.Video(
                            str(vid_path),
                            fps=int(tcfg.video_fps),
                            format="mp4",
                            caption=caption,
                        )
                except Exception as exc:  # pragma: no cover
                    print(f"[warn] video log failed at iter {it}: {exc}")
            wandb.log(log_payload, step=int(it))
        print(
            f"iter {it}: reward={row['reward_mean']:.3f} "
            f"[min={row['reward_min']:.3f} max={row['reward_max']:.3f} "
            f"grp_std={within_group_std_mean:.3f}] "
            f"loss={policy_loss_val:.4f} kl={kl_val:.4f} "
            f"track={row['tracking_mean']:.2f} "
            f"t={row['elapsed_sec']:.1f}s"
        )

        if (it + 1) % 10 == 0:
            torch.save(
                {
                    "iteration": it,
                    "model": model.state_dict(),
                    "config": grpo_raw,
                },
                out_dir / f"checkpoint_iter_{it + 1:05d}.pt",
            )

    torch.save({"model": model.state_dict(), "config": grpo_raw}, out_dir / "checkpoint_final.pt")
    if wandb_run is not None:
        import wandb

        wandb.finish()
    print(f"Done. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
