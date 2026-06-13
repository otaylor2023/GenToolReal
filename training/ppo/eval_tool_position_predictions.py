"""Evaluate tool-position policy on random held-out samples with overlays.

Writes one folder per sample containing:
- render.png (RGB with tool/goal/predicted markers)
- sample.json (input/output for that sample)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from training.ppo.train_tool_position_ppo import (
    ToolPositionActorCritic,
    _format_text_context,
)


@dataclass
class EvalRecord:
    scene_id: str
    image_path: Path
    intrinsics: Dict[str, float]
    world_from_camera: np.ndarray | None
    canonicalization: Dict[str, Any]
    keypoints: Dict[str, Dict[str, Any]]
    tool_keypoint_id: str
    instruction: str
    relation_string: str
    movement_token: str
    constraint_type: str
    goal_xyz_canonical: np.ndarray
    tool_xyz_canonical: np.ndarray
    all_keypoints_label_position: List[Dict[str, Any]]
    task_prompt: str
    world_scale_prompt: str


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pt found under {run_dir}")
    return ckpts[-1]


def _split_shards(
    dataset_dir: Path,
    split: str,
    *,
    train_fraction: float,
    seed: int,
) -> List[Path]:
    shard_paths = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No dataset shards found in {dataset_dir}")
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(shard_paths))
    rng.shuffle(idxs)
    cutoff = max(1, int(len(idxs) * train_fraction))
    if split == "train":
        keep = set(idxs[:cutoff].tolist())
    else:
        keep = set(idxs[cutoff:].tolist()) or set(idxs[-1:].tolist())
    return [p for i, p in enumerate(shard_paths) if i in keep]


def _build_records_from_shard(
    shard: Dict[str, Any],
    *,
    task_prompt: str,
    world_scale_prompt: str,
) -> List[EvalRecord]:
    keypoints = shard["keypoints"]
    out: List[EvalRecord] = []
    for dp in shard.get("datapoints", []):
        tool_id = str(dp.get("tool_keypoint_id", ""))
        if tool_id not in keypoints:
            continue
        tool_kp = keypoints[tool_id]
        if not tool_kp.get("valid", False):
            continue
        instructions = dp.get("instructions") or []
        if not instructions:
            continue
        tool_xyz = tool_kp.get("xyz_world")
        goal_xyz = dp.get("goal_tool_keypoint_xyz_world")
        if not isinstance(tool_xyz, list) or len(tool_xyz) != 3:
            continue
        if not isinstance(goal_xyz, list) or len(goal_xyz) != 3:
            continue
        all_kps = [
            {
                "label": str(v.get("label", "")),
                "object_name": str(v.get("object_name", "")).strip(),
                "position_xyz_world": v.get("xyz_world"),
            }
            for v in keypoints.values()
            if v.get("valid", False) and v.get("xyz_world") is not None
        ]
        out.append(
            EvalRecord(
                scene_id=str(shard["scene_id"]),
                image_path=Path(str(shard["image"])),
                intrinsics=dict(shard["camera"]["intrinsics"]),
                world_from_camera=np.asarray(
                    shard["camera"].get("world_from_camera"), dtype=np.float64
                )
                if shard["camera"].get("world_from_camera") is not None
                else None,
                canonicalization=dict(shard.get("canonicalization", {}) or {}),
                keypoints=keypoints,
                tool_keypoint_id=tool_id,
                instruction=str(instructions[0]),
                relation_string=str(dp.get("relation_string", "")),
                movement_token=str(dp.get("movement_token", "")),
                constraint_type=str(dp.get("constraint_type", "")),
                goal_xyz_canonical=np.asarray(goal_xyz, dtype=np.float32),
                tool_xyz_canonical=np.asarray(tool_xyz, dtype=np.float32),
                all_keypoints_label_position=all_kps,
                task_prompt=task_prompt,
                world_scale_prompt=world_scale_prompt,
            )
        )
    return out


def _canonical_to_original_world(
    xyz_canonical: np.ndarray, canonicalization: Dict[str, Any]
) -> np.ndarray:
    look_at = canonicalization.get("look_at_xyz_m")
    if not isinstance(look_at, list) or len(look_at) != 3:
        return xyz_canonical.astype(np.float64)
    x_w = float(xyz_canonical[0]) + float(look_at[0])
    y_w = float(xyz_canonical[1]) + float(look_at[1])
    z_origin = str(canonicalization.get("z_origin", ""))
    if z_origin == "tabletop_z0":
        table_z = canonicalization.get("table_z_world")
        z_w = float(xyz_canonical[2]) + float(table_z)
    else:
        z_shift = float(canonicalization.get("z_shift_applied_m", 0.0))
        z_w = float(xyz_canonical[2]) + float(look_at[2]) + z_shift
    return np.asarray([x_w, y_w, z_w], dtype=np.float64)


def _project_with_signs(
    xyz_world: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: Dict[str, float],
    *,
    sx: float,
    sy: float,
) -> Tuple[float, float] | None:
    camera_from_world = np.linalg.inv(world_from_camera)
    p_w = np.asarray([xyz_world[0], xyz_world[1], xyz_world[2], 1.0], dtype=np.float64)
    p_c = camera_from_world @ p_w
    xc = float(p_c[0])
    yc = float(p_c[1])
    zc = float(p_c[2])
    if zc <= 1e-6:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = fx * (sx * xc / zc) + cx
    v = fy * (sy * yc / zc) + cy
    return (u, v)


def _select_projection_signs(record: EvalRecord) -> Tuple[float, float]:
    if record.world_from_camera is None:
        return (1.0, 1.0)
    kp = record.keypoints.get(record.tool_keypoint_id, {})
    uv_ref = kp.get("uv_px")
    if not (isinstance(uv_ref, list) and len(uv_ref) == 2):
        return (1.0, 1.0)
    uv_ref = (float(uv_ref[0]), float(uv_ref[1]))
    tool_world = _canonical_to_original_world(
        record.tool_xyz_canonical, record.canonicalization
    )
    best_err = 1e18
    best = (1.0, 1.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            uv = _project_with_signs(
                tool_world, record.world_from_camera, record.intrinsics, sx=sx, sy=sy
            )
            if uv is None:
                continue
            err = (uv[0] - uv_ref[0]) ** 2 + (uv[1] - uv_ref[1]) ** 2
            if err < best_err:
                best_err = err
                best = (sx, sy)
    return best


def _project_canonical_to_uv(
    xyz_canonical: np.ndarray,
    record: EvalRecord,
    *,
    sx: float,
    sy: float,
) -> Tuple[float, float] | None:
    if record.world_from_camera is None:
        return None
    xyz_world = _canonical_to_original_world(xyz_canonical, record.canonicalization)
    return _project_with_signs(
        xyz_world, record.world_from_camera, record.intrinsics, sx=sx, sy=sy
    )


def _build_text_context(record: EvalRecord) -> str:
    tool_label = str(record.keypoints[record.tool_keypoint_id].get("label", ""))
    tool_obj = str(record.keypoints[record.tool_keypoint_id].get("object_name", "")).strip()
    full_tool_label = f"{tool_label} of {tool_obj}" if tool_obj else tool_label
    return _format_text_context(
        task_prompt=record.task_prompt,
        world_scale_prompt=record.world_scale_prompt,
        instruction=record.instruction,
        tool_keypoint_label=full_tool_label,
        tool_keypoint_xyz_world=record.tool_xyz_canonical.tolist(),
        all_keypoints_label_position=record.all_keypoints_label_position,
    )


def _tensor_obs(image_path: Path, image_size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((image_size_hw[1], image_size_hw[0]))
    arr = np.asarray(img, dtype=np.uint8)
    return torch.as_tensor(arr[None, ...], dtype=torch.float32, device=device)


def _draw_marker(draw: ImageDraw.ImageDraw, uv: Tuple[float, float], color: Tuple[int, int, int], label: str) -> None:
    x, y = float(uv[0]), float(uv[1])
    r = 8
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, fill=color, width=3)
    draw.text((x + 10, y - 10), label, fill=color)


def _draw_offscreen_marker(
    draw: ImageDraw.ImageDraw,
    uv: Tuple[float, float],
    image_wh: Tuple[int, int],
    color: Tuple[int, int, int],
    label: str,
) -> None:
    w, h = image_wh
    x = min(max(float(uv[0]), 0.0), float(w - 1))
    y = min(max(float(uv[1]), 0.0), float(h - 1))
    r = 8
    draw.rectangle((x - r, y - r, x + r, y + r), outline=color, fill=color, width=3)
    draw.text((x + 10, y - 10), f"{label} offscreen", fill=color)


def _draw_warning_banner(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    image_wh: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    w, _ = image_wh
    pad = 8
    h = 30
    draw.rectangle((0, 0, w, h + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO predictor on held-out samples.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory with metadata/checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override dataset dir.")
    parser.add_argument("--split", type=str, default="eval", choices=["train", "eval"], help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=24, help="Random samples to evaluate.")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed.")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="Shard split fraction used in training.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--image-size", type=int, default=224, help="Model image size.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir for renders + jsonl.")
    parser.add_argument(
        "--clip-to-workspace",
        action="store_true",
        default=True,
        help="Clip predicted XYZ to env workspace bounds before projection.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.run_dir is None:
        raise ValueError("Provide --run-dir or --checkpoint")

    checkpoint_path = args.checkpoint
    run_dir = args.run_dir
    if checkpoint_path is None:
        assert run_dir is not None
        checkpoint_path = _discover_latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(checkpoint.get("config", {}) or {})

    if run_dir is None:
        run_dir = checkpoint_path.parent
    metadata_path = run_dir / "metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    dataset_dir = args.dataset_dir or Path(
        str(cfg.get("dataset_dir") or metadata.get("dataset_dir", ""))
    )
    if not dataset_dir:
        raise ValueError("Could not resolve dataset_dir from checkpoint/config/metadata")

    task_prompt = str(
        cfg.get(
            "task_prompt_template",
            "You are a robot tool-positioning policy.",
        )
    )
    world_scale_prompt = str(cfg.get("world_scale_prompt_template", ""))

    shard_paths = _split_shards(
        dataset_dir,
        args.split,
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
    )
    records: List[EvalRecord] = []
    for sp in shard_paths:
        records.extend(
            _build_records_from_shard(
                _load_json(sp),
                task_prompt=task_prompt,
                world_scale_prompt=world_scale_prompt,
            )
        )
    if not records:
        raise RuntimeError("No records found for requested split.")

    random.seed(args.seed)
    random.shuffle(records)
    records = records[: max(1, min(int(args.num_samples), len(records)))]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ToolPositionActorCritic(
        use_real_qwen=bool(cfg.get("use_real_qwen", True)),
        vl_model_id=str(cfg.get("vl_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        device=device,
        hf_cache_dir=str(cfg.get("hf_cache_dir", "/home/ubuntu/.cache/huggingface")),
        qwen_local_files_only=bool(cfg.get("qwen_local_files_only", False)),
        qwen_forward_chunk_size=int(cfg.get("qwen_forward_chunk_size", 0)),
    ).to(device)
    model.action_head.load_state_dict(checkpoint["model_trainable_state"]["action_head"])
    model.value_head.load_state_dict(checkpoint["model_trainable_state"]["value_head"])
    model.log_std.data.copy_(checkpoint["model_trainable_state"]["log_std"].to(device))
    model.eval()
    workspace_min = np.asarray([-1.5, -1.5, -0.2], dtype=np.float32)
    workspace_max = np.asarray([1.5, 1.5, 1.8], dtype=np.float32)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (run_dir / f"eval_preview_{args.split}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs: List[str] = []
    with torch.no_grad():
        for idx, rec in enumerate(records):
            text_context = _build_text_context(rec)
            obs = {
                "image": _tensor_obs(rec.image_path, (args.image_size, args.image_size), device),
                "text_context": [text_context],
            }
            fused = model._fused(obs)
            pred_xyz = model.action_head(fused).detach().cpu().numpy()[0].astype(float)
            pred_xyz_arr = np.asarray(pred_xyz, dtype=np.float32)
            if bool(args.clip_to_workspace):
                pred_xyz_arr = np.clip(pred_xyz_arr, workspace_min, workspace_max)
            err_m = float(np.linalg.norm(pred_xyz_arr - rec.goal_xyz_canonical))

            full_img = Image.open(rec.image_path).convert("RGB")
            draw = ImageDraw.Draw(full_img)

            sx, sy = _select_projection_signs(rec)
            tool_kp = rec.keypoints.get(rec.tool_keypoint_id, {})
            tool_uv = tool_kp.get("uv_px")
            if isinstance(tool_uv, list) and len(tool_uv) == 2:
                tool_uv_xy = (float(tool_uv[0]), float(tool_uv[1]))
                _draw_marker(draw, tool_uv_xy, (33, 150, 243), "tool")

            goal_uv = _project_canonical_to_uv(rec.goal_xyz_canonical, rec, sx=sx, sy=sy)
            if goal_uv is not None:
                _draw_marker(draw, goal_uv, (76, 175, 80), "goal")
            pred_uv = _project_canonical_to_uv(pred_xyz_arr, rec, sx=sx, sy=sy)
            pred_offscreen = False
            if pred_uv is not None:
                w, h = full_img.size
                if 0.0 <= float(pred_uv[0]) < float(w) and 0.0 <= float(pred_uv[1]) < float(h):
                    _draw_marker(draw, pred_uv, (233, 30, 99), "pred")
                else:
                    pred_offscreen = True
                    _draw_offscreen_marker(draw, pred_uv, (w, h), (233, 30, 99), "pred")
            if pred_offscreen:
                _draw_warning_banner(
                    draw,
                    "PREDICTION IS OFFSCREEN (projected outside image bounds)",
                    image_wh=full_img.size,
                    color=(255, 64, 64),
                )

            sample_dir = out_dir / f"sample_{idx:04d}_{rec.scene_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            render_path = sample_dir / "render.png"
            full_img.save(render_path)

            row = {
                "sample_index": idx,
                "scene_id": rec.scene_id,
                "split": args.split,
                "checkpoint": str(checkpoint_path),
                "image_path": str(rec.image_path),
                "render_path": str(render_path),
                "instruction": rec.instruction,
                "relation_string": rec.relation_string,
                "movement_token": rec.movement_token,
                "constraint_type": rec.constraint_type,
                "input": {
                    "task_prompt": rec.task_prompt,
                    "world_scale_prompt": rec.world_scale_prompt,
                    "tool_keypoint_id": rec.tool_keypoint_id,
                    "tool_keypoint_label": str(tool_kp.get("label", "")),
                    "tool_object_name": str(tool_kp.get("object_name", "")),
                    "tool_keypoint_xyz_world": rec.tool_xyz_canonical.tolist(),
                    "all_keypoints_label_position": rec.all_keypoints_label_position,
                },
                "output": {
                    "predicted_goal_tool_keypoint_xyz_world": pred_xyz_arr.tolist(),
                    "target_goal_tool_keypoint_xyz_world": rec.goal_xyz_canonical.tolist(),
                    "l2_error_m": err_m,
                },
                "overlay_uv": {
                    "tool_uv_px": tool_uv if isinstance(tool_uv, list) else None,
                    "goal_uv_px": [float(goal_uv[0]), float(goal_uv[1])] if goal_uv is not None else None,
                    "pred_uv_px": [float(pred_uv[0]), float(pred_uv[1])] if pred_uv is not None else None,
                    "projection_signs": {"sx": sx, "sy": sy},
                },
            }
            (sample_dir / "sample.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            sample_dirs.append(str(sample_dir))

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "num_samples": len(records),
        "output_dir": str(out_dir),
        "sample_dirs": sample_dirs,
        "heldout_info": "This repo currently uses train/eval split from shard-level holdout (no separate dedicated test split).",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
