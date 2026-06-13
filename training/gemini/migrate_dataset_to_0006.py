"""Migrate dataset_0005 shards to dataset_0006 with corrected lifting/projection.

This script:
1) Re-lifts keypoints from UV + depth using:
      dir_c = normalize([(u-cx)/fx, -(v-cy)/fy, -1])
      p_w = R @ (dir_c * depth) + t
   where R,t come from the global runs_0036 root camera matrix.
2) Recomputes datapoint goals using token-aware deltas from updated logic.
3) Writes migrated shards into a new dataset folder.
4) Produces spot-check overlays for a few scenes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path("/home/ubuntu/Generative_STR")
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "training/datasets/dataset_0005_canonical_tablez0_s50_completed_only/position_shards_completed_only"
)
DEFAULT_OUTPUT_DATASET = REPO_ROOT / "training/datasets/dataset_0006_canonical_tablez0_s50_completed_only"
DEFAULT_GLOBAL_CAMERA = REPO_ROOT / "isaaclab_simtoolreal/runs_0036/camera.json"
DEFAULT_SPOTCHECK_DIR = REPO_ROOT / "training/verification/dataset_0006_spotcheck"
TOKENS = ("near", "next_to", "above", "exact_above", "over", "left", "right", "in_front", "behind", "between")


@dataclass
class SceneStats:
    scene_id: str
    keypoints_total: int
    keypoints_relifted: int
    goals_updated: int
    missing_depth_points: int


def _load_global_camera(path: Path) -> tuple[np.ndarray, Dict[str, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    key = "world_from_camera_exported" if "world_from_camera_exported" in obj else "world_from_camera"
    w = np.asarray(obj[key], dtype=np.float64)
    if w.shape != (4, 4):
        raise ValueError(f"{key} in {path} must be 4x4")
    fx, fy, cx, cy = [float(x) for x in obj["intrinsics_fx_fy_cx_cy_px"][:4]]
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return w, intr


def _goal_offset_for_token(token: str, dx: float, dy: float, dz: float) -> List[float]:
    z_lift = abs(float(dz))
    x_step = abs(float(dx))
    y_step = abs(float(dy))
    if x_step <= 1e-8:
        x_step = 0.05
    if y_step <= 1e-8:
        y_step = 0.05
    if z_lift <= 1e-8:
        z_lift = 0.05
    if token == "left":
        return [-x_step, 0.0, z_lift]
    if token == "right":
        return [x_step, 0.0, z_lift]
    if token == "in_front":
        return [0.0, -y_step, z_lift]
    if token == "behind":
        return [0.0, y_step, z_lift]
    return [0.0, 0.0, z_lift]


def _lift_world(u: int, v: int, depth_m: float, intr: Dict[str, float], wfc: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    dir_c = np.array([(float(u) - cx) / fx, -(float(v) - cy) / fy, -1.0], dtype=np.float64)
    dir_c /= max(np.linalg.norm(dir_c), 1e-12)
    r = wfc[:3, :3]
    t = wfc[:3, 3]
    return r @ (dir_c * float(depth_m)) + t


def _parse_scene_id_from_shard(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_position_dataset_v0_1"):
        return stem[: -len("_position_dataset_v0_1")]
    return stem


def _migrate_one_shard(shard_path: Path, out_dir: Path, wfc: np.ndarray, intr: Dict[str, float]) -> SceneStats:
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    scene_id = str(shard.get("scene_id") or _parse_scene_id_from_shard(shard_path))
    depth_path = Path(str(shard["depth"]))
    if not depth_path.is_absolute():
        depth_path = (REPO_ROOT / depth_path).resolve()
    depth = np.load(depth_path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float64, copy=False)
    h, w = depth.shape[:2]

    keypoints = shard.get("keypoints", {}) or {}
    relifted = 0
    missing_depth = 0
    for kp in keypoints.values():
        uv = kp.get("uv_px")
        if not isinstance(uv, list) or len(uv) != 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if not (0 <= u < w and 0 <= v < h):
            kp["valid"] = False
            kp["xyz_world"] = None
            missing_depth += 1
            continue
        d = float(depth[v, u])
        if not np.isfinite(d) or d <= 0.0:
            kp["valid"] = False
            kp["xyz_world"] = None
            missing_depth += 1
            continue
        xyz = _lift_world(u, v, d, intr, wfc)
        kp["xyz_world"] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        kp["valid"] = True
        relifted += 1

    # Update goals with new deltas and relifted refs.
    dx, dy, dz = [float(x) for x in (shard.get("movement_rules", {}).get("delta_world") or [0.0, 0.0, 0.05])]
    goals_updated = 0
    for dp in shard.get("datapoints", []) or []:
        token = str(dp.get("movement_token", "")).strip()
        refs = [r for r in (dp.get("ref_keypoint_ids") or []) if r in keypoints and keypoints[r].get("valid")]
        if token == "between":
            if len(refs) < 2:
                continue
            a = np.asarray(keypoints[refs[0]]["xyz_world"], dtype=np.float64)
            b = np.asarray(keypoints[refs[1]]["xyz_world"], dtype=np.float64)
            g = (a + b) * 0.5
            g[2] = g[2] + abs(dz if abs(dz) > 1e-8 else 0.05)
            dp["goal_tool_keypoint_xyz_world"] = [float(g[0]), float(g[1]), float(g[2])]
            goals_updated += 1
            continue
        if len(refs) < 1:
            continue
        r = np.asarray(keypoints[refs[0]]["xyz_world"], dtype=np.float64)
        if token == "exact_above":
            z_target = _goal_offset_for_token(token, dx, dy, dz)[2]
            g = np.array([r[0], r[1], r[2] + z_target], dtype=np.float64)
        else:
            off = np.asarray(_goal_offset_for_token(token, dx, dy, dz), dtype=np.float64)
            g = r + off
        dp["goal_tool_keypoint_xyz_world"] = [float(g[0]), float(g[1]), float(g[2])]
        goals_updated += 1

    # Update camera block with global transform used.
    shard.setdefault("camera", {})
    shard["camera"]["intrinsics"] = {
        "fx": intr["fx"],
        "fy": intr["fy"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "width": int(w),
        "height": int(h),
    }
    shard["camera"]["world_from_camera"] = wfc.tolist()
    shard["migration_v0006"] = {
        "type": "global_runs0036_pose_relift_with_token_goal_offsets",
        "unprojection": "dir=normalize([(u-cx)/fx, -(v-cy)/fy, -1]); p_world=R@(dir*depth)+t",
        "camera_source": str(DEFAULT_GLOBAL_CAMERA),
    }

    out_path = out_dir / shard_path.name
    out_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")
    return SceneStats(
        scene_id=scene_id,
        keypoints_total=len(keypoints),
        keypoints_relifted=relifted,
        goals_updated=goals_updated,
        missing_depth_points=missing_depth,
    )


def _project_world(xyz: List[float], cfw: np.ndarray, intr: Dict[str, float], wh: tuple[int, int]) -> List[int] | None:
    p = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    pc = cfw @ p
    d = -float(pc[2])
    if d <= 1e-9:
        return None
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    u = int((pc[0] / d) * fx + cx)
    v = int(cy - (pc[1] / d) * fy)
    if not (0 <= u < wh[0] and 0 <= v < wh[1]):
        return None
    return [u, v]


def _render_spotchecks(
    out_shard_dir: Path,
    out_vis_dir: Path,
    scenes: List[str],
) -> Dict[str, Any]:
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for scene_id in scenes:
        shard_path = out_shard_dir / f"{scene_id}_position_dataset_v0_1.json"
        if not shard_path.is_file():
            continue
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        img_path = Path(str(shard["image"]))
        if not img_path.is_absolute():
            img_path = (REPO_ROOT / img_path).resolve()
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        wh = img.size
        cfw = np.linalg.inv(np.asarray(shard["camera"]["world_from_camera"], dtype=np.float64))
        intr = {k: float(v) for k, v in shard["camera"]["intrinsics"].items() if k in {"fx", "fy", "cx", "cy"}}
        keypoints = shard.get("keypoints", {}) or {}

        scene_entries = []
        for token in TOKENS:
            match = None
            for i, dp in enumerate(shard.get("datapoints", []) or []):
                if str(dp.get("movement_token", "")).strip() == token:
                    match = (i, dp)
                    break
            if match is None:
                continue
            idx, dp = match
            tool_id = dp["tool_keypoint_id"]
            refs = list(dp.get("ref_keypoint_ids") or [])
            if tool_id not in keypoints:
                continue

            frame = img.copy()
            dr = ImageDraw.Draw(frame, "RGBA")

            # Relevant object keypoints (white) for referenced object ids.
            ref_obj_ids = {keypoints[r]["object_id"] for r in refs if r in keypoints}
            for kid, kp in keypoints.items():
                if kp.get("object_id") in ref_obj_ids:
                    uv = kp.get("uv_px")
                    if isinstance(uv, list) and len(uv) == 2:
                        u, v = int(uv[0]), int(uv[1])
                        dr.ellipse((u - 4, v - 4, u + 4, v + 4), fill=(255, 255, 255, 220))

            # Tool keypoint blue.
            tuv = keypoints[tool_id].get("uv_px")
            if isinstance(tuv, list) and len(tuv) == 2:
                tu, tv = int(tuv[0]), int(tuv[1])
                dr.ellipse((tu - 8, tv - 8, tu + 8, tv + 8), fill=(0, 170, 255, 255), outline=(255, 255, 255, 255), width=2)

            # Refs in yellow/cyan.
            ref_cols = [(255, 230, 0, 255), (0, 255, 255, 255)]
            for j, rid in enumerate(refs):
                if rid not in keypoints:
                    continue
                uv = keypoints[rid].get("uv_px")
                if not isinstance(uv, list) or len(uv) != 2:
                    continue
                ru, rv = int(uv[0]), int(uv[1])
                c = ref_cols[j % len(ref_cols)]
                dr.ellipse((ru - 8, rv - 8, ru + 8, rv + 8), fill=c, outline=(255, 255, 255, 255), width=2)

            # Goal red.
            goal = dp.get("goal_tool_keypoint_xyz_world")
            guv = _project_world(goal, cfw, intr, wh) if isinstance(goal, list) and len(goal) == 3 else None
            if guv is not None:
                gu, gv = guv
                dr.ellipse((gu - 8, gv - 8, gu + 8, gv + 8), fill=(255, 0, 0, 255), outline=(255, 255, 255, 255), width=2)
                dr.ellipse((gu - 12, gv - 12, gu + 12, gv + 12), outline=(255, 0, 0, 230), width=2)

            lines = [f"{scene_id} token={token}", f"dp={idx}"]
            x0, y0 = 16, 16
            for line in lines:
                bb = dr.textbbox((x0, y0), line)
                dr.rectangle((bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1), fill=(0, 0, 0, 180))
                dr.text((x0, y0), line, fill=(255, 255, 255, 255))
                y0 += 16

            out_img = out_vis_dir / f"{scene_id}_{token}.png"
            out_json = out_vis_dir / f"{scene_id}_{token}.json"
            frame.save(out_img)
            meta = {
                "scene_id": scene_id,
                "token": token,
                "datapoint_index": idx,
                "instruction": (dp.get("instructions") or [""])[0],
                "tool_keypoint_id": tool_id,
                "ref_keypoint_ids": refs,
                "goal_xyz_world": goal,
                "goal_uv": guv,
                "output_image": str(out_img),
            }
            out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            scene_entries.append({"token": token, "image": str(out_img), "meta": str(out_json)})

        rendered.append({"scene_id": scene_id, "entries": scene_entries})

    summary = {"spotcheck_dir": str(out_vis_dir), "scenes": rendered}
    (out_vis_dir / "spotcheck_index.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dataset-dir", type=Path, default=DEFAULT_OUTPUT_DATASET)
    p.add_argument("--global-camera-json", type=Path, default=DEFAULT_GLOBAL_CAMERA)
    p.add_argument("--spotcheck-dir", type=Path, default=DEFAULT_SPOTCHECK_DIR)
    p.add_argument("--spotcheck-scenes", type=int, default=3)
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    output_dataset_dir = args.output_dataset_dir.resolve()
    output_shards_dir = output_dataset_dir / "position_shards_completed_only"
    output_shards_dir.mkdir(parents=True, exist_ok=True)

    wfc, intr = _load_global_camera(args.global_camera_json.resolve())

    shard_paths = sorted(input_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {input_dir}")

    stats: List[SceneStats] = []
    for shard_path in shard_paths:
        stats.append(_migrate_one_shard(shard_path, output_shards_dir, wfc, intr))

    # Pick 3 representative scenes: first, middle, last.
    spot_n = max(1, int(args.spotcheck_scenes))
    ids = [s.scene_id for s in stats]
    sample_ids: List[str] = []
    if ids:
        sample_ids.append(ids[0])
    if len(ids) > 2:
        sample_ids.append(ids[len(ids) // 2])
    if len(ids) > 1:
        sample_ids.append(ids[-1])
    sample_ids = sample_ids[:spot_n]
    spot = _render_spotchecks(output_shards_dir, args.spotcheck_dir.resolve(), sample_ids)

    summary = {
        "input_dir": str(input_dir),
        "output_dataset_dir": str(output_dataset_dir),
        "global_camera_json": str(args.global_camera_json.resolve()),
        "total_scenes": len(stats),
        "total_keypoints_relifted": int(sum(s.keypoints_relifted for s in stats)),
        "total_goals_updated": int(sum(s.goals_updated for s in stats)),
        "total_missing_depth_points": int(sum(s.missing_depth_points for s in stats)),
        "spotcheck_scenes": sample_ids,
        "spotcheck_dir": str(args.spotcheck_dir.resolve()),
        "per_scene": [s.__dict__ for s in stats],
    }
    (output_dataset_dir / "migration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dataset_dir": str(output_dataset_dir), "total_scenes": len(stats), "spotcheck": spot}, indent=2))


if __name__ == "__main__":
    main()
