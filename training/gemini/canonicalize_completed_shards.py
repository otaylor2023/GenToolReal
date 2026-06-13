"""Canonicalize completed-only position shards for PPO.

This utility rewrites shard coordinates from world frame into a per-scene
canonical frame by translating XYZ values with:

    p_canonical = p_world - look_at_xyz_m

Input shards are expected to already be completed-only (instructions present).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT_DIR = Path(
    "/home/ubuntu/Generative_STR/training/datasets/dataset_0003_5552_s50_from_existing/position_shards_completed_only"
)
DEFAULT_RUNS_DIR = Path("/home/ubuntu/Generative_STR/isaaclab_simtoolreal/runs_0036")
DEFAULT_OUTPUT_DIR = Path(
    "/home/ubuntu/Generative_STR/training/datasets/dataset_0005_canonical_tablez0_s50_completed_only/position_shards_completed_only"
)
DEFAULT_SUMMARY_PATH = Path(
    "/home/ubuntu/Generative_STR/training/datasets/dataset_0005_canonical_tablez0_s50_completed_only/canonicalization_summary.json"
)
DEFAULT_TABLE_Z_WORLD = 0.9645966375863392


@dataclass
class SceneResult:
    scene_id: str
    status: str
    error: str | None
    keypoints_transformed: int
    goals_transformed: int
    input_datapoints: int
    output_path: str | None


def _parse_vec3(value: Any, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a length-3 list")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} values must be numeric") from exc


def _shift_vec3(value: Any, offset_xyz: Tuple[float, float, float]) -> List[float]:
    x, y, z = _parse_vec3(value, "xyz_world")
    ox, oy, oz = offset_xyz
    return [x - ox, y - oy, z - oz]


def _canonicalize_one_shard(
    shard_path: Path,
    *,
    runs_dir: Path,
    output_dir: Path,
    fail_on_missing_camera: bool,
    z_origin: str,
    table_z_world: float | None,
) -> SceneResult:
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    scene_id = str(shard.get("scene_id", "")).strip()
    if not scene_id:
        return SceneResult(
            scene_id="unknown",
            status="error",
            error=f"Missing scene_id in shard {shard_path}",
            keypoints_transformed=0,
            goals_transformed=0,
            input_datapoints=0,
            output_path=None,
        )

    camera_path = runs_dir / scene_id / "camera.json"
    if not camera_path.exists():
        message = f"Missing camera file: {camera_path}"
        if fail_on_missing_camera:
            return SceneResult(
                scene_id=scene_id,
                status="error",
                error=message,
                keypoints_transformed=0,
                goals_transformed=0,
                input_datapoints=len(shard.get("datapoints", [])),
                output_path=None,
            )
        return SceneResult(
            scene_id=scene_id,
            status="skipped",
            error=message,
            keypoints_transformed=0,
            goals_transformed=0,
            input_datapoints=len(shard.get("datapoints", [])),
            output_path=None,
        )

    try:
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        look_at = _parse_vec3(camera.get("look_at_xyz_m"), "look_at_xyz_m")
    except Exception as exc:  # noqa: BLE001
        return SceneResult(
            scene_id=scene_id,
            status="error",
            error=f"Invalid camera data in {camera_path}: {exc}",
            keypoints_transformed=0,
            goals_transformed=0,
            input_datapoints=len(shard.get("datapoints", [])),
            output_path=None,
        )

    keypoints_transformed = 0
    z_shift = 0.0
    if z_origin == "table":
        if table_z_world is None:
            raise ValueError("table_z_world is required when z_origin='table'")
        z_shift = float(table_z_world) - float(look_at[2])

    for kp in (shard.get("keypoints") or {}).values():
        xyz = kp.get("xyz_world")
        if xyz is None:
            continue
        shifted = _shift_vec3(xyz, look_at)
        shifted[2] = shifted[2] - z_shift
        kp["xyz_world"] = shifted
        keypoints_transformed += 1

    goals_transformed = 0
    for dp in shard.get("datapoints", []):
        goal_xyz = dp.get("goal_tool_keypoint_xyz_world")
        if goal_xyz is None:
            continue
        shifted = _shift_vec3(goal_xyz, look_at)
        shifted[2] = shifted[2] - z_shift
        dp["goal_tool_keypoint_xyz_world"] = shifted
        goals_transformed += 1

    # Keep camera info explicit for downstream debugging/audits.
    shard.setdefault("canonicalization", {})
    shard["canonicalization"].update(
        {
            "type": "look_at_translation_with_optional_table_z_rebase",
            "source_camera_path": str(camera_path),
            "look_at_xyz_m": list(look_at),
            "formula": (
                "p = p_world - look_at_xyz_m; if z_origin=table: p.z = p.z - (table_z_world - look_at_z)"
            ),
            "axis_convention": {
                "x": "right_relative_to_camera",
                "y": "away_from_camera",
                "z": "up",
            },
            "z_origin": "tabletop_z0" if z_origin == "table" else "look_at_z0",
            "table_z_world": float(table_z_world) if table_z_world is not None else None,
            "z_shift_applied_m": float(z_shift),
        }
    )

    out_path = output_dir / shard_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
    return SceneResult(
        scene_id=scene_id,
        status="ok",
        error=None,
        keypoints_transformed=keypoints_transformed,
        goals_transformed=goals_transformed,
        input_datapoints=len(shard.get("datapoints", [])),
        output_path=str(out_path),
    )


def run_canonicalization(
    *,
    input_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    summary_path: Path,
    max_workers: int,
    fail_on_missing_camera: bool,
    z_origin: str,
    table_z_world: float | None,
) -> Dict[str, Any]:
    shard_paths = sorted(input_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {input_dir}")

    results: List[SceneResult] = []
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _canonicalize_one_shard,
                shard_path,
                runs_dir=runs_dir,
                output_dir=output_dir,
                fail_on_missing_camera=fail_on_missing_camera,
                z_origin=z_origin,
                table_z_world=table_z_world,
            ): shard_path
            for shard_path in shard_paths
        }
        for fut in as_completed(future_map):
            results.append(fut.result())

    results.sort(key=lambda r: r.scene_id)
    summary = {
        "input_dir": str(input_dir),
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "max_workers": workers,
        "formula": (
            "p = p_world - look_at_xyz_m; if z_origin=table: p.z = p.z - (table_z_world - look_at_z)"
        ),
        "axis_convention": {
            "x": "right_relative_to_camera",
            "y": "away_from_camera",
            "z": "up",
        },
        "z_origin": z_origin,
        "table_z_world": float(table_z_world) if table_z_world is not None else None,
        "total_shards": len(shard_paths),
        "processed_ok": sum(1 for r in results if r.status == "ok"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "errors": sum(1 for r in results if r.status == "error"),
        "total_input_datapoints": sum(r.input_datapoints for r in results),
        "total_keypoints_transformed": sum(r.keypoints_transformed for r in results),
        "total_goals_transformed": sum(r.goals_transformed for r in results),
        "scene_results": [
            {
                "scene_id": r.scene_id,
                "status": r.status,
                "error": r.error,
                "input_datapoints": r.input_datapoints,
                "keypoints_transformed": r.keypoints_transformed,
                "goals_transformed": r.goals_transformed,
                "output_path": r.output_path,
            }
            for r in results
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonicalize completed-only shard coordinates by look_at translation."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--z-origin",
        type=str,
        choices=["look_at", "table"],
        default="table",
        help="Set z=0 at look_at point or at a fixed tabletop world z.",
    )
    parser.add_argument(
        "--table-z-world",
        type=float,
        default=DEFAULT_TABLE_Z_WORLD,
        help="Fixed tabletop height in world frame; used when --z-origin table.",
    )
    parser.add_argument(
        "--fail-on-missing-camera",
        action="store_true",
        help="If set, missing camera metadata marks scenes as errors instead of skipped.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_canonicalization(
        input_dir=args.input_dir,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        max_workers=args.max_workers,
        fail_on_missing_camera=bool(args.fail_on_missing_camera),
        z_origin=str(args.z_origin),
        table_z_world=float(args.table_z_world) if args.table_z_world is not None else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

