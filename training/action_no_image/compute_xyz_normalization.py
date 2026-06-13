"""Compute xyz_mean / xyz_std from shard(s): tool, ref, sec ref, goal, one table center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.action_no_image.dataset import (
    NoImageActionSample,
    load_no_image_samples_from_shard,
)


def _xyz_rows_from_samples(samples: list[NoImageActionSample]) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for s in samples:
        rows.append(np.asarray(s.tool_xyz_world, dtype=np.float64).reshape(1, 3))
        rows.append(np.asarray(s.ref_xyz_world, dtype=np.float64).reshape(1, 3))
        if s.secondary_ref_xyz_world is not None:
            rows.append(np.asarray(s.secondary_ref_xyz_world, dtype=np.float64).reshape(1, 3))
        rows.append(np.asarray(s.goal_xyz_world, dtype=np.float64).reshape(1, 3))
    return rows


def collect_xyz_arrays(
    shard_path: Path,
    table_xyz: np.ndarray,
    *,
    explode_instruction_variants: bool,
) -> np.ndarray:
    samples = load_no_image_samples_from_shard(
        shard_path,
        explode_instruction_variants=explode_instruction_variants,
    )
    tw = np.asarray(table_xyz, dtype=np.float64).reshape(1, 3)
    rows = [tw, *_xyz_rows_from_samples(samples)]
    return np.concatenate(rows, axis=0)


def collect_xyz_arrays_multi(
    shard_paths: list[Path],
    table_xyz: np.ndarray,
    *,
    explode_instruction_variants: bool,
) -> np.ndarray:
    tw = np.asarray(table_xyz, dtype=np.float64).reshape(1, 3)
    parts: list[np.ndarray] = [tw]
    for sp in shard_paths:
        samples = load_no_image_samples_from_shard(
            sp,
            explode_instruction_variants=explode_instruction_variants,
        )
        block = _xyz_rows_from_samples(samples)
        if block:
            parts.append(np.concatenate(block, axis=0))
    return np.concatenate(parts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard_path",
        type=str,
        default="",
        help="Path to a single *_shard.json (omit when using --dataset_dir + --max_shards)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Directory containing *_shard.json (use with --max_shards for multi-shard stats)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="If >0, take first N alphabetical shards under --dataset_dir",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--table_xyz",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.53],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--explode_instruction_variants",
        action="store_true",
        help="Match action_expert explode path (dataset_0007 usually needs this false)",
    )
    parser.add_argument("--norm_eps", type=float, default=1e-8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    table_xyz = np.array(args.table_xyz, dtype=np.float64)
    max_shards = int(args.max_shards or 0)

    if max_shards > 0:
        if not str(args.dataset_dir).strip():
            raise SystemExit("--max_shards requires --dataset_dir")
        dd = Path(args.dataset_dir)
        if not dd.is_absolute():
            dd = (repo_root / dd).resolve()
        from training.action_no_image.dataset import list_shard_paths_multi

        shard_paths = list_shard_paths_multi(
            dd,
            max_shards,
            explode_instruction_variants=bool(args.explode_instruction_variants),
        )
        pts = collect_xyz_arrays_multi(
            shard_paths,
            table_xyz,
            explode_instruction_variants=bool(args.explode_instruction_variants),
        )
        scene_ids = [
            str(json.loads(p.read_text(encoding="utf-8")).get("scene_id", ""))
            for p in shard_paths
        ]
        sp = shard_paths[0]
        scene_id = scene_ids[0] if len(scene_ids) == 1 else "multi"
    else:
        sp = Path(args.shard_path)
        if not sp.is_absolute():
            sp = (repo_root / sp).resolve()
        if not sp.is_file():
            raise FileNotFoundError(sp)
        shard_paths = [sp]
        pts = collect_xyz_arrays(
            sp,
            table_xyz,
            explode_instruction_variants=bool(args.explode_instruction_variants),
        )
        scene_id = str(json.loads(sp.read_text(encoding="utf-8")).get("scene_id", ""))
        scene_ids = [scene_id]

    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    payload: dict = {
        "scene_id": str(scene_id),
        "shard_path": str(sp),
        "shard_paths": [str(p) for p in shard_paths],
        "scene_ids": scene_ids,
        "xyz_mean": mean.tolist(),
        "xyz_std": std.tolist(),
        "norm_eps": float(args.norm_eps),
        "num_points": int(pts.shape[0]),
        "table_xyz_world": table_xyz.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} mean={mean} std={std} n={pts.shape[0]}")


if __name__ == "__main__":
    main()
