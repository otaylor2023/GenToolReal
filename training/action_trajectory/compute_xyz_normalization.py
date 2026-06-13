"""Compute xyz_mean / xyz_std for waypoint trajectory training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.action_trajectory.dataset import (
    WaypointTrajectorySample,
    load_waypoint_samples,
)


def _xyz_rows_from_samples(samples: list[WaypointTrajectorySample]) -> list[np.ndarray]:
    """Collect position rows: tool contact, material/destination refs, waypoint contacts, table."""
    rows: list[np.ndarray] = []
    for s in samples:
        rows.append(np.asarray(s.tool_contact_xyz_world, dtype=np.float64).reshape(1, 3))
        if s.material_xyz_world is not None:
            rows.append(np.asarray(s.material_xyz_world, dtype=np.float64).reshape(1, 3))
        if s.destination_xyz_world is not None:
            rows.append(np.asarray(s.destination_xyz_world, dtype=np.float64).reshape(1, 3))
        wp = np.asarray(s.waypoints, dtype=np.float64).reshape(-1, 9)
        rows.append(wp[:, 0:3])
        rows.append(np.asarray(s.table_xyz_world, dtype=np.float64).reshape(1, 3))
    return rows


def collect_xyz_arrays(
    shard_path: Path,
    table_xyz: np.ndarray,
    *,
    explode_instruction_variants: bool,
) -> np.ndarray:
    _ = table_xyz, explode_instruction_variants
    samples = load_waypoint_samples(shard_path)
    block = _xyz_rows_from_samples(samples)
    if not block:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(block, axis=0)


def collect_xyz_arrays_multi(
    shard_paths: list[Path],
    table_xyz: np.ndarray,
    *,
    explode_instruction_variants: bool,
) -> np.ndarray:
    _ = table_xyz
    parts: list[np.ndarray] = []
    for sp in shard_paths:
        block = collect_xyz_arrays(
            sp,
            table_xyz,
            explode_instruction_variants=explode_instruction_variants,
        )
        if block.shape[0] > 0:
            parts.append(block)
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _assemble_shard_paths_from_args(
    *,
    repo_root: Path,
    dataset_dirs: list[str],
    max_shards_per_dir: list[int],
    dataset_dir: str,
    max_shards: int,
    shard_path: str,
) -> list[Path]:
    if dataset_dirs:
        shard_paths: list[Path] = []
        for i, rel_dir in enumerate(dataset_dirs):
            dd = Path(rel_dir)
            if not dd.is_absolute():
                dd = (repo_root / dd).resolve()
            shards = sorted(dd.glob("*_shard.json"))
            if not shards:
                raise FileNotFoundError(f"No *_shard.json in {dd}")
            cap = int(max_shards_per_dir[i]) if i < len(max_shards_per_dir) else 0
            if cap > 0:
                shards = shards[:cap]
            shard_paths.extend(shards)
        if not shard_paths:
            raise FileNotFoundError("No shard files resolved from --dataset_dirs")
        return shard_paths

    max_shards = int(max_shards or 0)
    if max_shards > 0:
        if not str(dataset_dir).strip():
            raise SystemExit("--max_shards requires --dataset_dir")
        dd = Path(dataset_dir)
        if not dd.is_absolute():
            dd = (repo_root / dd).resolve()
        shard_paths = sorted(dd.glob("*_shard.json"))[:max_shards]
        if not shard_paths:
            raise FileNotFoundError(f"No *_shard.json in {dd}")
        return shard_paths

    sp = Path(shard_path)
    if not sp.is_absolute():
        sp = (repo_root / sp).resolve()
    if not sp.is_file():
        raise FileNotFoundError(sp)
    return [sp]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard_path",
        type=str,
        default="",
        help="Path to a single shard (omit when using --dataset_dir + --max_shards)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="",
        help="Directory containing shard files (use with --max_shards for multi-shard stats)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="If >0, take first N alphabetical shards under --dataset_dir",
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=[],
        help="Multiple shard directories (use with --max_shards_per_dir)",
    )
    parser.add_argument(
        "--max_shards_per_dir",
        type=int,
        nargs="+",
        default=[],
        help="Per-directory shard caps aligned with --dataset_dirs; 0 means all",
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
        help="Match explode path when loading samples",
    )
    parser.add_argument("--norm_eps", type=float, default=1e-8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    table_xyz = np.array(args.table_xyz, dtype=np.float64)
    dataset_dirs = [str(d).strip() for d in args.dataset_dirs if str(d).strip()]
    max_shards_per_dir = [int(x) for x in args.max_shards_per_dir]
    shard_paths = _assemble_shard_paths_from_args(
        repo_root=repo_root,
        dataset_dirs=dataset_dirs,
        max_shards_per_dir=max_shards_per_dir,
        dataset_dir=str(args.dataset_dir),
        max_shards=int(args.max_shards or 0),
        shard_path=str(args.shard_path),
    )
    if len(shard_paths) == 1:
        pts = collect_xyz_arrays(
            shard_paths[0],
            table_xyz,
            explode_instruction_variants=bool(args.explode_instruction_variants),
        )
        meta = json.loads(shard_paths[0].read_text(encoding="utf-8"))
        scene_id = str(meta.get("scene_id") or meta.get("shard_id", ""))
        scene_ids = [scene_id]
    else:
        pts = collect_xyz_arrays_multi(
            shard_paths,
            table_xyz,
            explode_instruction_variants=bool(args.explode_instruction_variants),
        )
        scene_id = "multi"
        scene_ids = [
            str(json.loads(p.read_text(encoding="utf-8")).get("shard_id", "")) for p in shard_paths
        ]

    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()

    payload: dict = {
        "scene_id": str(scene_id),
        "shard_path": str(shard_paths[0]),
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
