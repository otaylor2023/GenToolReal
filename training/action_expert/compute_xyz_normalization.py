"""Offline: compute global per-axis mean/std over all XYZ in position shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from training.action_expert.dataset import _load_shard, _shard_paths


def collect_all_xyz(dataset_dir: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for shard_path in _shard_paths(dataset_dir):
        shard: Dict[str, Any] = _load_shard(shard_path)
        keypoints = dict(shard.get("keypoints", {}))
        for v in keypoints.values():
            if not v.get("valid", False):
                continue
            xyz = v.get("xyz_world")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                continue
            rows.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        for dp in shard.get("datapoints", []):
            g = dp.get("goal_tool_keypoint_xyz_world")
            if isinstance(g, (list, tuple)) and len(g) == 3:
                rows.append([float(g[0]), float(g[1]), float(g[2])])
            tool_id = str(dp.get("tool_keypoint_id", ""))
            if tool_id in keypoints:
                tk = keypoints[tool_id]
                if tk.get("valid", False):
                    xyz = tk.get("xyz_world")
                    if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
                        rows.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    if not rows:
        raise RuntimeError(f"No XYZ collected from {dataset_dir}")
    return np.asarray(rows, dtype=np.float64)


def collect_keypoint_counts(dataset_dir: Path) -> Dict[str, Any]:
    per_shard: List[Dict[str, Any]] = []
    counts: List[int] = []
    for shard_path in _shard_paths(dataset_dir):
        shard: Dict[str, Any] = _load_shard(shard_path)
        keypoints = shard.get("keypoints", {})
        if isinstance(keypoints, dict):
            kp_count = len(keypoints)
        elif isinstance(keypoints, list):
            kp_count = len(keypoints)
        else:
            kp_count = 0
        counts.append(kp_count)
        per_shard.append(
            {
                "scene_id": str(shard.get("scene_id", "")),
                "shard_path": str(shard_path),
                "keypoint_count": int(kp_count),
                "datapoint_count": int(len(shard.get("datapoints", []))),
            }
        )
    if not counts:
        raise RuntimeError(f"No shards found in {dataset_dir}")
    arr = np.asarray(counts, dtype=np.int64)
    unique, freq = np.unique(arr, return_counts=True)
    histogram = {str(int(k)): int(v) for k, v in zip(unique, freq)}
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "num_shards": int(arr.shape[0]),
        "keypoint_count_min": int(arr.min()),
        "keypoint_count_max": int(arr.max()),
        "keypoint_count_mean": float(arr.mean()),
        "keypoint_count_median": float(np.median(arr)),
        "histogram": histogram,
        "per_shard": per_shard,
    }


def write_keypoint_histogram_png(histogram: Dict[str, int], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xs = sorted(int(k) for k in histogram.keys())
    ys = [int(histogram[str(x)]) for x in xs]
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(xs, ys, width=0.8, color="#4f8dd8")
    ax.set_title("Keypoints Per Shard")
    ax.set_xlabel("Keypoint count")
    ax.set_ylabel("Number of shards")
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute xyz_mean / xyz_std for action-expert training.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory containing *_position_dataset_v0_1.json shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/cfg/normalization_stats.json"),
        help="Where to write normalization_stats.json",
    )
    parser.add_argument(
        "--keypoint_stats_output",
        type=Path,
        default=Path("training/cfg/keypoint_count_stats.json"),
        help="Where to write keypoint-count stats JSON",
    )
    parser.add_argument(
        "--keypoint_hist_output",
        type=Path,
        default=Path("training/cfg/keypoint_count_histogram.png"),
        help="Where to write keypoint-count histogram image",
    )
    args = parser.parse_args()
    all_xyz = collect_all_xyz(args.dataset_dir)
    xyz_mean = all_xyz.mean(axis=0).astype(np.float64)
    xyz_std = all_xyz.std(axis=0).astype(np.float64)
    payload = {
        "xyz_mean": xyz_mean.tolist(),
        "xyz_std": xyz_std.tolist(),
        "norm_eps": 1e-8,
        "num_points": int(all_xyz.shape[0]),
        "dataset_dir": str(args.dataset_dir.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    keypoint_stats = collect_keypoint_counts(args.dataset_dir)
    args.keypoint_stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.keypoint_stats_output.write_text(json.dumps(keypoint_stats, indent=2), encoding="utf-8")
    wrote_hist = write_keypoint_histogram_png(keypoint_stats["histogram"], args.keypoint_hist_output)
    print(f"Wrote {args.output} num_points={payload['num_points']} mean={payload['xyz_mean']} std={payload['xyz_std']}")
    print(
        "Wrote "
        f"{args.keypoint_stats_output} num_shards={keypoint_stats['num_shards']} "
        f"kp_min={keypoint_stats['keypoint_count_min']} kp_max={keypoint_stats['keypoint_count_max']} "
        f"kp_mean={keypoint_stats['keypoint_count_mean']:.2f}"
    )
    if wrote_hist:
        print(f"Wrote {args.keypoint_hist_output}")
    else:
        print("Skipped keypoint histogram image (matplotlib unavailable).")


if __name__ == "__main__":
    main()
