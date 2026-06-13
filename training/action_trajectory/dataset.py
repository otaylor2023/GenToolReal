"""Dataset for waypoint-trajectory action expert."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from training.action_expert.xyz_normalization import normalize_xyz_np


@dataclass
class WaypointTrajectorySample:
    scene_id: str
    shard_path: str
    datapoint_index: int
    movement_token: str
    instruction: str
    tool_label: str
    tool_contact_xyz_world: np.ndarray
    tool_current_normal: np.ndarray
    tool_current_surface_dir: np.ndarray
    material_label: str | None
    material_xyz_world: np.ndarray | None
    material_normal: np.ndarray | None
    has_material: bool
    destination_label: str | None
    destination_xyz_world: np.ndarray | None
    destination_normal: np.ndarray | None
    has_destination: bool
    table_label: str
    table_xyz_world: np.ndarray
    table_normal: np.ndarray
    waypoints: np.ndarray  # [N, 9]: contact_xyz, normal, surface_dir per waypoint
    # Per-scene material-relative pan center (xy world). Only present for pan
    # tasks (flip/pour); None otherwise. Used by visualization to draw the pan
    # around the sampled material instead of at a fixed world point.
    pan_center_xy_world: np.ndarray | None = None


def _arr_or_none(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).reshape(3)


def load_waypoint_samples(
    shard_path: Path,
    *,
    explode_instruction_variants: bool = False,
) -> List[WaypointTrajectorySample]:
    """Load waypoint trajectory samples from a procedural brush shard file."""
    _ = explode_instruction_variants
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    scene_id_alias = str(shard.get("scene_id") or shard.get("shard_id") or "")

    out: List[WaypointTrajectorySample] = []
    for dp in shard.get("datapoints", []):
        out.append(
            WaypointTrajectorySample(
                scene_id=scene_id_alias,
                shard_path=str(shard_path),
                datapoint_index=int(dp["datapoint_index"]),
                movement_token=str(dp.get("movement_token", "")),
                instruction=str(dp["instruction"]),
                tool_label=str(dp["tool_label"]),
                tool_contact_xyz_world=np.array(dp["tool_contact_xyz_world"], dtype=np.float32),
                tool_current_normal=np.array(dp["tool_current_normal"], dtype=np.float32),
                tool_current_surface_dir=np.array(
                    dp["tool_current_surface_dir"], dtype=np.float32
                ),
                material_label=(str(dp["material_label"]) if dp.get("material_label") else None),
                material_xyz_world=_arr_or_none(dp.get("material_xyz_world")),
                material_normal=_arr_or_none(dp.get("material_normal")),
                has_material=bool(
                    dp.get("has_material", dp.get("material_xyz_world") is not None)
                ),
                destination_label=(
                    str(dp["destination_label"]) if dp.get("destination_label") else None
                ),
                destination_xyz_world=_arr_or_none(dp.get("destination_xyz_world")),
                destination_normal=_arr_or_none(dp.get("destination_normal")),
                has_destination=bool(
                    dp.get("has_destination", dp.get("destination_xyz_world") is not None)
                ),
                table_label=str(dp.get("table_label", "table surface center")),
                table_xyz_world=np.array(
                    dp.get("table_xyz_world", [0.0, 0.0, 0.53]), dtype=np.float32
                ),
                table_normal=np.array(dp.get("table_normal", [0.0, 0.0, 1.0]), dtype=np.float32),
                waypoints=np.array(dp["waypoints"], dtype=np.float32).reshape(-1, 9),
                pan_center_xy_world=(
                    np.asarray(dp["pan_center_xy_world"], dtype=np.float32).reshape(2)
                    if dp.get("pan_center_xy_world") is not None
                    else None
                ),
            )
        )
    return out


def load_waypoint_samples_from_shards(
    shard_paths: Sequence[Path],
    *,
    explode_instruction_variants: bool = False,
) -> List[WaypointTrajectorySample]:
    out: List[WaypointTrajectorySample] = []
    for p in shard_paths:
        out.extend(
            load_waypoint_samples(
                p,
                explode_instruction_variants=explode_instruction_variants,
            )
        )
    if not out:
        raise RuntimeError(f"No samples loaded from {len(shard_paths)} shard(s)")
    return out


def split_sample_indices(
    n: int,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(int(seed))
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test != n:
            n_train = max(1, n - n_val - n_test)
    train = idxs[:n_train].tolist()
    val = idxs[n_train : n_train + n_val].tolist()
    test = idxs[n_train + n_val :].tolist()
    if not val and test:
        val = [test.pop(0)]
    if not test and val:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


class WaypointTrajectoryDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        samples: Sequence[WaypointTrajectorySample],
        *,
        xyz_mean: np.ndarray,
        xyz_std: np.ndarray,
        norm_eps: float = 1e-8,
    ):
        self.samples = list(samples)
        self._xyz_mean = np.asarray(xyz_mean, dtype=np.float64).reshape(3)
        self._xyz_std = np.asarray(xyz_std, dtype=np.float64).reshape(3)
        self._norm_eps = float(norm_eps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        def _n_pos(x: np.ndarray) -> np.ndarray:
            a = np.asarray(x, dtype=np.float64).reshape(1, 3)
            return normalize_xyz_np(a, self._xyz_mean, self._xyz_std, self._norm_eps)[0].astype(
                np.float32
            )

        tool_contact_n = _n_pos(s.tool_contact_xyz_world)
        material_n = (
            _n_pos(s.material_xyz_world)
            if s.material_xyz_world is not None
            else np.zeros(3, dtype=np.float32)
        )
        destination_n = (
            _n_pos(s.destination_xyz_world)
            if s.destination_xyz_world is not None
            else np.zeros(3, dtype=np.float32)
        )
        table_n = _n_pos(s.table_xyz_world)

        waypoints = np.asarray(s.waypoints, dtype=np.float32).reshape(-1, 9)
        contact_norm = normalize_xyz_np(
            waypoints[:, 0:3].astype(np.float64),
            self._xyz_mean,
            self._xyz_std,
            self._norm_eps,
        ).astype(np.float32)
        waypoints_norm = waypoints.copy()
        waypoints_norm[:, 0:3] = contact_norm
        waypoints_flat = waypoints_norm.reshape(-1)

        return {
            "instruction_text": s.instruction,
            "tool_label": s.tool_label,
            "material_label": s.material_label or "",
            "destination_label": s.destination_label or "",
            "has_material": bool(s.has_material),
            "has_destination": bool(s.has_destination),
            "table_label": s.table_label,
            "tool_contact_xyz_norm": torch.from_numpy(tool_contact_n),
            "tool_normal": torch.from_numpy(
                np.asarray(s.tool_current_normal, dtype=np.float32).reshape(3)
            ),
            "tool_surface_dir": torch.from_numpy(
                np.asarray(s.tool_current_surface_dir, dtype=np.float32).reshape(3)
            ),
            "material_xyz_norm": torch.from_numpy(material_n),
            "destination_xyz_norm": torch.from_numpy(destination_n),
            "table_xyz_norm": torch.from_numpy(table_n),
            "waypoints_norm": torch.from_numpy(waypoints_flat),
            "waypoints_world": torch.from_numpy(waypoints.astype(np.float32).reshape(-1)),
            "scene_id": s.scene_id,
            "shard_path": s.shard_path,
            "datapoint_index": int(s.datapoint_index),
            "movement_token": s.movement_token,
        }


def waypoint_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    return {
        "instruction_text": [b["instruction_text"] for b in batch],
        "tool_label": [b["tool_label"] for b in batch],
        "material_label": [b["material_label"] for b in batch],
        "destination_label": [b["destination_label"] for b in batch],
        "has_material": default_collate([b["has_material"] for b in batch]),
        "has_destination": default_collate([b["has_destination"] for b in batch]),
        "table_label": [b["table_label"] for b in batch],
        "tool_contact_xyz_norm": default_collate([b["tool_contact_xyz_norm"] for b in batch]),
        "tool_normal": default_collate([b["tool_normal"] for b in batch]),
        "tool_surface_dir": default_collate([b["tool_surface_dir"] for b in batch]),
        "material_xyz_norm": default_collate([b["material_xyz_norm"] for b in batch]),
        "destination_xyz_norm": default_collate([b["destination_xyz_norm"] for b in batch]),
        "table_xyz_norm": default_collate([b["table_xyz_norm"] for b in batch]),
        "waypoints_norm": default_collate([b["waypoints_norm"] for b in batch]),
        "waypoints_world": default_collate([b["waypoints_world"] for b in batch]),
        "scene_id": [b["scene_id"] for b in batch],
        "shard_path": [b["shard_path"] for b in batch],
        "datapoint_index": [b["datapoint_index"] for b in batch],
        "movement_token": [b["movement_token"] for b in batch],
    }
