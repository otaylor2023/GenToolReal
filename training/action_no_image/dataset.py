"""Dataset for image-free action expert (single-shard rows)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from training.action_expert.losses import (
    RegionConstraintConfig,
    deterministic_goal_sample_seed,
    sample_goal_xyz_world_rejection,
)
from training.action_expert.xyz_normalization import normalize_xyz_np


@dataclass
class NoImageActionSample:
    scene_id: str
    shard_path: str
    datapoint_index: int
    goal_sample_group_index: int
    instruction_variant_index: int
    instruction: str
    tool_label: str
    tool_xyz_world: np.ndarray
    ref_label: str
    ref_xyz_world: np.ndarray
    secondary_ref_label: str | None
    secondary_ref_xyz_world: np.ndarray | None
    has_secondary_ref: bool
    goal_xyz_world: np.ndarray
    movement_token: str
    constraint_type: str
    constraint_params: Dict[str, Any]
    reference_xyz_world: np.ndarray | None
    secondary_reference_xyz_world: np.ndarray | None


def _load_shard(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_label(label: str, object_name: str) -> str:
    label_s = str(label).strip()
    obj_s = str(object_name).strip()
    return f"{label_s} of {obj_s}" if obj_s else label_s


def _optional_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        try:
            return int(v.strip())
        except Exception:
            return None
    return None


# Excluded from default single-shard pick and multi-shard discovery (bad / deprecated scenes).
MULTI_SHARD_EXCLUDED_SHARD_FILENAMES: frozenset[str] = frozenset({"scene_00001_shard.json"})


def _shards_sorted_filtered(dataset_dir: Path) -> List[Path]:
    shards = sorted(dataset_dir.glob("*_shard.json"))
    if not shards:
        shards = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    return [p for p in shards if p.name not in MULTI_SHARD_EXCLUDED_SHARD_FILENAMES]


def resolve_shard_path(dataset_dir: Path, shard_path: str) -> Path:
    if str(shard_path).strip():
        p = Path(shard_path)
        if not p.is_absolute():
            repo = Path(__file__).resolve().parent.parent.parent
            p = (repo / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"shard_path not found: {p}")
        return p
    # First alphabetical shard under dataset_dir (exclusions apply).
    shards = _shards_sorted_filtered(dataset_dir)
    if not shards:
        raise FileNotFoundError(f"No shard files in {dataset_dir}")
    return shards[0]


def list_shard_paths_multi(
    dataset_dir: Path,
    max_shards: int,
    *,
    explode_instruction_variants: bool = False,
) -> List[Path]:
    """First `max_shards` shard files that yield at least one row, in sorted path order (exclusions apply)."""
    shards = _shards_sorted_filtered(dataset_dir)
    if not shards:
        raise FileNotFoundError(f"No shard files in {dataset_dir}")
    n = int(max_shards)
    if n <= 0:
        raise ValueError(f"max_shards must be positive, got {max_shards}")
    chosen: List[Path] = []
    scanned = 0
    for p in shards:
        scanned += 1
        samples = load_no_image_samples_from_shard(
            p,
            explode_instruction_variants=explode_instruction_variants,
        )
        if not samples:
            continue
        chosen.append(p)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise FileNotFoundError(
            f"max_shards={n} but only {len(chosen)} non-empty shards after scanning "
            f"{scanned} shard files under {dataset_dir} (after exclusions)"
        )
    return chosen


def load_no_image_samples_from_shards(
    shard_paths: Sequence[Path],
    *,
    explode_instruction_variants: bool = False,
) -> List[NoImageActionSample]:
    out: List[NoImageActionSample] = []
    for p in shard_paths:
        out.extend(
            load_no_image_samples_from_shard(
                p,
                explode_instruction_variants=explode_instruction_variants,
            )
        )
    if not out:
        raise RuntimeError(f"No samples loaded from {len(shard_paths)} shard(s)")
    return out


def load_no_image_samples_from_shard(
    shard_path: Path,
    *,
    explode_instruction_variants: bool = False,
) -> List[NoImageActionSample]:
    p = shard_path
    shard = _load_shard(p)
    keypoints = dict(shard.get("keypoints", {}))
    scene_id = str(shard.get("scene_id", ""))
    out: List[NoImageActionSample] = []

    for dp_idx, dp in enumerate(shard.get("datapoints", [])):
        tool_id = str(dp.get("tool_keypoint_id", ""))
        if tool_id not in keypoints:
            continue
        tool_kp = keypoints[tool_id]
        if not tool_kp.get("valid", False):
            continue
        instructions = list(dp.get("instructions") or [])
        if not instructions:
            continue

        if bool(explode_instruction_variants):
            # dataset_0007: single instruction + instruction_variant_index
            from training.action_expert.dataset import (  # noqa: PLC0415
                _cleaned_instructions,
                _is_valid_expanded_instructions,
            )

            cleaned = _cleaned_instructions(dp)
            meta_iv = dp.get("instruction_variant_index")
            meta_iv_int = _optional_int(meta_iv)
            if len(cleaned) == 1 and meta_iv_int is not None:
                chosen_instructions = [str(cleaned[0])]
            elif _is_valid_expanded_instructions(dp):
                chosen_instructions = cleaned
            else:
                continue
        else:
            chosen_instructions = [str(instructions[0])]

        refs = list(dp.get("ref_keypoint_ids") or [])
        if not refs or refs[0] not in keypoints:
            continue
        rk0 = keypoints[refs[0]]
        if not rk0.get("valid", False) or rk0.get("xyz_world") is None:
            continue
        ref_xyz = np.array(rk0["xyz_world"], dtype=np.float32)
        ref_label = _format_label(str(rk0.get("label", "")), str(rk0.get("object_name", "")))

        sec_label: str | None = None
        sec_xyz: np.ndarray | None = None
        has_sec = False
        if len(refs) > 1 and refs[1] in keypoints:
            rk1 = keypoints[refs[1]]
            if rk1.get("valid", False) and rk1.get("xyz_world") is not None:
                sec_xyz = np.array(rk1["xyz_world"], dtype=np.float32)
                sec_label = _format_label(str(rk1.get("label", "")), str(rk1.get("object_name", "")))
                has_sec = True

        goal = np.array(dp["goal_tool_keypoint_xyz_world"], dtype=np.float32)
        meta_iv = dp.get("instruction_variant_index")
        meta_iv_int = _optional_int(meta_iv)

        for iv, instr in enumerate(chosen_instructions):
            if meta_iv_int is not None and len(chosen_instructions) == 1:
                row_iv = int(meta_iv_int)
            else:
                row_iv = int(iv)
            parent_dp_meta = _optional_int(dp.get("instruction_parent_datapoint_index"))
            goal_group_idx = int(parent_dp_meta) if parent_dp_meta is not None else int(dp_idx)

            out.append(
                NoImageActionSample(
                    scene_id=scene_id,
                    shard_path=str(p),
                    datapoint_index=int(dp_idx),
                    goal_sample_group_index=int(goal_group_idx),
                    instruction_variant_index=int(row_iv),
                    instruction=str(instr),
                    tool_label=_format_label(
                        str(tool_kp.get("label", "")),
                        str(tool_kp.get("object_name", "")),
                    ),
                    tool_xyz_world=np.array(tool_kp["xyz_world"], dtype=np.float32),
                    ref_label=ref_label,
                    ref_xyz_world=ref_xyz,
                    secondary_ref_label=sec_label,
                    secondary_ref_xyz_world=sec_xyz,
                    has_secondary_ref=bool(has_sec),
                    goal_xyz_world=goal,
                    movement_token=str(dp.get("movement_token", "")),
                    constraint_type=str(dp.get("constraint_type", "")),
                    constraint_params=dict(dp.get("constraint_params", {}) or {}),
                    reference_xyz_world=ref_xyz.copy(),
                    secondary_reference_xyz_world=sec_xyz.copy() if sec_xyz is not None else None,
                )
            )

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


class NoImageActionDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        samples: Sequence[NoImageActionSample],
        *,
        xyz_mean: np.ndarray,
        xyz_std: np.ndarray,
        norm_eps: float = 1e-8,
        table_xyz_world: np.ndarray,
        table_label: str,
        region_cfg: RegionConstraintConfig | None = None,
        sample_goal_in_constraint_region: bool = False,
        goal_rejection_sample_max_attempts: int = 512,
    ):
        self.samples = list(samples)
        self._xyz_mean = np.asarray(xyz_mean, dtype=np.float64).reshape(3)
        self._xyz_std = np.asarray(xyz_std, dtype=np.float64).reshape(3)
        self._norm_eps = float(norm_eps)
        self._table_world = np.asarray(table_xyz_world, dtype=np.float64).reshape(3)
        self._table_label = str(table_label)
        self._region_cfg = region_cfg
        self._sample_goal_in_region = bool(sample_goal_in_constraint_region)
        self._goal_rs_max = int(goal_rejection_sample_max_attempts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        dataset_goal_world = np.asarray(s.goal_xyz_world, dtype=np.float64).reshape(3)
        goal_world = dataset_goal_world.copy()
        if self._sample_goal_in_region and self._region_cfg is not None:
            seed = deterministic_goal_sample_seed(
                s.scene_id,
                s.shard_path,
                int(s.goal_sample_group_index),
                int(s.instruction_variant_index),
            )
            rng = np.random.default_rng(seed)
            goal_world, _ok = sample_goal_xyz_world_rejection(
                goal_xyz_world=goal_world,
                movement_token=str(s.movement_token),
                constraint_type=str(s.constraint_type),
                constraint_params=dict(s.constraint_params or {}),
                reference_xyz_world=s.reference_xyz_world,
                secondary_reference_xyz_world=s.secondary_reference_xyz_world,
                has_secondary_reference=s.secondary_reference_xyz_world is not None,
                cfg=self._region_cfg,
                rng=rng,
                max_attempts=int(self._goal_rs_max),
            )
            goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
        else:
            _ok = True

        def _n(x: np.ndarray) -> np.ndarray:
            a = np.asarray(x, dtype=np.float64).reshape(1, 3)
            return normalize_xyz_np(a, self._xyz_mean, self._xyz_std, self._norm_eps)[0].astype(
                np.float32
            )

        tool_n = _n(s.tool_xyz_world)
        ref_n = _n(s.ref_xyz_world)
        sec_n = _n(s.secondary_ref_xyz_world) if s.secondary_ref_xyz_world is not None else np.zeros(3, dtype=np.float32)
        table_n = _n(self._table_world)
        goal_n = _n(goal_world)

        return {
            "instruction_text": s.instruction,
            "tool_label": s.tool_label,
            "ref_label": s.ref_label,
            "secondary_ref_label": s.secondary_ref_label or "",
            "has_secondary_ref": bool(s.has_secondary_ref),
            "table_label": self._table_label,
            "tool_xyz_norm": torch.from_numpy(tool_n),
            "ref_xyz_norm": torch.from_numpy(ref_n),
            "sec_xyz_norm": torch.from_numpy(sec_n),
            "table_xyz_norm": torch.from_numpy(table_n),
            "goal_xyz_norm": torch.from_numpy(goal_n),
            "goal_xyz_world": torch.from_numpy(goal_world.astype(np.float32)),
            "dataset_goal_xyz_world": torch.from_numpy(dataset_goal_world.astype(np.float32)),
            "sampled_goal_in_region": bool(_ok),
            "scene_id": s.scene_id,
            "shard_path": s.shard_path,
            "datapoint_index": int(s.datapoint_index),
            "goal_sample_group_index": int(s.goal_sample_group_index),
            "instruction_variant_index": int(s.instruction_variant_index),
            "movement_token": s.movement_token,
            "constraint_type": s.constraint_type,
            "constraint_params": s.constraint_params,
            "reference_xyz_world": (
                torch.from_numpy(s.reference_xyz_world.copy())
                if s.reference_xyz_world is not None
                else torch.zeros(3, dtype=torch.float32)
            ),
            "secondary_reference_xyz_world": (
                torch.from_numpy(s.secondary_reference_xyz_world.copy())
                if s.secondary_reference_xyz_world is not None
                else torch.zeros(3, dtype=torch.float32)
            ),
            "has_secondary_reference_meta": s.secondary_reference_xyz_world is not None,
        }


def no_image_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    return {
        "instruction_text": [b["instruction_text"] for b in batch],
        "tool_label": [b["tool_label"] for b in batch],
        "ref_label": [b["ref_label"] for b in batch],
        "secondary_ref_label": [b["secondary_ref_label"] for b in batch],
        "has_secondary_ref": default_collate([b["has_secondary_ref"] for b in batch]),
        "table_label": [b["table_label"] for b in batch],
        "tool_xyz_norm": default_collate([b["tool_xyz_norm"] for b in batch]),
        "ref_xyz_norm": default_collate([b["ref_xyz_norm"] for b in batch]),
        "sec_xyz_norm": default_collate([b["sec_xyz_norm"] for b in batch]),
        "table_xyz_norm": default_collate([b["table_xyz_norm"] for b in batch]),
        "goal_xyz_norm": default_collate([b["goal_xyz_norm"] for b in batch]),
        "goal_xyz_world": default_collate([b["goal_xyz_world"] for b in batch]),
        "dataset_goal_xyz_world": default_collate([b["dataset_goal_xyz_world"] for b in batch]),
        "sampled_goal_in_region": [b["sampled_goal_in_region"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
        "shard_path": [b["shard_path"] for b in batch],
        "datapoint_index": [b["datapoint_index"] for b in batch],
        "goal_sample_group_index": [b["goal_sample_group_index"] for b in batch],
        "instruction_variant_index": [b["instruction_variant_index"] for b in batch],
        "movement_token": [b["movement_token"] for b in batch],
        "constraint_type": [b["constraint_type"] for b in batch],
        "constraint_params": [b["constraint_params"] for b in batch],
        "reference_xyz_world": default_collate([b["reference_xyz_world"] for b in batch]),
        "secondary_reference_xyz_world": default_collate([b["secondary_reference_xyz_world"] for b in batch]),
        "has_secondary_reference_meta": [b["has_secondary_reference_meta"] for b in batch],
    }
