"""Dataset and batching utilities for action-expert training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from training.action_expert.losses import (
    RegionConstraintConfig,
    deterministic_goal_sample_seed,
    sample_goal_xyz_world_rejection,
)
from training.action_expert.xyz_normalization import normalize_xyz_np


@dataclass
class ActionSample:
    scene_id: str
    shard_path: str
    datapoint_index: int
    goal_sample_group_index: int
    instruction_variant_index: int
    image_path: Path
    instruction: str
    keypoints: List[Dict[str, Any]]
    tool_keypoint_id: str
    tool_keypoint_label: str
    goal_xyz_world: np.ndarray
    movement_token: str
    constraint_type: str
    constraint_params: Dict[str, Any]
    reference_xyz_world: np.ndarray | None
    secondary_reference_xyz_world: np.ndarray | None


def _load_shard(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shard_paths(dataset_dir: Path) -> List[Path]:
    p1 = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    p2 = sorted(dataset_dir.glob("*_shard.json"))
    paths = p1 if p1 else p2
    if not paths:
        raise FileNotFoundError(f"No shard files found in {dataset_dir}")
    return paths


@lru_cache(maxsize=8192)
def _load_rgb_resized_cached(image_path: str, width: int, height: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return np.array(img, dtype=np.uint8)


def split_shards_85_10_5(
    dataset_dir: Path,
    *,
    seed: int,
    train_fraction: float = 0.85,
    val_fraction: float = 0.10,
) -> Dict[str, List[Path]]:
    shard_paths = _shard_paths(dataset_dir)
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(shard_paths))
    rng.shuffle(idxs)
    shuffled = [shard_paths[int(i)] for i in idxs]

    n = len(shuffled)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test != n:
            n_train = max(1, n - n_val - n_test)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    if not val and test:
        val = [test.pop(0)]
    if not test and val:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def _ordered_keypoints(
    *,
    keypoints: Dict[str, Dict[str, Any]],
    tool_keypoint_id: str,
    max_keypoints: int,
) -> List[Dict[str, Any]]:
    if tool_keypoint_id not in keypoints:
        return []
    tool_kp = keypoints[tool_keypoint_id]
    tool_obj = str(tool_kp.get("object_name", "")).strip()
    tool_label = str(tool_kp.get("label", "")).strip()

    valid_items: List[Dict[str, Any]] = []
    for kp_id, kp in keypoints.items():
        xyz = kp.get("xyz_world")
        if (not kp.get("valid", False)) or (not isinstance(xyz, (list, tuple))) or len(xyz) != 3:
            continue
        lbl = str(kp.get("label", "")).strip()
        obj = str(kp.get("object_name", "")).strip()
        if not lbl:
            continue
        valid_items.append(
            {
                "id": str(kp_id),
                "label": lbl,
                "object_name": obj,
                "position_xyz_world": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            }
        )

    # Explicit grouping order from spec.
    active: List[Dict[str, Any]] = []
    tool_rest: List[Dict[str, Any]] = []
    by_other_obj: Dict[str, List[Dict[str, Any]]] = {}
    for item in valid_items:
        if item["id"] == str(tool_keypoint_id):
            active.append(item)
            continue
        if item["object_name"] == tool_obj:
            tool_rest.append(item)
            continue
        by_other_obj.setdefault(item["object_name"], []).append(item)

    tool_rest.sort(key=lambda x: x["label"].lower())
    for k in list(by_other_obj.keys()):
        by_other_obj[k].sort(key=lambda x: x["label"].lower())
    ordered_other_obj_names = sorted(by_other_obj.keys(), key=lambda x: x.lower())

    ordered: List[Dict[str, Any]] = active
    ordered.extend(tool_rest)
    for obj_name in ordered_other_obj_names:
        ordered.extend(by_other_obj[obj_name])
    if max_keypoints > 0:
        ordered = ordered[: max_keypoints]
    # Guarantee active tool keypoint first.
    if ordered and ordered[0]["id"] != str(tool_keypoint_id):
        ordered.insert(
            0,
            {
                "id": str(tool_keypoint_id),
                "label": tool_label,
                "object_name": tool_obj,
                "position_xyz_world": [float(v) for v in tool_kp.get("xyz_world", [0.0, 0.0, 0.0])],
            },
        )
        if max_keypoints > 0:
            ordered = ordered[: max_keypoints]
    return ordered


def _format_label(label: str, object_name: str) -> str:
    label_s = str(label).strip()
    obj_s = str(object_name).strip()
    return f"{label_s} of {obj_s}" if obj_s else label_s


def _cleaned_instructions(dp: Dict[str, Any]) -> List[str]:
    instr = dp.get("instructions")
    if not isinstance(instr, list):
        return []
    return [str(x).strip() for x in instr if str(x).strip()]


def _is_valid_expanded_instructions(dp: Dict[str, Any]) -> bool:
    cleaned = _cleaned_instructions(dp)
    if len(cleaned) != 4:
        return False
    if len({c.lower() for c in cleaned}) != 4:
        return False
    st = str(dp.get("instruction_status", "")).strip().lower()
    if st and st != "ok":
        return False
    return True


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


def load_action_samples_from_shards(
    shard_paths: Sequence[Path],
    *,
    max_keypoints: int,
    region_cfg: RegionConstraintConfig,
    explode_instruction_variants: bool = False,
) -> List[ActionSample]:
    out: List[ActionSample] = []
    for p in shard_paths:
        shard = _load_shard(p)
        keypoints = dict(shard.get("keypoints", {}))
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
                cleaned = _cleaned_instructions(dp)
                meta_iv = dp.get("instruction_variant_index")
                meta_iv_int = _optional_int(meta_iv)
                # Pre-exploded shards (e.g. dataset_0007): one instruction row + metadata index.
                if len(cleaned) == 1 and meta_iv_int is not None:
                    chosen_instructions = [str(cleaned[0])]
                elif _is_valid_expanded_instructions(dp):
                    chosen_instructions = cleaned
                else:
                    continue
            else:
                chosen_instructions = [str(instructions[0])]
            refs = list(dp.get("ref_keypoint_ids") or [])
            ref = None
            if refs and refs[0] in keypoints and keypoints[refs[0]].get("xyz_world") is not None:
                ref = np.array(keypoints[refs[0]]["xyz_world"], dtype=np.float32)
            ref2 = None
            if len(refs) > 1 and refs[1] in keypoints and keypoints[refs[1]].get("xyz_world") is not None:
                ref2 = np.array(keypoints[refs[1]]["xyz_world"], dtype=np.float32)
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
                    ActionSample(
                        scene_id=str(shard.get("scene_id", "")),
                        shard_path=str(p),
                        datapoint_index=int(dp_idx),
                        goal_sample_group_index=int(goal_group_idx),
                        instruction_variant_index=int(row_iv),
                        image_path=Path(shard["image"]),
                        instruction=str(instr),
                        keypoints=_ordered_keypoints(
                            keypoints=keypoints,
                            tool_keypoint_id=tool_id,
                            max_keypoints=int(max_keypoints),
                        ),
                        tool_keypoint_id=tool_id,
                        tool_keypoint_label=_format_label(
                            str(tool_kp.get("label", "")),
                            str(tool_kp.get("object_name", "")),
                        ),
                        goal_xyz_world=goal,
                        movement_token=str(dp.get("movement_token", "")),
                        constraint_type=str(dp.get("constraint_type", "")),
                        constraint_params=dict(dp.get("constraint_params", {}) or {}),
                        reference_xyz_world=ref,
                        secondary_reference_xyz_world=ref2,
                    )
                )
    if not out:
        raise RuntimeError("No action samples loaded from selected shards")
    return out


class ActionExpertDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        samples: Sequence[ActionSample],
        *,
        image_size: Sequence[int] = (224, 224),
        xyz_mean: np.ndarray,
        xyz_std: np.ndarray,
        norm_eps: float = 1e-8,
        region_cfg: RegionConstraintConfig | None = None,
        sample_goal_in_constraint_region: bool = False,
        goal_rejection_sample_max_attempts: int = 512,
    ):
        self.samples = list(samples)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._xyz_mean = np.asarray(xyz_mean, dtype=np.float64).reshape(3)
        self._xyz_std = np.asarray(xyz_std, dtype=np.float64).reshape(3)
        self._norm_eps = float(norm_eps)
        self._region_cfg = region_cfg
        self._sample_goal_in_constraint_region = bool(sample_goal_in_constraint_region)
        self._goal_rejection_sample_max_attempts = int(goal_rejection_sample_max_attempts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = _load_rgb_resized_cached(
            str(s.image_path), self.image_size[1], self.image_size[0]
        )
        keypoint_labels: List[str] = []
        keypoint_positions: List[List[float]] = []
        object_labels: List[str] = []
        for kp in s.keypoints:
            keypoint_labels.append(_format_label(kp["label"], kp["object_name"]))
            kp_arr = np.asarray(kp["position_xyz_world"], dtype=np.float64).reshape(1, 3)
            kp_n = normalize_xyz_np(kp_arr, self._xyz_mean, self._xyz_std, self._norm_eps)[0]
            keypoint_positions.append(kp_n.astype(np.float32).tolist())
            if kp["object_name"]:
                object_labels.append(str(kp["object_name"]))
        object_labels = sorted(set(object_labels))
        dataset_goal_world = np.asarray(s.goal_xyz_world, dtype=np.float64).reshape(3)
        goal_world = dataset_goal_world.copy()
        if (
            self._sample_goal_in_constraint_region
            and self._region_cfg is not None
        ):
            seed = deterministic_goal_sample_seed(
                s.scene_id,
                s.shard_path,
                int(s.goal_sample_group_index),
                0,
            )
            rng = np.random.default_rng(seed)
            goal_world, _sampled_ok = sample_goal_xyz_world_rejection(
                goal_xyz_world=goal_world,
                movement_token=str(s.movement_token),
                constraint_type=str(s.constraint_type),
                constraint_params=dict(s.constraint_params or {}),
                reference_xyz_world=s.reference_xyz_world,
                secondary_reference_xyz_world=s.secondary_reference_xyz_world,
                has_secondary_reference=s.secondary_reference_xyz_world is not None,
                cfg=self._region_cfg,
                rng=rng,
                max_attempts=int(self._goal_rejection_sample_max_attempts),
            )
            goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
        else:
            _sampled_ok = True

        goal_arr = goal_world.reshape(1, 3)
        goal_n = normalize_xyz_np(goal_arr, self._xyz_mean, self._xyz_std, self._norm_eps)[0]
        return {
            "image": torch.from_numpy(image.copy()),
            "scene_id": s.scene_id,
            "shard_path": s.shard_path,
            "datapoint_index": int(s.datapoint_index),
            "goal_sample_group_index": int(s.goal_sample_group_index),
            "instruction_variant_index": int(s.instruction_variant_index),
            "instruction_text": s.instruction,
            "system_prompt": "",  # set in trainer from config to keep dataset portable.
            "keypoint_labels": keypoint_labels,
            "keypoint_positions": torch.tensor(keypoint_positions, dtype=torch.float32),
            "object_labels": object_labels,
            "goal_xyz_norm": torch.from_numpy(goal_n.astype(np.float32)),
            "goal_xyz_world": torch.from_numpy(goal_world.astype(np.float32)),
            "dataset_goal_xyz_world": torch.from_numpy(dataset_goal_world.astype(np.float32)),
            "sampled_goal_in_region": bool(_sampled_ok),
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
            "has_secondary_reference": s.secondary_reference_xyz_world is not None,
        }


def action_expert_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": default_collate([b["image"] for b in batch]),
        "scene_id": [b["scene_id"] for b in batch],
        "shard_path": [b["shard_path"] for b in batch],
        "datapoint_index": [b["datapoint_index"] for b in batch],
        "goal_sample_group_index": [b["goal_sample_group_index"] for b in batch],
        "instruction_variant_index": [b["instruction_variant_index"] for b in batch],
        "instruction_text": [b["instruction_text"] for b in batch],
        "system_prompt": [b["system_prompt"] for b in batch],
        "keypoint_labels": [b["keypoint_labels"] for b in batch],
        "keypoint_positions": [b["keypoint_positions"] for b in batch],
        "object_labels": [b["object_labels"] for b in batch],
        "goal_xyz_norm": default_collate([b["goal_xyz_norm"] for b in batch]),
        "goal_xyz_world": default_collate([b["goal_xyz_world"] for b in batch]),
        "dataset_goal_xyz_world": default_collate([b["dataset_goal_xyz_world"] for b in batch]),
        "sampled_goal_in_region": [b["sampled_goal_in_region"] for b in batch],
        "movement_token": [b["movement_token"] for b in batch],
        "constraint_type": [b["constraint_type"] for b in batch],
        "constraint_params": [b["constraint_params"] for b in batch],
        "reference_xyz_world": default_collate([b["reference_xyz_world"] for b in batch]),
        "secondary_reference_xyz_world": default_collate(
            [b["secondary_reference_xyz_world"] for b in batch]
        ),
        "has_secondary_reference": [b["has_secondary_reference"] for b in batch],
    }

