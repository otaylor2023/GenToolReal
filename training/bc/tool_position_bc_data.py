"""Dataset loading utilities for tool-position behavior cloning."""

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


@dataclass
class EpisodeSample:
    scene_id: str
    image_path: Path
    tool_keypoint_xyz_world: np.ndarray
    goal_xyz_world: np.ndarray
    instruction: str
    task_prompt: str
    world_scale_prompt: str
    tool_keypoint_label: str
    movement_token: str
    constraint_type: str
    constraint_params: Dict[str, Any]
    reference_xyz_world: np.ndarray | None
    secondary_reference_xyz_world: np.ndarray | None
    all_keypoints_label_position: List[Dict[str, Any]]


def _load_shard(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8192)
def _load_rgb_resized_cached(image_path: str, width: int, height: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return np.array(img, dtype=np.uint8)


def _format_text_context(
    *,
    task_prompt: str,
    world_scale_prompt: str,
    instruction: str,
    tool_keypoint_label: str,
    tool_keypoint_xyz_world: Sequence[float],
    all_keypoints_label_position: Sequence[Dict[str, Any]],
) -> str:
    tool_xyz = [
        float(tool_keypoint_xyz_world[0]),
        float(tool_keypoint_xyz_world[1]),
        float(tool_keypoint_xyz_world[2]),
    ]
    tool_name = tool_keypoint_label.split(" of ", 1)[1] if " of " in tool_keypoint_label else "unknown"
    tool_kp_name = tool_keypoint_label.split(" of ", 1)[0]
    kp_items = []
    for kp in all_keypoints_label_position:
        lbl = str(kp.get("label", "")).strip()
        obj = str(kp.get("object_name", "")).strip()
        pos = kp.get("position_xyz_world")
        if not lbl or not isinstance(pos, (list, tuple)) or len(pos) != 3:
            continue
        full_label = f"{lbl} of {obj}" if obj else lbl
        kp_items.append(
            (
                obj.lower(),
                lbl.lower(),
                f"- {full_label}: [{float(pos[0]):.5f}, {float(pos[1]):.5f}, {float(pos[2]):.5f}]",
            )
        )
    kp_items.sort(key=lambda x: (x[0], x[1]))
    kp_lines = [row[2] for row in kp_items]
    all_kp_block = "\n".join(kp_lines)
    return (
        f"Task: {task_prompt}\n"
        f"WorldScale: {world_scale_prompt}\n\n"
        f"AllKeypoints:\n{all_kp_block}\n\n"
        f"Tool:\n"
        f"- object: {tool_name}\n"
        f"- keypoint: {tool_kp_name}\n"
        f"- position_xyz: [{tool_xyz[0]:.5f}, {tool_xyz[1]:.5f}, {tool_xyz[2]:.5f}]\n\n"
        f"Instruction:\n{instruction}"
    )


def split_shards_85_10_5(
    dataset_dir: Path,
    *,
    seed: int,
    train_fraction: float = 0.85,
    val_fraction: float = 0.10,
) -> Dict[str, List[Path]]:
    shard_paths = sorted(dataset_dir.glob("*_position_dataset_v0_1.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No dataset shards found in {dataset_dir}")

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(shard_paths))
    rng.shuffle(idxs)
    shuffled = [shard_paths[int(i)] for i in idxs]

    n = len(shuffled)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val

    # Guardrails for tiny datasets.
    if n >= 3:
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
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
    if not train and (val or test):
        if val:
            train = [val.pop(0)]
        else:
            train = [test.pop(0)]
    return {"train": train, "val": val, "test": test}


def load_episode_samples_from_shards(
    shard_paths: Sequence[Path],
    *,
    task_prompt: str,
    world_scale_prompt: str,
) -> List[EpisodeSample]:
    samples: List[EpisodeSample] = []
    for p in shard_paths:
        shard = _load_shard(p)
        keypoints = shard["keypoints"]
        for dp in shard["datapoints"]:
            tool_id = dp["tool_keypoint_id"]
            if tool_id not in keypoints:
                continue
            tool_kp = keypoints[tool_id]
            if not tool_kp.get("valid", False):
                continue
            instructions = dp.get("instructions") or []
            if not instructions:
                continue
            refs = list(dp.get("ref_keypoint_ids") or [])

            primary_ref_xyz = None
            if refs and refs[0] in keypoints and keypoints[refs[0]].get("xyz_world") is not None:
                primary_ref_xyz = np.array(keypoints[refs[0]]["xyz_world"], dtype=np.float32)
            secondary_ref_xyz = None
            if len(refs) > 1 and refs[1] in keypoints and keypoints[refs[1]].get("xyz_world") is not None:
                secondary_ref_xyz = np.array(keypoints[refs[1]]["xyz_world"], dtype=np.float32)

            samples.append(
                EpisodeSample(
                    scene_id=shard["scene_id"],
                    image_path=Path(shard["image"]),
                    tool_keypoint_xyz_world=np.array(tool_kp["xyz_world"], dtype=np.float32),
                    goal_xyz_world=np.array(dp["goal_tool_keypoint_xyz_world"], dtype=np.float32),
                    instruction=str(instructions[0]),
                    task_prompt=str(task_prompt),
                    world_scale_prompt=str(world_scale_prompt),
                    tool_keypoint_label=str(tool_kp["label"]),
                    movement_token=str(dp.get("movement_token", "")),
                    constraint_type=str(dp.get("constraint_type", "point_goal_v0")),
                    constraint_params=dict(dp.get("constraint_params", {}) or {}),
                    reference_xyz_world=primary_ref_xyz,
                    secondary_reference_xyz_world=secondary_ref_xyz,
                    all_keypoints_label_position=[
                        {
                            "label": str(v.get("label", "")),
                            "object_name": str(v.get("object_name", "")).strip(),
                            "position_xyz_world": v.get("xyz_world"),
                        }
                        for v in keypoints.values()
                        if v.get("valid", False) and v.get("xyz_world") is not None
                    ],
                )
            )
    if not samples:
        raise RuntimeError("No valid episode samples were loaded from selected shards")
    return samples


class ToolPositionBCDataset(Dataset[Dict[str, Any]]):
    """Simple supervised dataset for XYZ behavior cloning."""

    def __init__(self, samples: Sequence[EpisodeSample], *, image_size: Sequence[int] = (224, 224)):
        self.samples = list(samples)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image = _load_rgb_resized_cached(
            str(s.image_path),
            self.image_size[1],
            self.image_size[0],
        )
        text_context = _format_text_context(
            task_prompt=s.task_prompt,
            world_scale_prompt=s.world_scale_prompt,
            instruction=s.instruction,
            tool_keypoint_label=s.tool_keypoint_label,
            tool_keypoint_xyz_world=s.tool_keypoint_xyz_world,
            all_keypoints_label_position=s.all_keypoints_label_position,
        )
        return {
            "image": torch.from_numpy(image.copy()),
            "text_context": text_context,
            "goal_xyz_world": torch.from_numpy(s.goal_xyz_world.copy()),
            "movement_token": s.movement_token,
            "constraint_type": s.constraint_type,
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
            "constraint_params": s.constraint_params,
        }

