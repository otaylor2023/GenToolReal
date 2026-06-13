"""Tool-position PPO environment backed by dataset shards.

This env is intentionally separate from SimToolReal control-policy training.
It optimizes absolute world-frame XYZ actions for a selected tool keypoint.
"""

from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("tool_position_ppo_env requires gymnasium") from exc


@dataclass
class EpisodeSample:
    scene_id: str
    image_path: Path
    depth_path: Path
    width: int
    height: int
    intrinsics: Dict[str, float]
    keypoints: Dict[str, Dict[str, Any]]
    tool_keypoint_id: str
    tool_keypoint_label: str
    tool_keypoint_xyz_world: np.ndarray
    relation_string: str
    instruction: str
    task_prompt: str
    world_scale_prompt: str
    goal_xyz_world: np.ndarray
    initial_tool_xyz_world: np.ndarray
    movement_token: str
    constraint_type: str
    constraint_params: Dict[str, Any]
    reference_xyz_world: np.ndarray | None
    all_keypoints_label_position: List[Dict[str, Any]]


def _load_shard(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8192)
def _load_rgb_resized_cached(image_path: str, width: int, height: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return np.array(img, dtype=np.uint8)


def load_episode_samples(
    dataset_dir: Path,
    *,
    max_keypoints: int,
    split: str = "train",
    train_fraction: float = 0.9,
    seed: int = 7,
    task_prompt: str = (
        "You are a robot tool-positioning policy. Your goal is to output the absolute "
        "3D target position (x, y, z) in meters where the tool should move, based on "
        "the instruction. Movement is defined using the specified tool keypoint. You "
        "are given the scene image and a set of scene keypoints with coordinates to "
        "help you understand where objects are located. Return only the target position."
    ),
    world_scale_prompt: str = (
        "Coordinate system: canonical tabletop frame in meters. The tabletop surface is z = 0. "
        "Axis directions are fixed: +X points to the right relative to the camera, +Y points away "
        "from the camera, and +Z points upward. All scene keypoint coordinates and the tool "
        "keypoint position are provided in this same frame."
    ),
) -> List[EpisodeSample]:
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

    samples: List[EpisodeSample] = []
    for i, p in enumerate(shard_paths):
        if i not in keep:
            continue
        shard = _load_shard(p)
        keypoints = shard["keypoints"]
        width = int(shard["camera"]["intrinsics"]["width"])
        height = int(shard["camera"]["intrinsics"]["height"])
        intr = shard["camera"]["intrinsics"]
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
            samples.append(
                EpisodeSample(
                    scene_id=shard["scene_id"],
                    image_path=Path(shard["image"]),
                    depth_path=Path(shard["depth"]),
                    width=width,
                    height=height,
                    intrinsics=intr,
                    keypoints=keypoints,
                    tool_keypoint_id=tool_id,
                    tool_keypoint_label=str(tool_kp["label"]),
                    tool_keypoint_xyz_world=np.array(tool_kp["xyz_world"], dtype=np.float32),
                    relation_string=str(dp["relation_string"]),
                    instruction=str(instructions[0]),
                    task_prompt=str(task_prompt),
                    world_scale_prompt=str(world_scale_prompt),
                    goal_xyz_world=np.array(dp["goal_tool_keypoint_xyz_world"], dtype=np.float32),
                    initial_tool_xyz_world=np.array(tool_kp["xyz_world"], dtype=np.float32),
                    movement_token=str(dp.get("movement_token", "")),
                    constraint_type=str(dp.get("constraint_type", "point_goal_v0")),
                    constraint_params=dict(dp.get("constraint_params", {}) or {}),
                    reference_xyz_world=(
                        np.array(
                            keypoints[dp["ref_keypoint_ids"][0]]["xyz_world"], dtype=np.float32
                        )
                        if dp.get("ref_keypoint_ids")
                        and len(dp["ref_keypoint_ids"]) > 0
                        and dp["ref_keypoint_ids"][0] in keypoints
                        and keypoints[dp["ref_keypoint_ids"][0]].get("valid", False)
                        and keypoints[dp["ref_keypoint_ids"][0]].get("xyz_world") is not None
                        else None
                    ),
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
        raise RuntimeError("No valid episode samples were loaded from dataset shards")
    return samples


class ToolPositionPPOEnv(gym.Env):
    """Single-env episodic task for absolute world XYZ tool placement."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        samples: List[EpisodeSample],
        *,
        max_steps: int = 32,
        workspace_min: Sequence[float] = (-1.5, -1.5, -0.2),
        workspace_max: Sequence[float] = (1.5, 1.5, 1.8),
        success_threshold_m: float = 0.02,
        invalid_penalty: float = -1.0,
        success_bonus: float = 2.0,
        smoothness_coef: float = 0.01,
        token_success_bonus: float = 1.0,
        exact_above_xy_margin_m: float = 0.01,
        exact_above_z_margin_m: float = 0.01,
        exact_above_w_xy: float = 1.5,
        exact_above_w_z: float = 1.0,
        image_size: Sequence[int] = (224, 224),
    ):
        super().__init__()
        self.samples = samples
        self.max_steps = int(max_steps)
        self.workspace_min = np.array(workspace_min, dtype=np.float32)
        self.workspace_max = np.array(workspace_max, dtype=np.float32)
        self.success_threshold_m = float(success_threshold_m)
        self.invalid_penalty = float(invalid_penalty)
        self.success_bonus = float(success_bonus)
        self.smoothness_coef = float(smoothness_coef)
        self.token_success_bonus = float(token_success_bonus)
        self.exact_above_xy_margin_m = float(exact_above_xy_margin_m)
        self.exact_above_z_margin_m = float(exact_above_z_margin_m)
        self.exact_above_w_xy = float(exact_above_w_xy)
        self.exact_above_w_z = float(exact_above_w_z)
        self.image_size = (int(image_size[0]), int(image_size[1]))

        self.action_space = spaces.Box(
            low=self.workspace_min,
            high=self.workspace_max,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.image_size[0], self.image_size[1], 3),
                    dtype=np.uint8,
                ),
                "current_tool_xyz_world": spaces.Box(
                    low=-10.0, high=10.0, shape=(3,), dtype=np.float32
                ),
                "tool_keypoint_xyz_world": spaces.Box(
                    low=-10.0, high=10.0, shape=(3,), dtype=np.float32
                ),
            }
        )
        self._rng = np.random.default_rng(0)
        self._step_count = 0
        self._sample: EpisodeSample | None = None
        self._current_xyz = np.zeros(3, dtype=np.float32)
        self._prev_action = np.zeros(3, dtype=np.float32)

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _build_observation(self) -> Dict[str, Any]:
        assert self._sample is not None
        # Use plain RGB observation (no runtime overlay rendering).
        img = _load_rgb_resized_cached(
            str(self._sample.image_path),
            self.image_size[1],
            self.image_size[0],
        )
        return {
            "image": img,
            "current_tool_xyz_world": self._current_xyz.astype(np.float32),
            "tool_keypoint_xyz_world": self._sample.tool_keypoint_xyz_world.astype(np.float32),
        }

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        idx = int(self._rng.integers(0, len(self.samples)))
        self._sample = self.samples[idx]
        self._step_count = 0
        self._current_xyz = self._sample.initial_tool_xyz_world.copy()
        self._prev_action = self._current_xyz.copy()
        obs = self._build_observation()
        info = {
            "scene_id": self._sample.scene_id,
            "tool_keypoint_label": self._sample.tool_keypoint_label,
            "tool_keypoint_xyz_world": self._sample.tool_keypoint_xyz_world.tolist(),
            "relation_string": self._sample.relation_string,
            "instruction": self._sample.instruction,
            "task_prompt": self._sample.task_prompt,
            "world_scale_prompt": self._sample.world_scale_prompt,
            "all_keypoints_label_position": self._sample.all_keypoints_label_position,
        }
        return obs, info

    def step(self, action: np.ndarray):
        assert self._sample is not None
        self._step_count += 1

        clipped = np.clip(np.asarray(action, dtype=np.float32), self.workspace_min, self.workspace_max)
        out_of_bounds = not np.allclose(clipped, np.asarray(action, dtype=np.float32))
        self._current_xyz = clipped

        pos_err = float(np.linalg.norm(self._current_xyz - self._sample.goal_xyz_world))
        reward = -pos_err
        xy_penalty = 0.0
        z_penalty = 0.0
        token_success = False
        token_reward = 0.0

        if (
            self._sample.constraint_type == "exact_above_cylinder"
            and self._sample.reference_xyz_world is not None
        ):
            params = self._sample.constraint_params
            xy_radius = float(params.get("xy_radius_m", 0.02))
            z_min = float(params.get("z_min_offset_m", 0.03))
            z_max = float(params.get("z_max_offset_m", 0.08))
            ref = self._sample.reference_xyz_world.astype(np.float32)

            dx = float(self._current_xyz[0] - ref[0])
            dy = float(self._current_xyz[1] - ref[1])
            dz = float(self._current_xyz[2] - ref[2])
            xy_dist = float(np.sqrt(dx * dx + dy * dy))

            xy_overflow = max(0.0, xy_dist - xy_radius)
            z_under = max(0.0, z_min - dz)
            z_over = max(0.0, dz - z_max)
            z_overflow = z_under + z_over

            # Hinge-style penalties with a denominator margin for stable gradients.
            xy_penalty = xy_overflow / max(self.exact_above_xy_margin_m, 1e-6)
            z_penalty = z_overflow / max(self.exact_above_z_margin_m, 1e-6)
            token_reward -= self.exact_above_w_xy * xy_penalty
            token_reward -= self.exact_above_w_z * z_penalty
            token_success = (xy_overflow <= 1e-6) and (z_overflow <= 1e-6)
            if token_success:
                token_reward += self.token_success_bonus

            reward += token_reward
        if out_of_bounds:
            reward += self.invalid_penalty
        reward -= self.smoothness_coef * float(np.linalg.norm(clipped - self._prev_action))
        self._prev_action = clipped.copy()

        success = pos_err <= self.success_threshold_m
        if success:
            reward += self.success_bonus

        terminated = success
        truncated = self._step_count >= self.max_steps
        obs = self._build_observation()
        info = {
            "position_error_m": pos_err,
            "success": success,
            "tool_keypoint_label": self._sample.tool_keypoint_label,
            "tool_keypoint_xyz_world": self._sample.tool_keypoint_xyz_world.tolist(),
            "all_keypoints_label_position": self._sample.all_keypoints_label_position,
            "instruction": self._sample.instruction,
            "task_prompt": self._sample.task_prompt,
            "world_scale_prompt": self._sample.world_scale_prompt,
            "movement_token": self._sample.movement_token,
            "constraint_type": self._sample.constraint_type,
            "xy_penalty": float(xy_penalty),
            "z_penalty": float(z_penalty),
            "token_success": bool(token_success),
            "token_reward": float(token_reward),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

