"""Closed-loop receding-horizon brush policy."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from closed_loop.frames import contact_frame_direct, contact_frame_from_root, robot_to_model_xyz
from closed_loop.inference import BrushPolicy
from closed_loop.orientation import align_waypoints_to_reference
from closed_loop.scene import SceneState
from closed_loop.viz import default_instruction_for_control_frame

GOAL_REGION_RADIUS_M = 0.05
OBJECT_RADIUS_M = 0.02


class ClosedLoopBrushPolicy:
    """Receding-horizon VLA: predict chunk, observe, replan until material reaches goal."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        control_frame: str = "blue_brush",
        instruction: str | None = None,
        frame_shift: tuple[float, float, float] = (0.0, 0.8, 0.0),
        chunk_size: int = 5,
        max_replans: int = 30,
        goal_region_radius_m: float = GOAL_REGION_RADIUS_M,
        tool_pose_is_root: bool = True,
        unflip_orientation: bool = True,
        **brush_policy_kwargs,
    ):
        self.frame_shift = np.asarray(frame_shift, dtype=np.float64)
        self.chunk_size = max(1, int(chunk_size))
        self.max_replans = int(max_replans)
        self.goal_region_radius_m = float(goal_region_radius_m)
        self.tool_pose_is_root = bool(tool_pose_is_root)
        self.unflip_orientation = bool(unflip_orientation)
        resolved_instruction = (
            str(instruction)
            if instruction is not None
            else default_instruction_for_control_frame(control_frame)
        )
        self.instruction = resolved_instruction

        self._brush = BrushPolicy(
            device=device,
            control_frame=control_frame,
            instruction=resolved_instruction,
            **brush_policy_kwargs,
        )
        self.T_oc = self._brush.T_oc
        self.table_z = self._brush.table_z

        self._destination_model: np.ndarray | None = None
        self._material_model: np.ndarray | None = None
        self._contact: np.ndarray | None = None
        self._normal: np.ndarray | None = None
        self._surface_dir: np.ndarray | None = None
        self._current_chunk: List[Tuple[np.ndarray, np.ndarray]] = []
        self._chunk_index = 0
        self._replans = 0
        self._done = False
        self._last_plan_waypoints: np.ndarray | None = None
        self._last_plan_object_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self._ref_object_quat: np.ndarray | None = None
        self._last_replan_unflipped = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_replan_unflipped(self) -> bool:
        """True if the most recent replan un-flipped the predicted orientation."""
        return self._last_replan_unflipped

    @property
    def last_plan_waypoints(self) -> np.ndarray | None:
        """Raw [15, 9] model-frame waypoints from the most recent replan."""
        return self._last_plan_waypoints

    @property
    def last_plan_object_poses(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Full (untruncated) list of (xyz_robot, quat_xyzw) from the last replan."""
        return list(self._last_plan_object_poses)

    def set_control_frame(self, name_or_path: str):
        """Switch tool control frame (reload ``T_obj_from_contact``)."""
        path = self._brush.set_control_frame(name_or_path)
        self.T_oc = self._brush.T_oc
        return path

    def set_destination(self, destination_xyz_robot: np.ndarray) -> None:
        dest = np.asarray(destination_xyz_robot, dtype=np.float64).reshape(3)
        self._destination_model = robot_to_model_xyz(dest, self.frame_shift).astype(np.float32)

    def _tool_to_contact(
        self, tool_xyz_robot: np.ndarray, tool_quat_xyzw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tool_xyz_model = robot_to_model_xyz(tool_xyz_robot, self.frame_shift).astype(np.float64)
        if self.tool_pose_is_root:
            return contact_frame_from_root(tool_xyz_model, tool_quat_xyzw, self.T_oc)
        return contact_frame_direct(tool_xyz_model, tool_quat_xyzw)

    def reset(
        self,
        tool_root_xyz_robot: np.ndarray,
        tool_root_quat_xyzw: np.ndarray,
        material_xyz_robot: np.ndarray,
    ) -> None:
        if self._destination_model is None:
            raise RuntimeError("Call set_destination() before reset()")
        contact, normal, sd = self._tool_to_contact(tool_root_xyz_robot, tool_root_quat_xyzw)
        self._contact = contact
        self._normal = normal
        self._surface_dir = sd
        self._material_model = robot_to_model_xyz(material_xyz_robot, self.frame_shift).astype(
            np.float32
        )
        self._chunk_index = 0
        self._replans = 0
        self._done = False
        self._ref_object_quat = None
        self._replan()

    def _make_scene(self) -> SceneState:
        assert self._contact is not None
        assert self._material_model is not None
        assert self._destination_model is not None
        return SceneState(
            instruction=self.instruction,
            tool_label=self._brush.tool_label,
            tool_contact_xyz_world=self._contact.copy(),
            tool_current_normal=self._normal.copy(),
            tool_current_surface_dir=self._surface_dir.copy(),
            material_xyz_world=self._material_model.copy(),
            destination_xyz_world=self._destination_model.copy(),
            table_xyz_world=np.array([0.0, 0.0, self.table_z], dtype=np.float32),
            table_z=self.table_z,
        )

    def _replan(self) -> None:
        if self._done or self._replans >= self.max_replans:
            self._done = True
            return
        waypoints = self._brush.predict_waypoints(self._make_scene())
        self._last_replan_unflipped = False
        if self.unflip_orientation and self._ref_object_quat is not None:
            waypoints, info = align_waypoints_to_reference(
                waypoints, self._ref_object_quat, self.T_oc
            )
            self._last_replan_unflipped = bool(info["applied"])
        full_object_poses = self._brush.waypoints_to_object_poses_robot(
            waypoints, self.frame_shift
        )
        if self.unflip_orientation and full_object_poses:
            self._ref_object_quat = np.asarray(
                full_object_poses[0][1], dtype=np.float64
            ).copy()
        self._last_plan_waypoints = waypoints
        self._last_plan_object_poses = full_object_poses
        self._current_chunk = full_object_poses[: self.chunk_size]
        self._chunk_index = 0
        self._replans += 1

    def plan_chunk(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return current chunk as list of (xyz, quat_xyzw) in robot frame."""
        if not self._current_chunk:
            self._replan()
        return list(self._current_chunk)

    def current_goal_pose(self) -> Tuple[np.ndarray, np.ndarray] | None:
        if not self._current_chunk:
            return None
        idx = min(self._chunk_index, len(self._current_chunk) - 1)
        return self._current_chunk[idx]

    def advance_goal_index(self) -> bool:
        """Advance within current chunk. Returns True if chunk finished."""
        if not self._current_chunk:
            return True
        if self._chunk_index < len(self._current_chunk) - 1:
            self._chunk_index += 1
            return False
        return True

    def observe(
        self,
        tool_root_xyz_robot: np.ndarray,
        tool_root_quat_xyzw: np.ndarray,
        material_xyz_robot: np.ndarray,
    ) -> None:
        self._material_model = robot_to_model_xyz(material_xyz_robot, self.frame_shift).astype(
            np.float32
        )
        if self._destination_model is not None:
            dist = float(
                np.linalg.norm(self._material_model[:2] - self._destination_model[:2])
            )
            if dist <= self.goal_region_radius_m:
                self._done = True
                return
        contact, normal, sd = self._tool_to_contact(tool_root_xyz_robot, tool_root_quat_xyzw)
        self._contact = contact
        self._normal = normal
        self._surface_dir = sd
        self._replan()

    def material_at_goal(self, material_xyz_robot: np.ndarray) -> bool:
        if self._destination_model is None:
            return False
        mat = robot_to_model_xyz(material_xyz_robot, self.frame_shift)
        return (
            float(np.linalg.norm(mat[:2] - self._destination_model[:2]))
            <= self.goal_region_radius_m
        )
