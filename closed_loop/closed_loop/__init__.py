"""Standalone brush VLA closed-loop inference for SimToolReal."""

from closed_loop.inference import BrushPolicy
from closed_loop.orientation import (
    align_waypoints_to_reference,
    forward_axis_from_quat,
    is_orientation_flipped,
    orientation_flip,
    quat_geodesic_closeness,
    unflip_waypoints,
)
from closed_loop.policy import ClosedLoopBrushPolicy
from closed_loop.registry import (
    RegisteredModel,
    list_models,
    load_closed_loop_policy,
    load_policy,
    resolve_model,
)
# Single source of truth for the basic per-task default instructions (keyed off
# control_frame/tool). Shared by the viz and the robot runtime. Importing these
# does NOT pull in trimesh (lazily imported only by the mesh builders).
from closed_loop.viz import (
    DEFAULT_INSTRUCTIONS,
    default_instruction_for_control_frame,
    movement_token_for_control_frame,
)

__all__ = [
    "BrushPolicy",
    "ClosedLoopBrushPolicy",
    "forward_axis_from_quat",
    "orientation_flip",
    "is_orientation_flipped",
    "unflip_waypoints",
    "align_waypoints_to_reference",
    "quat_geodesic_closeness",
    "RegisteredModel",
    "list_models",
    "load_policy",
    "load_closed_loop_policy",
    "resolve_model",
    "DEFAULT_INSTRUCTIONS",
    "default_instruction_for_control_frame",
    "movement_token_for_control_frame",
]
