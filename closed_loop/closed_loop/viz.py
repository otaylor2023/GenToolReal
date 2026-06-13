"""Scene visualization helpers for closed-loop viz (pan geometry, task gating).

Task-identity constants and the per-task default instructions live here so that
both the viz and the robot runtime share a single source of truth. The mesh
builders import ``trimesh`` lazily (it is an optional ``[viz]`` dependency), so
importing the instruction/gating helpers stays dependency-light for the robot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import trimesh

# Handleless pan geometry (matches generative_str_pipeline flip/pour sim).
TABLE_Z_DEFAULT = 0.53
PAN_RADIUS_M = 4.5 * 0.0254
PAN_WALL_HEIGHT_M = 2.0 * 0.0254
PAN_WALL_THICKNESS_M = 0.008
PAN_XYZ = (0.06, 0.0)  # flip legacy fallback center (xy)
POUR_PAN_XYZ = (-0.08, 0.0)  # pour legacy fallback center (xy)

COLOR_PAN_WALL = (60, 60, 66)
COLOR_PAN_RIM = (20, 20, 24)

# Map packaged control-frame stems to training movement tokens.
_CONTROL_FRAME_TO_MOVEMENT_TOKEN: dict[str, str] = {
    "flat_spatula": "flip",
    "spoon_spatula": "pour",
    "blue_brush": "sweep",
    "red_brush": "sweep",
    "claw_hammer": "press",
    "mallet_hammer": "press",
}

# Basic, plain per-task default prompts (NOT the long templated dataset prompts).
# Single source of truth shared by the viz and the robot runtime, keyed off the
# tool/control_frame the same way the pan gating is.
DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "blue_brush": "Sweep the material",
    "red_brush": "Sweep the material",
    "flat_spatula": "Flip the cube with the spatula",
    "spoon_spatula": "Scoop and pour the material",
    "claw_hammer": "Hammer the nail",
    "mallet_hammer": "Hammer the nail",
}

_FALLBACK_INSTRUCTION = "Sweep the material"


def movement_token_for_control_frame(control_frame: str) -> str:
    """Infer training movement token from a packaged control-frame name."""
    stem = str(control_frame).strip()
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return _CONTROL_FRAME_TO_MOVEMENT_TOKEN.get(stem, "sweep")


def default_instruction_for_control_frame(control_frame: str) -> str:
    """Basic, plain default prompt for a tool/control_frame (single source)."""
    stem = str(control_frame).strip()
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return DEFAULT_INSTRUCTIONS.get(stem, _FALLBACK_INSTRUCTION)


def pan_viz_settings(movement_token: str) -> tuple[bool, tuple[float, float]]:
    """Return (draw_pan, pan_center_xy) for a task movement token.

    Fixed-center fallback used only when no material-relative center is available.
    """
    token = str(movement_token).strip().lower()
    if token == "pour":
        return True, (float(POUR_PAN_XYZ[0]), float(POUR_PAN_XYZ[1]))
    if token == "flip":
        return True, (float(PAN_XYZ[0]), float(PAN_XYZ[1]))
    return False, (float(PAN_XYZ[0]), float(PAN_XYZ[1]))


def pan_center_for_scene(
    *,
    control_frame: str,
    material_xyz_world: np.ndarray,
    pan_center_xy_world: np.ndarray | None = None,
) -> tuple[bool, tuple[float, float]]:
    """Return (draw_pan, pan_center_xy) material-relative for a closed-loop scene.

    Pan is drawn only for flip/pour (spatula scoop / spoon pour). Center priority:
    explicit ``pan_center_xy_world`` on the scene, else material xy, else legacy fixed.
    """
    token = movement_token_for_control_frame(control_frame)
    draw_pan = token in ("flip", "pour")
    if pan_center_xy_world is not None:
        c = np.asarray(pan_center_xy_world, dtype=np.float64).reshape(-1)
        return draw_pan, (float(c[0]), float(c[1]))
    mat = np.asarray(material_xyz_world, dtype=np.float64).reshape(-1)
    if mat.size >= 2:
        return draw_pan, (float(mat[0]), float(mat[1]))
    _, fixed = pan_viz_settings(token)
    return draw_pan, fixed


def _make_pan_wall(
    table_z: float,
    center_xy: tuple[float, float] = PAN_XYZ,
) -> "trimesh.Trimesh":
    """Handleless pan wall ring sitting on the table (open top, no bottom)."""
    import trimesh

    wall = trimesh.creation.annulus(
        r_min=float(PAN_RADIUS_M),
        r_max=float(PAN_RADIUS_M) + float(PAN_WALL_THICKNESS_M),
        height=float(PAN_WALL_HEIGHT_M),
        sections=64,
    )
    wall.apply_translation(
        [
            float(center_xy[0]),
            float(center_xy[1]),
            float(table_z) + 0.5 * float(PAN_WALL_HEIGHT_M),
        ]
    )
    wall.visual.face_colors = [(*COLOR_PAN_WALL, 130)] * len(wall.faces)
    return wall


def _make_pan_rim(
    table_z: float,
    center_xy: tuple[float, float] = PAN_XYZ,
) -> "trimesh.Trimesh":
    """Bright circle marking the pan rim radius at the top of the wall."""
    import trimesh

    rim = trimesh.creation.torus(
        major_radius=float(PAN_RADIUS_M) + 0.5 * float(PAN_WALL_THICKNESS_M),
        minor_radius=0.004,
        major_sections=64,
        minor_sections=12,
    )
    rim.apply_translation(
        [
            float(center_xy[0]),
            float(center_xy[1]),
            float(table_z) + float(PAN_WALL_HEIGHT_M),
        ]
    )
    rim.visual.face_colors = [(*COLOR_PAN_RIM, 255)] * len(rim.faces)
    return rim


def make_pan_meshes(
    table_z: float,
    center_xy: tuple[float, float],
) -> Tuple["trimesh.Trimesh", "trimesh.Trimesh"]:
    """Build wall + rim trimesh pair for Viser/pyrender overlays."""
    return _make_pan_wall(table_z, center_xy), _make_pan_rim(table_z, center_xy)


def pan_center_for_sample_like(sample: Any) -> tuple[bool, tuple[float, float]]:
    """Sample/scene dict helper mirroring main-pipeline ``pan_center_for_sample``."""
    token = str(getattr(sample, "movement_token", "")).strip().lower()
    if not token:
        cf = getattr(sample, "control_frame", None) or getattr(sample, "tool_label", "")
        token = movement_token_for_control_frame(str(cf))
    draw_pan = token in ("flip", "pour")
    center = getattr(sample, "pan_center_xy_world", None)
    if center is not None:
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        return draw_pan, (float(c[0]), float(c[1]))
    mat = getattr(sample, "material_xyz_world", None)
    if mat is not None:
        m = np.asarray(mat, dtype=np.float64).reshape(-1)
        return draw_pan, (float(m[0]), float(m[1]))
    _, fixed = pan_viz_settings(token)
    return draw_pan, fixed
