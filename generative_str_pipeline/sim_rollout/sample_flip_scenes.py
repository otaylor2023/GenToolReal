"""Sample spatula-flip RL training scenes (keypoints + prompt) for GRPO rollouts.

The flat target starts inside a fixed handleless pan. The spatula stages outside
the +x rim, approaches over the lip, descends into the pan, slides under the
object, then lifts/rotates it into an inverted apex above the pan.

Waypoints are left zero — the VLA predicts them at rollout time.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import _safe_format
from generative_str_pipeline.build_dataset_0012_spatula_flip_reactive import (
    DESTINATIONS_FLIP,
    MATERIAL_SIZE_M,
    MATERIALS_FLIP,
    TEMPLATES_FLIP,
    TOOL_LABEL,
)

TABLE_Z = 0.53

# Flat target box dimensions (length x, width y, height z), matching the dataset.
FLIP_MATERIAL_SIZE = [float(MATERIAL_SIZE_M[0]), float(MATERIAL_SIZE_M[1]), float(MATERIAL_SIZE_M[2])]
_HALF_X = 0.5 * FLIP_MATERIAL_SIZE[0]
_HALF_Y = 0.5 * FLIP_MATERIAL_SIZE[1]
_HALF_XY_DIAG = float(np.linalg.norm([_HALF_X, _HALF_Y]))

# Handleless pan geometry for the flip setup.
PAN_RADIUS_M = 4.5 * 0.0254
PAN_WALL_HEIGHT_M = 2.0 * 0.0254
PAN_WALL_THICKNESS_M = 0.008
PAN_SEGMENTS = 24
PAN_LAYERS = 3
PAN_XYZ = [0.06, 0.0, TABLE_Z]

# Reachable-workspace box for the flat object. The object xy is sampled first,
# then the pan is placed AROUND it (material-relative), so the pan center tracks
# the object instead of sitting at a fixed world point.
MATERIAL_X_RANGE_M = (-0.10, 0.02)
MATERIAL_Y_RANGE_M = (-0.06, 0.06)

# Object band, kept entirely inside the pan but near the rim. The object sits in
# an outer annulus whose radius is capped by the target's XY half-diagonal, so
# the corners stay inside the pan wall instead of placing the object's center
# directly on the edge. This is the object's offset FROM the pan center; the pan
# center is derived as object_xy - offset.
OBJ_EDGE_MARGIN_M = 0.008
OBJ_EDGE_CENTER_R_MAX = PAN_RADIUS_M - _HALF_XY_DIAG - OBJ_EDGE_MARGIN_M
OBJ_EDGE_CENTER_R_MIN = max(0.01, OBJ_EDGE_CENTER_R_MAX - 0.025)
# The arm/tool stages outside the +x rim, so spawn the object near the opposite
# (-x) edge of the pan, farther from the tool start. theta is centered on pi
# (the -x rim) so the spatula has to slide across the pan to reach the object.
OBJ_EDGE_THETA_RANGE = (0.60 * np.pi, 1.40 * np.pi)

# Spatula staging: in front of the object toward +x (the reachable arm side), so
# the whole tool (incl. the long handle that extends +x) stays on the table.
SPATULA_FWD_M_RANGE = (PAN_RADIUS_M + 0.035, PAN_RADIUS_M + 0.065)
SPATULA_Y_JITTER = 0.03

# Deprecated wall aliases kept so older imports fail softly; new sim uses PAN_*.
WALL_SIZE = [PAN_WALL_THICKNESS_M, 2.0 * PAN_RADIUS_M, PAN_WALL_HEIGHT_M]
WALL_XYZ = [PAN_XYZ[0] - PAN_RADIUS_M, PAN_XYZ[1], PAN_XYZ[2]]
WALL_YAW = 0.0

# Apex height above the table for the inverted hold target.
FLIP_APEX_Z_ABOVE_TABLE_M = 0.12


def _unit_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def sample_flip_scene(rng: np.random.Generator, *, table_z: float = TABLE_Z) -> Dict[str, Any]:
    # Sample the object first inside the reachable workspace, then place the pan
    # AROUND it so the object sits near the far (-x) rim from the +x tool entry.
    obj_x = float(rng.uniform(*MATERIAL_X_RANGE_M))
    obj_y = float(rng.uniform(*MATERIAL_Y_RANGE_M))
    theta = float(rng.uniform(*OBJ_EDGE_THETA_RANGE))
    radius = float(rng.uniform(OBJ_EDGE_CENTER_R_MIN, OBJ_EDGE_CENTER_R_MAX))
    offset = radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    pan_center = np.array([obj_x, obj_y], dtype=np.float64) - offset
    material_xyz = np.array([obj_x, obj_y, table_z], dtype=np.float32)

    fwd = float(rng.uniform(*SPATULA_FWD_M_RANGE))
    spat_x = float(pan_center[0]) + fwd  # stage outside +x rim (reachable arm side)
    spat_y = obj_y + float(rng.uniform(-SPATULA_Y_JITTER, SPATULA_Y_JITTER))
    tool_contact = np.array(
        [spat_x, spat_y, table_z + PAN_WALL_HEIGHT_M + 0.035],
        dtype=np.float32,
    )

    motion = _unit_xy(material_xyz - tool_contact)  # ~ -x (toward object/wall)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    surface_dir = (-motion).astype(np.float32)  # front->back, ~ +x (handle toward arm)

    destination_xyz = np.array(
        [obj_x, obj_y, table_z + FLIP_APEX_Z_ABOVE_TABLE_M], dtype=np.float32
    )

    material_word = str(rng.choice(MATERIALS_FLIP))
    destination_word = str(rng.choice(DESTINATIONS_FLIP))
    template = str(rng.choice(TEMPLATES_FLIP))
    instruction = _safe_format(
        template, material=material_word, destination=destination_word, tool=TOOL_LABEL
    )

    return {
        "movement_token": "flip",
        "instruction": instruction,
        "tool_label": TOOL_LABEL,
        "tool_contact_xyz_world": tool_contact.tolist(),
        "tool_current_normal": normal.tolist(),
        "tool_current_surface_dir": surface_dir.tolist(),
        "material_label": material_word,
        "material_xyz_world": material_xyz.tolist(),
        "material_quat_world": [0.0, 0.0, 0.0, 1.0],
        "material_size": list(FLIP_MATERIAL_SIZE),
        "has_material": True,
        "destination_label": destination_word,
        "destination_xyz_world": destination_xyz.tolist(),
        "has_destination": True,
        "pan_xyz_world": [float(pan_center[0]), float(pan_center[1]), float(table_z)],
        "table_label": "table surface center",
        "table_xyz_world": [0.0, 0.0, float(table_z)],
        "table_normal": [0.0, 0.0, 1.0],
        "waypoints": np.zeros((15, 9), dtype=np.float32).tolist(),
    }


def sample_flip_scenes(
    num_scenes: int, *, seed: int = 0, table_z: float = TABLE_Z
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    return [sample_flip_scene(rng, table_z=table_z) for _ in range(int(num_scenes))]
