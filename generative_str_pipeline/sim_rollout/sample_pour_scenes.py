"""Sample spoon scoop-and-pour RL training scenes for GRPO rollouts.

The material starts inside a fixed handleless pan (the same pan used by the
spatula flip). The spoon stages outside the +x rim, reaches over the lip,
descends to scoop the material from the pan, carries it over a goal plate that
sits on the table OUTSIDE the pan (to the +x, reachable side), and rolls
sideways to pour it onto the plate.

Waypoints are left zero -- the VLA predicts them at rollout time.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import _safe_format
from generative_str_pipeline.build_dataset_0013_spoon_pour_reactive import (
    DESTINATIONS_POUR,
    MATERIAL_SIZE_M,
    MATERIALS_POUR,
    TEMPLATES_POUR,
    TOOL_LABEL,
)
# Reuse the exact pan geometry from the spatula flip so the env spawns an
# identical handleless pan for both tasks.
from generative_str_pipeline.sim_rollout.sample_flip_scenes import (
    PAN_LAYERS,
    PAN_RADIUS_M,
    PAN_SEGMENTS,
    PAN_WALL_HEIGHT_M,
    PAN_WALL_THICKNESS_M,
)

TABLE_Z = 0.53

# Material box dimensions (length x, width y, height z), matching the dataset.
POUR_MATERIAL_SIZE = [
    float(MATERIAL_SIZE_M[0]),
    float(MATERIAL_SIZE_M[1]),
    float(MATERIAL_SIZE_M[2]),
]
POUR_MATERIAL_HALF_Z = 0.5 * POUR_MATERIAL_SIZE[2]
_HALF_X = 0.5 * POUR_MATERIAL_SIZE[0]
_HALF_Y = 0.5 * POUR_MATERIAL_SIZE[1]
_HALF_XY_DIAG = float(np.linalg.norm([_HALF_X, _HALF_Y]))

# Deprecated fixed pan center kept for back-compat imports. The pan is now placed
# material-relative (see ``sample_pour_scene``); this is only a fallback default.
POUR_PAN_XYZ = [-0.08, 0.0, TABLE_Z]

# Reachable-workspace box for the material. The material xy is sampled first,
# then the pan is placed AROUND it so the pan center tracks the material instead
# of sitting at a fixed world point. The plate stays on the +x reachable side.
MATERIAL_X_RANGE_M = (-0.16, -0.04)
MATERIAL_Y_RANGE_M = (-0.06, 0.06)

# Object band, kept entirely inside the pan but near the rim (mirrors the flip
# sampler). This is the object's offset FROM the pan center; the pan center is
# derived as object_xy - offset so the object sits near the far (-x) rim.
OBJ_EDGE_MARGIN_M = 0.008
OBJ_EDGE_CENTER_R_MAX = PAN_RADIUS_M - _HALF_XY_DIAG - OBJ_EDGE_MARGIN_M
OBJ_EDGE_CENTER_R_MIN = max(0.01, OBJ_EDGE_CENTER_R_MAX - 0.030)
# The spoon stages outside the +x rim, so spawn the material near the opposite
# (-x) edge of the pan, farther from the tool start. theta is centered on pi
# (the -x rim) so the spoon has to reach across the pan to scoop the material.
OBJ_EDGE_THETA_RANGE = (0.60 * np.pi, 1.40 * np.pi)

# Goal plate: rests on the table OUTSIDE the pan, on the +x (reachable) side.
GOAL_X_RANGE = (0.08, 0.18)
GOAL_Y_RANGE = (-0.09, 0.12)
MIN_MATERIAL_GOAL_SEP_M = 0.16

# Spoon staging: outside the +x rim of the pan (the reachable arm side), lifted
# above the pan wall so it can swing over the lip and descend to scoop. Mirrors
# the flip sampler's spatula staging.
SPOON_FWD_M_RANGE = (PAN_RADIUS_M + 0.035, PAN_RADIUS_M + 0.065)
SPOON_Y_JITTER = 0.03
SPOON_STAGE_Z_ABOVE_WALL_M = 0.035


def _unit_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    v[2] = 0.0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def sample_pour_scene(rng: np.random.Generator, *, table_z: float = TABLE_Z) -> Dict[str, Any]:
    # Sample the material first inside the reachable workspace, then place the pan
    # AROUND it so the material sits near the far (-x) rim. The goal plate stays
    # on the +x reachable side, clear of the material.
    for _ in range(100):
        mat_x = float(rng.uniform(*MATERIAL_X_RANGE_M))
        mat_y = float(rng.uniform(*MATERIAL_Y_RANGE_M))
        theta = float(rng.uniform(*OBJ_EDGE_THETA_RANGE))
        radius = float(rng.uniform(OBJ_EDGE_CENTER_R_MIN, OBJ_EDGE_CENTER_R_MAX))
        offset = radius * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        pan_center = np.array([mat_x, mat_y], dtype=np.float64) - offset
        goal_xy = np.array(
            [float(rng.uniform(*GOAL_X_RANGE)), float(rng.uniform(*GOAL_Y_RANGE))],
            dtype=np.float32,
        )
        if float(np.linalg.norm(goal_xy - np.array([mat_x, mat_y]))) >= MIN_MATERIAL_GOAL_SEP_M:
            break

    material_xyz = np.array(
        [mat_x, mat_y, table_z + POUR_MATERIAL_HALF_Z], dtype=np.float32
    )
    # Goal plate rests on the table top, outside the pan.
    destination_xyz = np.array(
        [float(goal_xy[0]), float(goal_xy[1]), table_z], dtype=np.float32
    )

    fwd = float(rng.uniform(*SPOON_FWD_M_RANGE))
    spoon_x = float(pan_center[0]) + fwd  # stage outside +x rim (reachable arm side)
    spoon_y = mat_y + float(rng.uniform(-SPOON_Y_JITTER, SPOON_Y_JITTER))
    tool_contact = np.array(
        [spoon_x, spoon_y, table_z + PAN_WALL_HEIGHT_M + SPOON_STAGE_Z_ABOVE_WALL_M],
        dtype=np.float32,
    )

    motion = _unit_xy(material_xyz - tool_contact)  # ~ -x (toward material)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # bowl opening up
    surface_dir = (-motion).astype(np.float32)  # front lip -> handle, ~ +x

    material_word = str(rng.choice(MATERIALS_POUR))
    destination_word = str(rng.choice(DESTINATIONS_POUR))
    template = str(rng.choice(TEMPLATES_POUR))
    instruction = _safe_format(
        template, material=material_word, destination=destination_word, tool=TOOL_LABEL
    )

    return {
        "movement_token": "pour",
        "instruction": instruction,
        "tool_label": TOOL_LABEL,
        "tool_contact_xyz_world": tool_contact.tolist(),
        "tool_current_normal": normal.tolist(),
        "tool_current_surface_dir": surface_dir.tolist(),
        "material_label": material_word,
        "material_xyz_world": material_xyz.tolist(),
        "material_quat_world": [0.0, 0.0, 0.0, 1.0],
        "material_size": list(POUR_MATERIAL_SIZE),
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


def sample_pour_scenes(
    num_scenes: int, *, seed: int = 0, table_z: float = TABLE_Z
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    return [sample_pour_scene(rng, table_z=table_z) for _ in range(int(num_scenes))]
