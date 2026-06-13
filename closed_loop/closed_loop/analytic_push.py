"""Self-contained analytic brush push model (matches dataset_0011 constants)."""

from __future__ import annotations

import numpy as np

BRUSH_BODY_FRONT_M = 0.05
OBJECT_RADIUS_M = 0.02
CONTACT_XY_TOL_M = 0.03
CONTACT_Z_TOL_M = 0.035
GOAL_REGION_RADIUS_M = 0.05


def sweep_unit(material_xyz: np.ndarray, destination_xyz: np.ndarray) -> np.ndarray:
    sweep_vec = destination_xyz - material_xyz
    sweep_vec[2] = 0.0
    n = float(np.linalg.norm(sweep_vec))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (sweep_vec / n).astype(np.float32)


def object_from_brush(brush_contact: np.ndarray, sweep_unit_vec: np.ndarray, table_z: float) -> np.ndarray:
    offset = float(BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M)
    xy = brush_contact[:2] + sweep_unit_vec[:2] * offset
    return np.array([xy[0], xy[1], table_z], dtype=np.float32)


def execute_chunk(
    plan: np.ndarray,
    *,
    object_xyz: np.ndarray,
    in_contact: bool,
    destination_xyz: np.ndarray,
    table_z: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Execute first ``chunk`` waypoints; return (new_brush_wp[9], new_obj, new_contact)."""
    dest = np.asarray(destination_xyz, dtype=np.float32).reshape(3)
    obj = np.asarray(object_xyz, dtype=np.float32).reshape(3).copy()
    contact = bool(in_contact)
    n_exec = max(1, min(int(chunk), int(plan.shape[0])))
    contact_xy_radius = float(BRUSH_BODY_FRONT_M + OBJECT_RADIUS_M + CONTACT_XY_TOL_M)
    sweep = sweep_unit(obj, dest)

    for i in range(n_exec):
        bc = plan[i, 0:3].astype(np.float32)
        if not contact:
            near_xy = float(np.linalg.norm(bc[:2] - obj[:2])) <= contact_xy_radius
            low = float(bc[2]) <= float(obj[2]) + float(CONTACT_Z_TOL_M)
            if near_xy and low:
                contact = True
        if contact:
            pushed = object_from_brush(bc, sweep, table_z)
            pushed[2] = obj[2]
            if float(np.dot(dest[:2] - pushed[:2], sweep[:2])) < 0.0:
                pushed[0], pushed[1] = float(dest[0]), float(dest[1])
            obj = pushed.astype(np.float32)

    new_brush = plan[n_exec - 1].astype(np.float32)
    return new_brush, obj, contact
