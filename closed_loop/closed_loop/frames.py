"""Robot <-> model frame transforms and contact-frame recovery from tool root pose."""

from __future__ import annotations

import numpy as np

from closed_loop.waypoint_to_pose import matrix_from_quat_xyzw


def robot_to_model_xyz(xyz: np.ndarray, frame_shift: np.ndarray) -> np.ndarray:
    return np.asarray(xyz, dtype=np.float64).reshape(3) + np.asarray(frame_shift, dtype=np.float64)


def model_to_robot_xyz(xyz: np.ndarray, frame_shift: np.ndarray) -> np.ndarray:
    return np.asarray(xyz, dtype=np.float64).reshape(3) - np.asarray(frame_shift, dtype=np.float64)


def contact_frame_from_root(
    root_xyz: np.ndarray,
    root_quat_xyzw: np.ndarray,
    T_oc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover world contact point + (normal, surface_dir) from tool root pose."""
    R = matrix_from_quat_xyzw(np.asarray(root_quat_xyzw, dtype=np.float64).reshape(4))
    T_wo = np.eye(4, dtype=np.float64)
    T_wo[:3, :3] = R
    T_wo[:3, 3] = np.asarray(root_xyz, dtype=np.float64).reshape(3)
    T_wc = T_wo @ np.asarray(T_oc, dtype=np.float64).reshape(4, 4)
    contact = T_wc[:3, 3].astype(np.float32)
    surface_dir = T_wc[:3, 0].astype(np.float32)
    normal = T_wc[:3, 2].astype(np.float32)
    return contact, normal, surface_dir


def contact_frame_direct(xyz: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpret root pose rotation columns as [surface_dir, y, normal] (contact frame)."""
    R = matrix_from_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    contact = np.asarray(xyz, dtype=np.float64).reshape(3).astype(np.float32)
    return contact, R[:, 2].astype(np.float32), R[:, 0].astype(np.float32)
