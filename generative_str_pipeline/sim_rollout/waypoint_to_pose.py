"""Convert VLA contact-frame waypoints to brush object 7D world poses.

A VLA waypoint is (contact_xyz, normal, surface_dir) in world space. The brush
object's contact frame (control point + surface_dir + normal, expressed in the
object-local frame) is captured once by the annotation tool as a 4x4 transform
``T_obj_from_contact`` that maps contact-frame coordinates to object-local
coordinates.

For a desired world contact frame ``T_world_contact`` we recover the brush
object world pose ``T_world_obj`` such that the object's contact frame lands on
the desired one:

    T_world_contact = T_world_obj @ T_obj_from_contact
    => T_world_obj  = T_world_contact @ inv(T_obj_from_contact)

The object pose is returned as ``(xyz, quat_xyzw)`` to match the dextoolbench
trajectory JSON / IsaacGym convention.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

Vec3 = np.ndarray
Mat4 = np.ndarray


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def contact_frame_world(
    contact_xyz: np.ndarray,
    normal: np.ndarray,
    surface_dir: np.ndarray,
) -> Mat4:
    """Build a 4x4 world transform for a contact frame.

    Columns are [surface_dir, normal x surface_dir, normal], matching the
    convention used by the annotation tool's ``T_obj_from_contact``. ``surface_dir``
    is re-orthogonalized against ``normal`` so the result is a proper rotation.
    """
    n = _unit(normal)
    s_raw = np.asarray(surface_dir, dtype=np.float64).reshape(3)
    s = s_raw - np.dot(s_raw, n) * n
    s = _unit(s)
    y = np.cross(n, s)
    R = np.stack([s, y, n], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(contact_xyz, dtype=np.float64).reshape(3)
    return T


def quat_xyzw_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3) -> unit quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def matrix_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [x, y, z, w] -> rotation matrix (3x3)."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_control_frame(path: Path) -> Mat4:
    """Load ``T_obj_from_contact`` (4x4) from an annotation JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    T = np.asarray(data["T_obj_from_contact"], dtype=np.float64).reshape(4, 4)
    return T


def waypoint_to_object_pose(
    contact_xyz: np.ndarray,
    normal: np.ndarray,
    surface_dir: np.ndarray,
    T_obj_from_contact: Mat4,
) -> Tuple[Vec3, np.ndarray]:
    """Return brush object world pose ``(xyz, quat_xyzw)`` for one waypoint."""
    T_wc = contact_frame_world(contact_xyz, normal, surface_dir)
    T_wo = T_wc @ np.linalg.inv(np.asarray(T_obj_from_contact, dtype=np.float64))
    xyz = T_wo[:3, 3].copy()
    quat = quat_xyzw_from_matrix(T_wo[:3, :3])
    return xyz, quat


@functools.lru_cache(maxsize=4)
def _load_obj_min_extent_vertices(mesh_path: str) -> np.ndarray:
    """Load OBJ vertices (object-local frame) as an [N,3] array (cached)."""
    verts = []
    with open(mesh_path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3)


def flat_rest_object_pose(
    contact_xyz: np.ndarray,
    heading_dir: np.ndarray,
    T_obj_from_contact: Mat4,
    *,
    table_z: float,
    mesh_path: Optional[str] = None,
    clearance_m: float = 0.002,
    up_sign: float = 1.0,
    contact_normal: Optional[np.ndarray] = None,
    contact_surface_dir: Optional[np.ndarray] = None,
    table_half_x: Optional[float] = None,
    table_half_y: Optional[float] = None,
    table_x_bounds: Optional[Tuple[float, float]] = None,
    table_y_bounds: Optional[Tuple[float, float]] = None,
    table_margin_m: float = 0.01,
) -> Tuple[Vec3, np.ndarray]:
    """Brush/tool object pose resting on the table at startup.

    Default (brush) behavior: the contact-frame ``surface_dir`` is forced vertical
    (``+/- z`` world) and the contact-frame ``normal`` is laid horizontal along
    ``heading_dir`` so the tool rests flat on the table instead of standing/tilted.

    Upright override: when both ``contact_normal`` and ``contact_surface_dir`` are
    given, the start pose realizes that *actual* contact frame instead (used for
    the spatula flip, where the trained ``theta=0`` pose has the blade face up
    (``normal=+z``) and a horizontal ``surface_dir`` — i.e. right-side up).

    When ``mesh_path`` is given, the object height is set from the rotated mesh so
    its lowest vertex sits at ``table_z + clearance_m`` (no clipping into the
    table); otherwise the contact point is anchored at ``table_z``.

    Footprint clamping keeps the whole rotated tool on the table. Pass
    ``table_x_bounds``/``table_y_bounds`` as ``(min, max)`` for an asymmetric
    table (e.g. a +x-extended table for brush staging); otherwise the symmetric
    ``table_half_x``/``table_half_y`` are used as ``(-half, +half)``.
    """
    c = np.asarray(contact_xyz, dtype=np.float64).reshape(3).copy()
    c[2] = float(table_z)
    if contact_normal is not None and contact_surface_dir is not None:
        # Upright: keep the tool's real contact-frame orientation (e.g. spatula
        # blade face up, handle horizontal) rather than the flat-lay override.
        n = np.asarray(contact_normal, dtype=np.float64).reshape(3)
        s = np.asarray(contact_surface_dir, dtype=np.float64).reshape(3)
        xyz, quat = waypoint_to_object_pose(c, n, s, T_obj_from_contact)
    else:
        h = np.asarray(heading_dir, dtype=np.float64).reshape(3).copy()
        h[2] = 0.0
        if float(np.linalg.norm(h)) < 1e-6:
            h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        h = h / float(np.linalg.norm(h))
        surface_dir = np.array([0.0, 0.0, float(np.sign(up_sign) or 1.0)], dtype=np.float64)
        xyz, quat = waypoint_to_object_pose(c, h, surface_dir, T_obj_from_contact)
    if mesh_path is not None:
        try:
            verts = _load_obj_min_extent_vertices(str(mesh_path))
            R = matrix_from_quat_xyzw(quat)
            world = verts @ R.T  # vertices in world orientation, relative to origin
            xyz = xyz.copy()
            xyz[2] = float(table_z) + float(clearance_m) - float(world[:, 2].min())
            margin = float(table_margin_m)
            bounds = [None, None]
            if table_x_bounds is not None:
                bounds[0] = (float(table_x_bounds[0]), float(table_x_bounds[1]))
            elif table_half_x is not None:
                bounds[0] = (-float(table_half_x), float(table_half_x))
            if table_y_bounds is not None:
                bounds[1] = (float(table_y_bounds[0]), float(table_y_bounds[1]))
            elif table_half_y is not None:
                bounds[1] = (-float(table_half_y), float(table_half_y))
            for ax, bd in enumerate(bounds):
                if bd is None:
                    continue
                bmin = bd[0] + margin
                bmax = bd[1] - margin
                lo = xyz[ax] + float(world[:, ax].min())
                hi = xyz[ax] + float(world[:, ax].max())
                if lo < bmin:
                    xyz[ax] += (bmin - lo)
                elif hi > bmax:
                    xyz[ax] -= (hi - bmax)
        except (OSError, ValueError):
            pass
    return xyz, quat


def waypoints_to_object_poses(
    contacts: np.ndarray,
    normals: np.ndarray,
    surface_dirs: np.ndarray,
    T_obj_from_contact: Mat4,
) -> np.ndarray:
    """Convert [N,3] contacts/normals/surface_dirs to [N,7] poses (xyz + quat_xyzw)."""
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    surface_dirs = np.asarray(surface_dirs, dtype=np.float64).reshape(-1, 3)
    n = contacts.shape[0]
    out = np.zeros((n, 7), dtype=np.float64)
    for i in range(n):
        xyz, quat = waypoint_to_object_pose(
            contacts[i], normals[i], surface_dirs[i], T_obj_from_contact
        )
        out[i, :3] = xyz
        out[i, 3:7] = quat
    return out
