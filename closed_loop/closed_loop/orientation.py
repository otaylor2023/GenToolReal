"""Pure (numpy-only) orientation-continuity helpers for closed-loop replanning.

No ROS, no torch. Shared by :class:`closed_loop.policy.ClosedLoopBrushPolicy`
(for in-policy orientation smoothing) and by the viz node (for flip detection),
so the smoothing logic and the visual flip detector agree on the definition of
the tool "forward axis" and the flip test.

The tool object orientation is built from the contact-frame triad
``(normal, surface_dir)`` (see ``waypoint_to_pose.contact_frame_world``). A
~180 degree replan flip shows up as the contact-frame ``surface_dir`` reversing,
which reverses the object's forward axis. The un-flip therefore rotates the
contact frame 180 degrees about its (unchanged) normal, i.e. negates
``surface_dir`` (and, after re-orthogonalization, the in-plane axis). This keeps
the contact POSITION fixed and only changes orientation.

By default :func:`align_waypoints_to_reference` only corrects ``surface_dir``
flips (180 degrees about the normal) and NEVER flips the normal, since that is
the only flip the model produces in practice.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from closed_loop.waypoint_to_pose import matrix_from_quat_xyzw, waypoint_to_object_pose

# All sign combinations applied to (surface_dir, normal). Each non-identity combo
# corresponds to one of the three nontrivial 180 degree contact-frame flips:
#   (+1, -1) -> 180 about surface_dir (tool forward / x axis)
#   (-1, +1) -> 180 about normal      (z axis)  [the user's case]
#   (-1, -1) -> 180 about y axis
# The default smoothing path only corrects surface_dir flips (180 about the
# normal, combo (-1, +1)) and NEVER flips the normal; see _SURFACE_DIR_COMBOS and
# the ``surface_dir_only`` flag on :func:`align_waypoints_to_reference`.
_SIGN_COMBOS: Tuple[Tuple[float, float], ...] = (
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)

# surface_dir-only combos: identity and a 180 about the normal (negate surface_dir).
# The normal column (cols 3:6) is never negated under these combos.
_SURFACE_DIR_COMBOS: Tuple[Tuple[float, float], ...] = (
    (1.0, 1.0),
    (-1.0, 1.0),
)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return v / n


def forward_axis_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Tool forward axis (object-local +x) expressed in the parent frame, unit length."""
    R = matrix_from_quat_xyzw(quat_xyzw)
    return _unit(R[:, 0])


def orientation_flip(
    prev_axis: np.ndarray, new_axis: np.ndarray, *, eps: float = 1e-6
) -> tuple[bool, float]:
    """Detect a ~180 degree flip between two forward axes.

    Returns ``(flipped, dot)`` where ``dot`` is the cosine between the unit axes;
    ``flipped`` is True when the axes point in opposing hemispheres (dot < 0).
    Zero-length axes are treated as not flipped (dot = 1.0).
    """
    a = _unit(prev_axis)
    b = _unit(new_axis)
    if float(np.linalg.norm(a)) < eps or float(np.linalg.norm(b)) < eps:
        return False, 1.0
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return dot < 0.0, dot


def is_orientation_flipped(
    prev_axis: np.ndarray, new_axis: np.ndarray, *, eps: float = 1e-6
) -> bool:
    """Boolean form of :func:`orientation_flip`."""
    return orientation_flip(prev_axis, new_axis, eps=eps)[0]


def _apply_sign_combo(waypoints: np.ndarray, ss: float, sn: float) -> np.ndarray:
    """Return a ``[N, 9]`` copy with ``surface_dir`` scaled by ``ss`` (cols 6:9)
    and ``normal`` scaled by ``sn`` (cols 3:6). Contact xyz (cols 0:3) is fixed."""
    wp = np.array(waypoints, dtype=np.float64, copy=True)
    if ss < 0.0:
        wp[:, 6:9] = -wp[:, 6:9]
    if sn < 0.0:
        wp[:, 3:6] = -wp[:, 3:6]
    return wp


def unflip_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Return a copy of ``[N, 9]`` waypoints with the contact-frame orientation
    rotated 180 degrees about the (unchanged) normal.

    This negates ``surface_dir`` (columns 6:9). Re-deriving the object pose from
    the result rotates the tool 180 degrees about its contact normal, which
    reverses the object forward axis while leaving the contact xyz (columns 0:3)
    and the normal (columns 3:6) untouched.
    """
    return _apply_sign_combo(waypoints, -1.0, 1.0)


def quat_geodesic_closeness(q_ref: np.ndarray, q: np.ndarray) -> float:
    """Rotation closeness of two quaternions accounting for double cover.

    Returns ``abs(dot(q_ref, q))`` on the unit quaternions: ``1.0`` means the two
    quaternions represent the identical rotation (q or -q), ``0.0`` means a 180
    degree separation. Zero-length inputs return ``0.0``.
    """
    a = np.asarray(q_ref, dtype=np.float64).reshape(4)
    b = np.asarray(q, dtype=np.float64).reshape(4)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(a / na, b / nb)))


def align_waypoints_to_reference(
    waypoints: np.ndarray,
    ref_quat: Optional[np.ndarray],
    T_oc: np.ndarray,
    surface_dir_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Pick the (surface_dir, normal) sign combo whose object orientation is
    closest to ``ref_quat`` and apply it to ALL waypoints.

    By default (``surface_dir_only=True``) this only corrects ``surface_dir``
    flips, i.e. it searches just the combos ``(+1, +1)`` and ``(-1, +1)`` (a 180
    degree rotation about the contact normal) and NEVER negates the normal column
    (cols 3:6). This matches the only flip the model exhibits in practice.

    When ``surface_dir_only=False`` the full four-combo search over
    ``(surface_dir, normal)`` is used. Each non-identity combo is a 180 degree
    flip about one contact-frame axis, so it can also correct flips about the
    tool forward axis (negate normal) and the y axis (negate both). In all cases
    the world contact point is left unchanged.

    Returns ``(aligned_waypoints, info)`` where ``info`` has keys ``applied``
    (combo != (+1, +1)), ``combo`` (the chosen ``(ss, sn)``), ``closeness_before``
    (closeness of the raw combo), and ``closeness_after`` (closeness of the chosen
    combo). If ``ref_quat`` is None or there are no waypoints, returns an unchanged
    copy with ``applied=False``.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    info: Dict[str, object] = {
        "applied": False,
        "combo": (1.0, 1.0),
        "closeness_before": None,
        "closeness_after": None,
    }
    if ref_quat is None or wp.ndim != 2 or wp.shape[0] == 0:
        return np.array(waypoints, dtype=np.float64, copy=True), info

    combos = _SURFACE_DIR_COMBOS if surface_dir_only else _SIGN_COMBOS
    rep = wp[0]
    contact = rep[0:3]
    normal = rep[3:6]
    surface_dir = rep[6:9]
    base_close: Optional[float] = None
    best_close = -1.0
    best_combo: Tuple[float, float] = (1.0, 1.0)
    for ss, sn in combos:
        _xyz, q = waypoint_to_object_pose(contact, sn * normal, ss * surface_dir, T_oc)
        close = quat_geodesic_closeness(ref_quat, q)
        if (ss, sn) == (1.0, 1.0):
            base_close = close
        if close > best_close:
            best_close = close
            best_combo = (ss, sn)

    aligned = _apply_sign_combo(wp, best_combo[0], best_combo[1])
    info["combo"] = best_combo
    info["closeness_before"] = base_close
    info["closeness_after"] = best_close
    info["applied"] = best_combo != (1.0, 1.0)
    return aligned, info
