"""Unit tests for the waypoint -> brush object pose adapter (CPU only)."""

from __future__ import annotations

import numpy as np

from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    contact_frame_world,
    matrix_from_quat_xyzw,
    quat_xyzw_from_matrix,
    waypoint_to_object_pose,
)


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_quat_matrix_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = _rand_rotation(rng)
        q = quat_xyzw_from_matrix(R)
        R2 = matrix_from_quat_xyzw(q)
        assert np.allclose(R, R2, atol=1e-8)


def test_contact_frame_is_orthonormal() -> None:
    T = contact_frame_world(
        contact_xyz=np.array([0.1, -0.2, 0.5]),
        normal=np.array([0.0, 0.0, 1.0]),
        surface_dir=np.array([1.0, 0.0, 0.3]),  # not orthogonal on purpose
    )
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    # normal column preserved
    assert np.allclose(R[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_waypoint_pose_roundtrip() -> None:
    """Placing the object at pose P and reading off its contact frame should
    recover P via the adapter."""
    rng = np.random.default_rng(1)
    # arbitrary object-local control frame
    T_oc = np.eye(4)
    T_oc[:3, :3] = _rand_rotation(rng)
    T_oc[:3, 3] = np.array([0.2, 0.01, 0.02])

    for _ in range(25):
        T_wo = np.eye(4)
        R_wo = _rand_rotation(rng)
        T_wo[:3, :3] = R_wo
        T_wo[:3, 3] = rng.uniform(-0.3, 0.3, size=3)

        # world contact frame implied by this object pose
        T_wc = T_wo @ T_oc
        contact = T_wc[:3, 3]
        surface_dir = T_wc[:3, 0]
        normal = T_wc[:3, 2]

        xyz, quat = waypoint_to_object_pose(contact, normal, surface_dir, T_oc)
        assert np.allclose(xyz, T_wo[:3, 3], atol=1e-7)
        R_rec = matrix_from_quat_xyzw(quat)
        assert np.allclose(R_rec, R_wo, atol=1e-7)
