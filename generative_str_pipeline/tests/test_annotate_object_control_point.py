"""Unit tests for control-frame geometry (no Viser)."""

from __future__ import annotations

import numpy as np

from generative_str_pipeline.annotate_object_control_point import (
    build_T_obj_from_contact,
    regularize_rectangle,
)


def test_regularize_rectangle_and_transform() -> None:
    fl = np.array([0.0, -0.02, 0.0])
    fr = np.array([0.0, 0.02, 0.0])
    bl = np.array([0.1, -0.02, 0.0])
    br = np.array([0.1, 0.02, 0.0])
    centroid = np.array([0.05, 0.0, -0.01])

    corners_rect, cp, sd, n = regularize_rectangle(
        fl, fr, bl, br, centroid, flip_normal=False
    )

    assert np.allclose(cp, np.zeros(3), atol=1e-5)
    assert np.allclose(sd, np.array([1.0, 0.0, 0.0]), atol=1e-5)
    assert abs(float(np.dot(sd, n))) < 1e-5

    front_mid = 0.5 * (corners_rect["front_left"] + corners_rect["front_right"])
    assert np.allclose(cp, front_mid)

    T = build_T_obj_from_contact(cp, sd, n)
    assert abs(float(np.linalg.det(T[:3, :3])) - 1.0) < 1e-5
