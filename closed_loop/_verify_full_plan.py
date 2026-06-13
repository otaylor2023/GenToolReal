"""CPU-only verification for full-plan accessors, flip detection, and the
in-policy full-orientation nearest-flip smoothing (un-flip) feature.

Run from the closed_loop/ dir:
    CUDA_VISIBLE_DEVICES='' python _verify_full_plan.py
"""

from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1] / "simtoolreal" / "deployment"
sys.path.insert(0, str(DEPLOY))

import goal_pose_node_closed_loop_viz as viz  # noqa: E402

import closed_loop.policy as policy_mod  # noqa: E402
from closed_loop import ClosedLoopBrushPolicy  # noqa: E402
from closed_loop.orientation import (  # noqa: E402
    align_waypoints_to_reference,
    forward_axis_from_quat,
    orientation_flip,
    quat_geodesic_closeness,
    unflip_waypoints,
)
from closed_loop.waypoint_to_pose import (  # noqa: E402
    matrix_from_quat_xyzw,
    waypoint_to_object_pose,
)


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


# Non-degenerate control frame: rotation about the contact x-axis (so the object
# forward axis stays aligned with surface_dir, making the negate-normal flip a
# true roll about the forward axis) plus a translation (so the object ROOT moves
# under the flip while the contact POINT stays fixed).
def _hand_built_T_oc() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot_x(30.0)
    T[:3, 3] = np.array([0.05, -0.02, 0.10], dtype=np.float64)
    return T


def _base_waypoints() -> np.ndarray:
    wp = np.zeros((15, 9), dtype=np.float64)
    wp[:, 0] = np.linspace(-0.2, -0.36, 15)
    wp[:, 1] = 0.03
    wp[:, 2] = 0.55
    wp[:, 3:6] = np.array([0.0, 0.0, 1.0])  # normal +z
    wp[:, 6:9] = np.array([1.0, 0.0, 0.0])  # surface_dir +x
    return wp


def _flip(wp: np.ndarray, kind: str) -> np.ndarray:
    out = wp.copy()
    if kind in ("x", "y"):  # x-flip / y-flip negate the normal
        out[:, 3:6] = -out[:, 3:6]
    if kind in ("z", "y"):  # z-flip / y-flip negate surface_dir
        out[:, 6:9] = -out[:, 6:9]
    return out


def _axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    # Rotation axis = the real eigenvector with eigenvalue +1.
    vals, vecs = np.linalg.eig(R)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    axis = np.real(vecs[:, idx])
    return axis / max(np.linalg.norm(axis), 1e-12), angle


def _obj_quat(wp_row: np.ndarray, T_oc: np.ndarray) -> np.ndarray:
    _xyz, q = waypoint_to_object_pose(wp_row[0:3], wp_row[3:6], wp_row[6:9], T_oc)
    return q


def _world_contact(wp_row: np.ndarray, T_oc: np.ndarray) -> np.ndarray:
    xyz, q = waypoint_to_object_pose(wp_row[0:3], wp_row[3:6], wp_row[6:9], T_oc)
    T_wo = np.eye(4)
    T_wo[:3, :3] = matrix_from_quat_xyzw(q)
    T_wo[:3, 3] = xyz
    return (T_wo @ T_oc)[:3, 3]


def test_flip_detection() -> None:
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])
    q_180_z = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about z
    q_small = np.array([0.0, 0.0, 0.0871557, 0.9961947])  # ~10 deg about z

    ax_id = forward_axis_from_quat(q_identity)
    ax_small = forward_axis_from_quat(q_small)
    ax_180 = forward_axis_from_quat(q_180_z)

    flipped_aligned, dot_aligned = orientation_flip(ax_id, ax_small)
    flipped_180, dot_180 = orientation_flip(ax_id, ax_180)
    flipped_90, dot_90 = orientation_flip(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )

    assert not flipped_aligned, f"aligned should NOT flip (dot={dot_aligned})"
    assert flipped_180, f"180deg should flip (dot={dot_180})"
    assert dot_180 < -0.99, f"180deg dot should be ~-1, got {dot_180}"
    assert not flipped_90, f"90deg (dot=0) should NOT flip, got dot={dot_90}"
    assert viz.forward_axis_from_quat is forward_axis_from_quat
    assert viz.orientation_flip is orientation_flip
    print(
        f"[ok] flip detection: aligned dot={dot_aligned:+.3f} (no flip), "
        f"90deg dot={dot_90:+.3f} (no flip), 180deg dot={dot_180:+.3f} (flip)"
    )


def test_unflip_preserves_contact_position() -> None:
    """Legacy unflip (= negate surface_dir) keeps the contact point fixed while
    moving the object ROOT under an offset T_oc."""
    T_oc = np.eye(4)
    T_oc[:3, 3] = np.array([0.05, -0.02, 0.10])

    wp = _base_waypoints()
    wp_unflipped = unflip_waypoints(wp)
    assert np.allclose(wp_unflipped[:, 0:3], wp[:, 0:3]), "contact xyz must not change"
    assert np.allclose(wp_unflipped[:, 3:6], wp[:, 3:6]), "normal must not change"
    assert np.allclose(wp_unflipped[:, 6:9], -wp[:, 6:9]), "surface_dir must negate"

    root_moved = False
    for i in range(wp.shape[0]):
        c0 = _world_contact(wp[i], T_oc)
        c1 = _world_contact(wp_unflipped[i], T_oc)
        assert np.allclose(c0, c1, atol=1e-9), f"contact point preserved (wp {i})"
        xyz0, _ = waypoint_to_object_pose(wp[i, 0:3], wp[i, 3:6], wp[i, 6:9], T_oc)
        xyz1, _ = waypoint_to_object_pose(
            wp_unflipped[i, 0:3], wp_unflipped[i, 3:6], wp_unflipped[i, 6:9], T_oc
        )
        if not np.allclose(xyz0, xyz1, atol=1e-6):
            root_moved = True
    assert root_moved, "with an offset T_oc the object ROOT should move under unflip"
    print("[ok] legacy unflip preserves contact position (root moves), negates surface_dir")


def _assert_180(q_base: np.ndarray, q_flip: np.ndarray, label: str) -> float:
    R_rel = matrix_from_quat_xyzw(q_flip) @ matrix_from_quat_xyzw(q_base).T
    _axis, angle = _axis_angle(R_rel)
    assert abs(angle - 180.0) < 1e-3, f"{label} must be a 180 deg rotation, got {angle}"
    return angle


def test_geometry_surface_dir_flip_corrected() -> None:
    """surface_dir flip (negate cols 6:9) IS corrected under the default
    surface_dir_only=True: combo (-1, 1) (180 about the normal), the normal
    column is never touched, and the world contact point is preserved."""
    T_oc = _hand_built_T_oc()
    base = _base_waypoints()
    flipped = _flip(base, "z")  # negate surface_dir

    q_base = _obj_quat(base[0], T_oc)
    angle = _assert_180(q_base, _obj_quat(flipped[0], T_oc), "surface_dir-flip")

    aligned, info = align_waypoints_to_reference(flipped, q_base, T_oc)
    assert info["applied"], "surface_dir flip must be corrected by default"
    assert info["combo"] == (-1.0, 1.0), f"combo should be (-1,1), got {info['combo']}"
    assert float(info["closeness_after"]) > 0.999, (
        f"post-alignment closeness must be ~1, got {info['closeness_after']}"
    )
    assert np.allclose(aligned[:, 3:6], flipped[:, 3:6]), "normal column must be untouched"

    for i in range(base.shape[0]):
        c_flip = _world_contact(flipped[i], T_oc)
        c_align = _world_contact(aligned[i], T_oc)
        assert np.allclose(c_flip, c_align, atol=1e-12), (
            f"alignment must preserve world contact point (wp {i})"
        )
        assert np.allclose(c_align, base[i, 0:3], atol=1e-12), "contact = contact_xyz"
        q_a = _obj_quat(aligned[i], T_oc)
        assert quat_geodesic_closeness(_obj_quat(base[i], T_oc), q_a) > 0.999, (
            f"aligned orientation must match base (wp {i})"
        )
    print(
        f"[ok] surface_dir flip CORRECTED: 180deg rot (angle={angle:.1f}), "
        f"combo={info['combo']} closeness "
        f"{info['closeness_before']:.3f}->{info['closeness_after']:.3f}, "
        f"normal untouched, contact point preserved"
    )


def test_geometry_normal_flip_not_corrected() -> None:
    """normal flip (negate cols 3:6) is NEVER corrected under the default; the
    normal column is left exactly as the model produced it (still negated)."""
    T_oc = _hand_built_T_oc()
    base = _base_waypoints()
    flipped = _flip(base, "x")  # negate normal

    q_base = _obj_quat(base[0], T_oc)
    _assert_180(q_base, _obj_quat(flipped[0], T_oc), "normal-flip")

    aligned, info = align_waypoints_to_reference(flipped, q_base, T_oc)
    assert not info["applied"], "normal flip must NOT be corrected by default"
    assert info["combo"] == (1.0, 1.0), f"combo should be identity, got {info['combo']}"
    assert np.allclose(aligned[:, 3:6], flipped[:, 3:6]), (
        "normal column must be unchanged (function never touches the normal)"
    )
    assert np.allclose(aligned[:, 3:6], -base[:, 3:6]), "normal stays negated as produced"
    assert np.allclose(aligned[:, 6:9], flipped[:, 6:9]), "surface_dir unchanged too"
    print(
        f"[ok] normal flip NOT corrected: applied={info['applied']}, "
        f"combo={info['combo']}, normal column left negated as produced"
    )


def test_geometry_normal_flip_corrected_general() -> None:
    """With surface_dir_only=False the general four-combo path still corrects a
    normal flip via combo (1, -1) (180 about surface_dir)."""
    T_oc = _hand_built_T_oc()
    base = _base_waypoints()
    flipped = _flip(base, "x")  # negate normal

    q_base = _obj_quat(base[0], T_oc)
    aligned, info = align_waypoints_to_reference(
        flipped, q_base, T_oc, surface_dir_only=False
    )
    assert info["applied"], "general path must correct the normal flip"
    assert info["combo"] == (1.0, -1.0), f"combo should be (1,-1), got {info['combo']}"
    assert float(info["closeness_after"]) > 0.999, (
        f"post-alignment closeness must be ~1, got {info['closeness_after']}"
    )
    for i in range(base.shape[0]):
        q_a = _obj_quat(aligned[i], T_oc)
        assert quat_geodesic_closeness(_obj_quat(base[i], T_oc), q_a) > 0.999, (
            f"general-path aligned orientation must match base (wp {i})"
        )
    print(
        f"[ok] normal flip corrected by general path (surface_dir_only=False): "
        f"combo={info['combo']} closeness "
        f"{info['closeness_before']:.3f}->{info['closeness_after']:.3f}"
    )


class _FakeFlippingBrush:
    """Stand-in for BrushPolicy that flips on the second prediction. The default
    ``flip_kind='z'`` negates the SURFACE_DIR (a 180-deg roll about the contact
    normal), which is the only flip the model produces in practice and the only
    one smoothing should correct. ``flip_kind='x'`` negates the NORMAL instead
    (never corrected). Uses a non-identity control frame so the root moves."""

    flip_kind = "z"

    def __init__(self, **kwargs):
        self.T_oc = _hand_built_T_oc()
        self.table_z = 0.53
        self.tool_label = "the brush"
        self.control_frame_path = Path("fake.json")
        self._calls = 0

    def predict_waypoints(self, scene) -> np.ndarray:
        wp = _base_waypoints().astype(np.float32)
        if self._calls > 0:
            wp = _flip(wp, self.flip_kind).astype(np.float32)
        self._calls += 1
        return wp

    def waypoints_to_object_poses_robot(self, waypoints, frame_shift):
        shift = np.asarray(frame_shift, dtype=np.float64).reshape(3)
        out = []
        for i in range(waypoints.shape[0]):
            xyz_m, quat = waypoint_to_object_pose(
                waypoints[i, 0:3], waypoints[i, 3:6], waypoints[i, 6:9], self.T_oc
            )
            out.append(((xyz_m - shift).astype(np.float64), quat.astype(np.float64)))
        return out


def _drive_two_plans(unflip: bool, flip_kind: str = "z"):
    orig = policy_mod.BrushPolicy
    prev_kind = _FakeFlippingBrush.flip_kind
    _FakeFlippingBrush.flip_kind = flip_kind
    policy_mod.BrushPolicy = _FakeFlippingBrush
    try:
        pol = ClosedLoopBrushPolicy(
            device="cpu",
            control_frame="blue_brush",
            instruction="push the cube",
            frame_shift=(0.0, 0.8, 0.0),
            chunk_size=5,
            tool_pose_is_root=False,
            unflip_orientation=unflip,
        )
        pol.set_destination(np.array([-0.365, -0.056, 0.517]))
        tool_xyz = np.array([-0.036, 0.030, 0.548])
        tool_quat = np.array([0.0, 0.0, 0.0, 1.0])
        mat_xyz = np.array([-0.221, 0.035, 0.549])

        pol.reset(tool_xyz, tool_quat, mat_xyz)
        q_first = np.asarray(pol.last_plan_object_poses[0][1], dtype=np.float64)

        pol.observe(tool_xyz, tool_quat, mat_xyz)
        q_second = np.asarray(pol.last_plan_object_poses[0][1], dtype=np.float64)
        q_chunk = np.asarray(pol._current_chunk[0][1], dtype=np.float64)
        return pol, q_first, q_second, q_chunk
    finally:
        policy_mod.BrushPolicy = orig
        _FakeFlippingBrush.flip_kind = prev_kind


def test_policy_unflip_on_surface_dir() -> None:
    pol, q_first, q_second, q_chunk = _drive_two_plans(unflip=True, flip_kind="z")
    close = quat_geodesic_closeness(q_first, q_second)
    close_c = quat_geodesic_closeness(q_first, q_chunk)
    assert close > 0.999, f"smoothing ON: replanned orientation must stay continuous, {close}"
    assert close_c > 0.999, f"smoothing ON: chunk orientation must stay continuous, {close_c}"
    assert pol.last_replan_unflipped, "policy should report it corrected this surface_dir flip"
    assert len(pol.last_plan_object_poses) == 15
    assert len(pol._current_chunk) == 5
    print(
        f"[ok] policy smoothing ON (surface_dir flip): plan/chunk continuous "
        f"(closeness={close:.4f}/{close_c:.4f}), last_replan_unflipped="
        f"{pol.last_replan_unflipped}"
    )


def test_policy_normal_flip_not_corrected() -> None:
    pol, q_first, q_second, _q_chunk = _drive_two_plans(unflip=True, flip_kind="x")
    close = quat_geodesic_closeness(q_first, q_second)
    assert close < 0.01, (
        f"smoothing ON but normal flip must NOT be corrected (flip persists), {close}"
    )
    assert not pol.last_replan_unflipped, "policy must NOT report correcting a normal flip"
    print(
        f"[ok] policy smoothing ON (normal flip): NOT corrected, flip persists "
        f"(closeness={close:.4f}), last_replan_unflipped={pol.last_replan_unflipped}"
    )


def test_policy_unflip_off() -> None:
    pol, q_first, q_second, _q_chunk = _drive_two_plans(unflip=False, flip_kind="z")
    close = quat_geodesic_closeness(q_first, q_second)
    assert close < 0.01, f"smoothing OFF: the ~180 deg replan flip must persist, {close}"
    assert not pol.last_replan_unflipped, "no correction should be reported when disabled"
    print(f"[ok] policy smoothing OFF: surface_dir replan flip persists (closeness={close:.4f})")


def test_viz_closed_loop_continuity() -> None:
    """Drive InteractiveBrushViz._play_closed_loop with a fake surface_dir-flipping
    policy and assert the drawn object_poses[0] orientation stays continuous."""
    from closed_loop.tools.viz_interactive import InteractiveBrushViz

    app = object.__new__(InteractiveBrushViz)
    app.policy = _FakeFlippingBrush()  # default flip_kind="z" (surface_dir)
    app.policy.instruction = "push the cube"
    app.table_z = 0.53
    app._chunk_size = 5
    app._max_replans = 3
    app._unflip_orientation = True
    app._play_stop = threading.Event()
    app.status_md = types.SimpleNamespace(content="")
    app._play_finished_status = None

    drawn_quats: list = []
    app._sync_policy_instruction = lambda: None
    app._read_gizmo_poses = lambda: (
        np.array([-0.036, 0.030, 0.548]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([-0.221, 0.035, 0.549]),
        np.array([0.90, 0.90, 0.517]),  # far goal so it never delivers
    )
    app._update_pose_panel = lambda poses: None
    app._animate_poses = lambda poses: True
    app._set_cube_pose = lambda xyz: None
    app._set_tool_pose = lambda xyz, quat: None

    def _capture_draw(waypoints, object_poses):
        drawn_quats.append(np.asarray(object_poses[0][1], dtype=np.float64))

    app._draw_plan = _capture_draw

    app._play_closed_loop()

    assert len(drawn_quats) >= 2, f"expected >=2 generations, got {len(drawn_quats)}"
    close = quat_geodesic_closeness(drawn_quats[0], drawn_quats[1])
    assert close > 0.999, f"viz: drawn orientation must stay continuous across gens, {close}"
    print(
        f"[ok] viz closed-loop: {len(drawn_quats)} gens drawn, gen0->gen1 "
        f"orientation continuous (closeness={close:.4f})"
    )


if __name__ == "__main__":
    test_flip_detection()
    test_unflip_preserves_contact_position()
    test_geometry_surface_dir_flip_corrected()
    test_geometry_normal_flip_not_corrected()
    test_geometry_normal_flip_corrected_general()
    test_policy_unflip_on_surface_dir()
    test_policy_normal_flip_not_corrected()
    test_policy_unflip_off()
    test_viz_closed_loop_continuity()
    print("ALL CPU-ONLY VERIFICATION PASSED")
