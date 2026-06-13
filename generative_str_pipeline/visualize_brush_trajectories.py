"""Render brush procedural trajectory datapoints to PNG (pyrender + trimesh)."""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_str_pipeline.sim_rollout.sample_flip_scenes import (
    PAN_RADIUS_M,
    PAN_WALL_HEIGHT_M,
    PAN_WALL_THICKNESS_M,
    PAN_XYZ,
)
from generative_str_pipeline.sim_rollout.sample_pour_scenes import POUR_PAN_XYZ
from generative_str_pipeline.sim_workspace import SIM_TABLE_SIZE_X_M, SIM_TABLE_SIZE_Y_M
from training.action_trajectory.dataset import WaypointTrajectorySample, load_waypoint_samples

MOVEMENT_CHOICES = ("stroke_sweep", "paint_dip", "paint_stroke", "scrub", "press")

COLOR_TABLE = (180, 140, 100, 255)
COLOR_SURFACE = (140, 160, 200, 80)
COLOR_PATH = (180, 180, 200, 255)
COLOR_CONTACT = (78, 205, 196, 255)
COLOR_NORMAL = (235, 64, 52, 255)
COLOR_SURFDIR = (245, 190, 60, 255)
COLOR_MATERIAL = (120, 230, 60, 255)
COLOR_DESTINATION = (247, 151, 30, 255)
COLOR_TOOL_BODY = (240, 240, 240, 255)
COLOR_PAN_WALL = (60, 60, 66, 130)
COLOR_PAN_RIM = (20, 20, 24, 255)

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
PANEL_HEIGHT = 240
OUTPUT_HEIGHT = RENDER_HEIGHT + PANEL_HEIGHT

FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


LEGEND_ENTRIES: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("waypoint normal", COLOR_NORMAL),
    ("surface_dir", COLOR_SURFDIR),
    ("waypoint contact", COLOR_CONTACT),
    ("trajectory path", COLOR_PATH),
    ("tool home pose", COLOR_TOOL_BODY),
    ("material (+ normal)", COLOR_MATERIAL),
    ("destination (+ normal)", COLOR_DESTINATION),
    ("destination surface", COLOR_SURFACE),
)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _make_arrow(
    origin: np.ndarray,
    direction: np.ndarray,
    color_rgba: tuple[int, int, int, int],
    length: float = 0.08,
    shaft_radius: float = 0.004,
    tip_at_origin: bool = False,
    tip_gap: float = 0.0,
    base_gap: float = 0.0,
) -> trimesh.Trimesh | None:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return None
    direction = direction / n

    shaft_h = length * 0.75
    head_h = length * 0.25
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_h, sections=16)
    head = trimesh.creation.cone(radius=shaft_radius * 2.5, height=head_h, sections=16)
    shaft.apply_translation([0, 0, shaft_h / 2.0])
    head.apply_translation([0, 0, shaft_h + head_h / 2.0])
    mesh = trimesh.util.concatenate([shaft, head])

    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(direction, z):
        R = np.eye(4)
    elif np.allclose(direction, -z):
        R = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = np.cross(z, direction)
        axis /= max(np.linalg.norm(axis), 1e-9)
        angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
        R = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(R)
    if tip_at_origin:
        mesh.apply_translation(-direction * (length + tip_gap))
    elif base_gap > 0.0:
        mesh.apply_translation(direction * base_gap)
    mesh.apply_translation(origin)
    mesh.visual.face_colors = list(color_rgba)
    return mesh


def pan_viz_settings(movement_token: str) -> tuple[bool, tuple[float, float]]:
    """Return (draw_pan, pan_center_xy) for a task movement token.

    Fixed-center fallback used only when no material-relative center is available
    (e.g. legacy shards). The pan is now placed material-relative; see
    ``pan_center_for_sample``.
    """
    token = str(movement_token).strip().lower()
    if token == "pour":
        return True, (float(POUR_PAN_XYZ[0]), float(POUR_PAN_XYZ[1]))
    if token == "flip":
        return True, (float(PAN_XYZ[0]), float(PAN_XYZ[1]))
    return False, (float(PAN_XYZ[0]), float(PAN_XYZ[1]))


def pan_center_for_sample(sample: Any) -> tuple[bool, tuple[float, float]]:
    """Return (draw_pan, pan_center_xy) for a sample, material-relative.

    The pan is drawn only for the flip/pour tasks. Its center tracks the sample:
    the per-scene ``pan_center_xy_world`` (object near the rim) when present, else
    the sample's material xy, else the legacy fixed center.
    """
    token = str(getattr(sample, "movement_token", "")).strip().lower()
    draw_pan = token in ("flip", "pour")
    center = getattr(sample, "pan_center_xy_world", None)
    if center is not None:
        return draw_pan, (float(center[0]), float(center[1]))
    mat = getattr(sample, "material_xyz_world", None)
    if mat is not None:
        return draw_pan, (float(mat[0]), float(mat[1]))
    _, fixed = pan_viz_settings(token)
    return draw_pan, fixed


def _make_pan_wall(
    table_z: float,
    center_xy: tuple[float, float] = (float(PAN_XYZ[0]), float(PAN_XYZ[1])),
) -> trimesh.Trimesh:
    """Handleless pan wall ring sitting on the table (open top, no bottom)."""
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
    return wall


def _make_pan_rim(
    table_z: float,
    center_xy: tuple[float, float] = (float(PAN_XYZ[0]), float(PAN_XYZ[1])),
) -> trimesh.Trimesh:
    """Bright circle marking the pan rim radius at the top of the wall."""
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
    rim.visual.face_colors = list(COLOR_PAN_RIM)
    return rim


def _make_table(table_z: float, color: tuple[int, int, int, int] = COLOR_TABLE) -> trimesh.Trimesh:
    # Match SimToolReal table_narrow.urdf footprint (0.475 x 0.4 m).
    t = trimesh.creation.box(extents=[SIM_TABLE_SIZE_X_M, SIM_TABLE_SIZE_Y_M, 0.02])
    t.apply_translation([0.0, 0.0, table_z - 0.01])
    t.visual.face_colors = list(color)
    return t


def _make_destination_surface(
    center: np.ndarray,
    normal: np.ndarray,
    color: tuple[int, int, int, int] = COLOR_SURFACE,
    size: float = 0.40,
    thickness: float = 0.005,
) -> trimesh.Trimesh:
    panel = trimesh.creation.box(extents=[size, size, thickness])
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(n, z):
        R = np.eye(4)
    elif np.allclose(n, -z):
        R = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = np.cross(z, n)
        axis /= max(np.linalg.norm(axis), 1e-9)
        angle = float(np.arccos(np.clip(np.dot(z, n), -1.0, 1.0)))
        R = trimesh.transformations.rotation_matrix(angle, axis)
    panel.apply_transform(R)
    panel.apply_translation(center)
    panel.visual.face_colors = list(color)
    return panel


def _make_trajectory_tube(
    contacts: list[np.ndarray],
    color: tuple[int, int, int, int] = COLOR_PATH,
    radius: float = 0.0025,
    segment_colors: list[tuple[int, int, int, int]] | None = None,
) -> trimesh.Trimesh | None:
    """Tube along contacts. Optional per-segment colors (len = len(contacts)-1)."""
    parts: list[trimesh.Trimesh] = []
    for i in range(len(contacts) - 1):
        a = np.asarray(contacts[i], dtype=np.float64)
        b = np.asarray(contacts[i + 1], dtype=np.float64)
        if np.linalg.norm(b - a) < 1e-6:
            continue
        seg = trimesh.creation.cylinder(radius=radius, segment=[a, b], sections=12)
        seg_color = (
            segment_colors[i]
            if segment_colors is not None and i < len(segment_colors)
            else color
        )
        seg.visual.face_colors = list(seg_color)
        parts.append(seg)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _viridis_rgba(t: float, alpha: int = 255) -> tuple[int, int, int, int]:
    """Approximate matplotlib viridis at t in [0, 1] (no matplotlib dependency)."""
    t = float(np.clip(t, 0.0, 1.0))
    # Key stops: purple -> blue -> green -> yellow
    stops = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float64,
    )
    x = t * (len(stops) - 1)
    i0 = int(np.floor(x))
    i1 = min(i0 + 1, len(stops) - 1)
    f = x - i0
    rgb = (1.0 - f) * stops[i0] + f * stops[i1]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha))


def _make_sphere(
    center: np.ndarray,
    radius: float,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    s = trimesh.creation.icosphere(radius=radius, subdivisions=2)
    s.apply_translation(np.asarray(center, dtype=np.float64))
    s.visual.face_colors = list(color)
    return s


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [x, y, z, w] -> rotation matrix (3x3)."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _make_oriented_box(
    center: np.ndarray,
    size: np.ndarray,
    quat_xyzw: np.ndarray,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    """Axis-aligned box in local frame, posed by quaternion at ``center``."""
    extents = np.asarray(size, dtype=np.float64).reshape(3)
    box = trimesh.creation.box(extents=extents)
    R = _quat_xyzw_to_matrix(quat_xyzw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    box.apply_transform(T)
    box.visual.face_colors = list(color)
    return box


def _make_material_mesh(
    center: np.ndarray,
    color: tuple[int, int, int, int],
    *,
    quat_xyzw: np.ndarray | None = None,
    size: np.ndarray | None = None,
    sphere_radius: float = 0.024,
) -> trimesh.Trimesh:
    """Render material as an oriented box when pose/size are given, else a sphere."""
    if quat_xyzw is not None and size is not None:
        return _make_oriented_box(center, size, quat_xyzw, color)
    return _make_sphere(center, sphere_radius, color)


def _add_mesh(scene: pyrender.Scene, mesh: trimesh.Trimesh | None) -> None:
    if mesh is None:
        return
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))


def _add_translucent_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh | None,
    color_rgba: tuple[int, int, int, int],
) -> None:
    if mesh is None:
        return
    r, g, b, a = color_rgba
    material = pyrender.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        baseColorFactor=[r / 255.0, g / 255.0, b / 255.0, a / 255.0],
        metallicFactor=0.0,
        roughnessFactor=0.8,
        doubleSided=True,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material))


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-9:
        return np.eye(4)
    f = f / f_norm
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-9:
        return np.eye(4)
    s = s / s_norm
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return np.linalg.inv(M)


def build_scene(
    sample: WaypointTrajectorySample,
    movement_token: str,
    *,
    draw_destination_surface: bool = True,
    draw_pan: bool | None = None,
    pan_center_xy: tuple[float, float] | None = None,
) -> pyrender.Scene:
    if draw_pan is None or pan_center_xy is None:
        auto_draw, auto_center = pan_center_for_sample(sample)
        if draw_pan is None:
            draw_pan = auto_draw
        if pan_center_xy is None:
            pan_center_xy = auto_center
    scene = pyrender.Scene(
        bg_color=[26, 30, 46, 255],
        ambient_light=[0.4, 0.4, 0.4],
    )

    table_z = float(sample.table_xyz_world[2])
    _add_mesh(scene, _make_table(table_z))
    if draw_pan:
        _add_translucent_mesh(scene, _make_pan_wall(table_z, pan_center_xy), COLOR_PAN_WALL)
        _add_mesh(scene, _make_pan_rim(table_z, pan_center_xy))

    if (
        draw_destination_surface
        and sample.has_destination
        and sample.destination_xyz_world is not None
    ):
        dest_normal = sample.destination_normal
        if dest_normal is None:
            dest_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _add_translucent_mesh(
            scene,
            _make_destination_surface(
                np.asarray(sample.destination_xyz_world, dtype=np.float64),
                np.asarray(dest_normal, dtype=np.float64),
            ),
            COLOR_SURFACE,
        )

    wp_contacts = [
        np.asarray(sample.waypoints[i, 0:3], dtype=np.float64) for i in range(6)
    ]
    tube_path = [np.asarray(sample.tool_contact_xyz_world, dtype=np.float64)] + wp_contacts
    _add_mesh(scene, _make_trajectory_tube(tube_path))

    for i in range(6):
        contact = wp_contacts[i]
        normal = np.asarray(sample.waypoints[i, 3:6], dtype=np.float64)
        surface_dir = np.asarray(sample.waypoints[i, 6:9], dtype=np.float64)
        _add_mesh(scene, _make_sphere(contact, 0.013, COLOR_CONTACT))
        _add_mesh(
            scene,
            _make_arrow(contact, normal, COLOR_NORMAL, length=0.07, base_gap=0.013),
        )
        _add_mesh(
            scene,
            _make_arrow(contact, surface_dir, COLOR_SURFDIR, length=0.05, base_gap=0.013),
        )

    if sample.has_material and sample.material_xyz_world is not None:
        mat_xyz = np.asarray(sample.material_xyz_world, dtype=np.float64)
        mat_normal = sample.material_normal
        if mat_normal is None:
            mat_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _add_mesh(scene, _make_sphere(mat_xyz, 0.024, COLOR_MATERIAL))
        _add_mesh(
            scene,
            _make_arrow(
                mat_xyz,
                np.asarray(mat_normal, dtype=np.float64),
                COLOR_MATERIAL,
                length=0.07,
                base_gap=0.024,
            ),
        )

    if sample.has_destination and sample.destination_xyz_world is not None:
        dest_xyz = np.asarray(sample.destination_xyz_world, dtype=np.float64)
        dest_normal = sample.destination_normal
        if dest_normal is None:
            dest_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _add_mesh(scene, _make_sphere(dest_xyz, 0.024, COLOR_DESTINATION))
        _add_mesh(
            scene,
            _make_arrow(
                dest_xyz,
                np.asarray(dest_normal, dtype=np.float64),
                COLOR_DESTINATION,
                length=0.07,
                base_gap=0.024,
            ),
        )

    tool_xyz = np.asarray(sample.tool_contact_xyz_world, dtype=np.float64)
    tool_normal = np.asarray(sample.tool_current_normal, dtype=np.float64)
    tool_surface_dir = np.asarray(sample.tool_current_surface_dir, dtype=np.float64)
    _add_mesh(scene, _make_sphere(tool_xyz, 0.020, COLOR_TOOL_BODY))
    _add_mesh(
        scene,
        _make_arrow(tool_xyz, tool_normal, COLOR_NORMAL, length=0.06, base_gap=0.020),
    )
    _add_mesh(
        scene,
        _make_arrow(tool_xyz, tool_surface_dir, COLOR_SURFDIR, length=0.04, base_gap=0.020),
    )

    # Match vec_rollout IsaacGym camera (target + offset).
    target = np.array([0.0, 0.0, 0.62], dtype=np.float64)
    eye = target + np.array([0.35, -0.55, 0.30], dtype=np.float64)
    cam_pose = _look_at(eye, target)

    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(45.0),
        aspectRatio=float(RENDER_WIDTH) / float(RENDER_HEIGHT),
    )
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    fill_pose = _look_at(target + np.array([0.0, 0.0, 1.5]), target)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill_pose)

    return scene


def build_rollout_scene(
    sample: WaypointTrajectorySample,
    movement_token: str,
    generations: list[dict[str, Any]],
    *,
    draw_destination_surface: bool = True,
) -> pyrender.Scene:
    """Multi-generation closed-loop rollout scene (viridis time colors).

    Each generation dict has:
      - ``material_xyz``: object placement at that replan step [3]
      - ``path_contacts``: executed brush contact points [N, 3] or list of [3]
    """
    _ = movement_token
    scene = pyrender.Scene(
        bg_color=[26, 30, 46, 255],
        ambient_light=[0.4, 0.4, 0.4],
    )
    table_z = float(sample.table_xyz_world[2])
    _add_mesh(scene, _make_table(table_z))

    if (
        draw_destination_surface
        and sample.has_destination
        and sample.destination_xyz_world is not None
    ):
        dest_normal = sample.destination_normal
        if dest_normal is None:
            dest_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        _add_translucent_mesh(
            scene,
            _make_destination_surface(
                np.asarray(sample.destination_xyz_world, dtype=np.float64),
                np.asarray(dest_normal, dtype=np.float64),
            ),
            COLOR_SURFACE,
        )

    n_gen = max(1, len(generations))
    for gi, gen in enumerate(generations):
        t = float(gi) / float(max(n_gen - 1, 1))
        color = _viridis_rgba(t)
        mat = np.asarray(gen["material_xyz"], dtype=np.float64).reshape(3)
        _add_mesh(scene, _make_sphere(mat, 0.024, color))

        path = gen.get("path_contacts") or []
        contacts = [np.asarray(c, dtype=np.float64).reshape(3) for c in path]
        if len(contacts) >= 2:
            seg_colors = [color] * (len(contacts) - 1)
            _add_mesh(
                scene,
                _make_trajectory_tube(contacts, segment_colors=seg_colors),
            )

    # Draw the per-waypoint normal + surface_dir arrows for the current (latest)
    # generation's executed chunk, matching the pretrain trajectory viz so the
    # orientation of each executed waypoint is visible.
    if generations:
        cur = generations[-1]
        cur_contacts = cur.get("path_contacts") or []
        cur_normals = cur.get("path_normals") or []
        cur_surface_dirs = cur.get("path_surface_dirs") or []
        for wi, c in enumerate(cur_contacts):
            contact = np.asarray(c, dtype=np.float64).reshape(3)
            _add_mesh(scene, _make_sphere(contact, 0.013, COLOR_CONTACT))
            if wi < len(cur_normals):
                normal = np.asarray(cur_normals[wi], dtype=np.float64).reshape(3)
                _add_mesh(
                    scene,
                    _make_arrow(contact, normal, COLOR_NORMAL, length=0.07, base_gap=0.013),
                )
            if wi < len(cur_surface_dirs):
                surface_dir = np.asarray(cur_surface_dirs[wi], dtype=np.float64).reshape(3)
                _add_mesh(
                    scene,
                    _make_arrow(contact, surface_dir, COLOR_SURFDIR, length=0.05, base_gap=0.013),
                )

    if sample.has_destination and sample.destination_xyz_world is not None:
        dest_xyz = np.asarray(sample.destination_xyz_world, dtype=np.float64)
        _add_mesh(scene, _make_sphere(dest_xyz, 0.024, COLOR_DESTINATION))

    tool_xyz = np.asarray(sample.tool_contact_xyz_world, dtype=np.float64)
    _add_mesh(scene, _make_sphere(tool_xyz, 0.020, COLOR_TOOL_BODY))

    target = np.array([0.0, 0.0, 0.62], dtype=np.float64)
    eye = target + np.array([0.35, -0.55, 0.30], dtype=np.float64)
    cam_pose = _look_at(eye, target)
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(45.0),
        aspectRatio=float(RENDER_WIDTH) / float(RENDER_HEIGHT),
    )
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=cam_pose)
    fill_pose = _look_at(target + np.array([0.0, 0.0, 1.5]), target)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill_pose)
    return scene


def render_rollout_panel(
    sample: WaypointTrajectorySample,
    movement_token: str,
    generations: list[dict],
    *,
    draw_destination_surface: bool = True,
) -> Image.Image:
    """Render closed-loop rollout with per-generation viridis path + material balls."""
    scene = build_rollout_scene(
        sample,
        movement_token,
        generations,
        draw_destination_surface=draw_destination_surface,
    )
    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH,
        viewport_height=RENDER_HEIGHT,
    )
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()

    img = Image.fromarray(color)
    panel = _make_text_panel(
        sample, movement_token, draw_destination_surface=draw_destination_surface
    )
    full = Image.new("RGB", (RENDER_WIDTH, OUTPUT_HEIGHT), color=(20, 24, 40))
    full.paste(img, (0, 0))
    full.paste(panel, (0, RENDER_HEIGHT))
    return full


def render_datapoint(
    sample: WaypointTrajectorySample,
    movement_token: str,
    *,
    draw_destination_surface: bool = True,
    draw_pan: bool | None = None,
    pan_center_xy: tuple[float, float] | None = None,
) -> Image.Image:
    scene = build_scene(
        sample,
        movement_token,
        draw_destination_surface=draw_destination_surface,
        draw_pan=draw_pan,
        pan_center_xy=pan_center_xy,
    )
    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH,
        viewport_height=RENDER_HEIGHT,
    )
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()

    img = Image.fromarray(color)

    panel = _make_text_panel(
        sample, movement_token, draw_destination_surface=draw_destination_surface
    )

    full = Image.new("RGB", (RENDER_WIDTH, OUTPUT_HEIGHT), color=(20, 24, 40))
    full.paste(img, (0, 0))
    full.paste(panel, (0, RENDER_HEIGHT))
    return full


def _make_text_panel(
    sample: WaypointTrajectorySample,
    movement_token: str,
    *,
    draw_destination_surface: bool = True,
) -> Image.Image:
    panel = Image.new("RGB", (RENDER_WIDTH, PANEL_HEIGHT), color=(20, 24, 40))
    draw = ImageDraw.Draw(panel)

    title_font = _load_font(FONT_PATH_BOLD, 26)
    body_font = _load_font(FONT_PATH_REGULAR, 20)
    legend_title_font = _load_font(FONT_PATH_BOLD, 18)
    legend_font = _load_font(FONT_PATH_REGULAR, 16)

    text_x = 24
    draw.text(
        (text_x, 16),
        f"{movement_token}   dp {sample.datapoint_index}",
        fill=(220, 220, 240),
        font=title_font,
    )
    draw.text(
        (text_x, 60),
        f'"{sample.instruction}"',
        fill=(190, 190, 210),
        font=body_font,
    )
    has_m = "material" if sample.has_material else "(no material)"
    has_d = "destination" if sample.has_destination else "(no destination)"
    draw.text(
        (text_x, 96),
        f"{has_m}   {has_d}",
        fill=(160, 160, 180),
        font=body_font,
    )

    legend_x = 760
    legend_y = 12
    draw.text(
        (legend_x, legend_y),
        "Legend",
        fill=(220, 220, 240),
        font=legend_title_font,
    )

    col_w = 250
    row_h = 26
    swatch_size = 16
    rows_per_col = 4
    entry_x0 = legend_x
    entry_y0 = legend_y + 30

    legend_entries = [
        e
        for e in LEGEND_ENTRIES
        if draw_destination_surface or e[0] != "destination surface"
    ]
    for idx, (label, color) in enumerate(legend_entries):
        col = idx // rows_per_col
        row = idx % rows_per_col
        ex = entry_x0 + col * col_w
        ey = entry_y0 + row * row_h
        draw.rectangle(
            [ex, ey, ex + swatch_size, ey + swatch_size],
            fill=color[:3],
            outline=(80, 80, 100),
        )
        draw.text(
            (ex + swatch_size + 10, ey - 2),
            label,
            fill=(200, 200, 220),
            font=legend_font,
        )

    return panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize brush procedural trajectory datapoints to PNG."
    )
    parser.add_argument("--shard_path", type=str, required=True, help="Path to *_shard.json")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/verification/dataset_0008_brush_procedural_viz",
        help="Directory for output PNG files",
    )
    parser.add_argument(
        "--movement_token",
        type=str,
        default=None,
        choices=list(MOVEMENT_CHOICES),
        help="Optional filter to one movement type",
    )
    parser.add_argument(
        "--max_datapoints",
        type=int,
        default=0,
        help="Max datapoints to render after filter (0 = all)",
    )
    args = parser.parse_args()

    shard_path = _resolve_path(Path(args.shard_path))
    out_dir = _resolve_path(Path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_waypoint_samples(shard_path)
    raw = json.loads(shard_path.read_text(encoding="utf-8"))
    movement_tokens = [str(dp["movement_token"]) for dp in raw.get("datapoints", [])]
    if len(movement_tokens) != len(samples):
        raise RuntimeError(
            f"movement_token count {len(movement_tokens)} != sample count {len(samples)}"
        )

    filter_token = str(args.movement_token).strip() if args.movement_token else ""
    pairs: list[tuple[WaypointTrajectorySample, str]] = []
    for sample, mtoken in zip(samples, movement_tokens):
        if filter_token and mtoken != filter_token:
            continue
        pairs.append((sample, mtoken))

    if int(args.max_datapoints) > 0:
        pairs = pairs[: int(args.max_datapoints)]

    if not pairs:
        print("No datapoints matched filters.")
        return

    for sample, mtoken in pairs:
        img = render_datapoint(sample, mtoken)
        out_path = out_dir / f"viz_{mtoken}_{sample.datapoint_index:06d}.png"
        img.save(out_path)
        print(f"Wrote {out_path}")

    print(f"Done: {len(pairs)} figure(s) in {out_dir}")


if __name__ == "__main__":
    main()
