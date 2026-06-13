"""Annotate per-object control frames (control point, surface dir, normal) via Viser.

Pick the object from the "Object mesh" dropdown (every DexToolBench mesh is
listed), click four corners on the contact face (front L/R, back L/R) on the
mesh in the browser, regularize to a rectangle, compute the contact frame,
verify arrows, and save JSON.

Usage (after pip install -e closed_loop/):

    python -m closed_loop.tools.annotate_object_control_point \\
        --object-name blue_brush --port 8080

``--object-name`` just sets the initially selected mesh; you can switch to any
other DexToolBench object live via the dropdown.

Port-forward on AWS: ssh -L 8080:localhost:8080 ...
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Python 3.8 compatibility shim: viser>=1.0 serves its frontend via a handler
# that calls ``Path.is_relative_to`` (added in Python 3.9). Without this, every
# HTTP request 500s with "Failed to open a WebSocket connection" and the client
# never loads. Backport the method onto PurePath so the static server works.
if not hasattr(pathlib.PurePath, "is_relative_to"):

    def _is_relative_to(self: pathlib.PurePath, *other: Any) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    pathlib.PurePath.is_relative_to = _is_relative_to  # type: ignore[attr-defined]

import numpy as np
import trimesh
import tyro
import viser

TOOL_VERSION = "0.1.0"

CORNER_ORDER = ("front_left", "front_right", "back_left", "back_right")
CORNER_PROMPTS = {
    "front_left": "Click **front edge LEFT** (leading/sweep edge, left corner).",
    "front_right": "Click **front edge RIGHT**.",
    "back_left": "Click **back edge LEFT**.",
    "back_right": "Click **back edge RIGHT**.",
}

# Viser colors (uint8 RGB)
COLOR_CLICK = (255, 80, 80)
COLOR_RECT = (80, 200, 255)
COLOR_SURFACE = (50, 120, 255)
COLOR_NORMAL = (255, 60, 60)


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def simtoolreal_root() -> Path:
    """Sibling SimToolReal checkout (deployable unit includes simtoolreal/)."""
    return package_root().parent.parent / "simtoolreal"


DEXTOOLBENCH_CATEGORIES = (
    "brush",
    "spatula",
    "hammer",
    "screwdriver",
    "eraser",
    "marker",
)


def dextoolbench_root() -> Path:
    return simtoolreal_root() / "assets" / "urdf" / "dextoolbench"


def default_obj_path(object_name: str) -> Path:
    """Resolve ``{category}/{object_name}/{object_name}.obj`` under DexToolBench."""
    base = dextoolbench_root()
    for category in DEXTOOLBENCH_CATEGORIES:
        candidate = base / category / object_name / f"{object_name}.obj"
        if candidate.is_file():
            return candidate
    return base / "brush" / object_name / f"{object_name}.obj"


def list_dextoolbench_objects() -> Dict[str, Path]:
    """Return ``{object_name: obj_path}`` for every ``{category}/{name}/{name}.obj``."""
    base = dextoolbench_root()
    out: Dict[str, Path] = {}
    if not base.is_dir():
        return out
    for obj_path in sorted(base.glob("*/*/*.obj")):
        if obj_path.stem == obj_path.parent.name:
            out[obj_path.stem] = obj_path.resolve()
    return out


def default_output_dir() -> Path:
    return package_root() / "assets" / "control_frames"


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return (v / n).astype(np.float64)


def _project_to_plane(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project Nx3 points onto plane through origin with unit normal."""
    n = _unit(normal)
    d = points - origin[None, :]
    return points - np.outer(d @ n, n)


def _fit_plane_normal(corners: np.ndarray) -> np.ndarray:
    """Unit normal from 4 corners (SVD on centered points)."""
    centered = corners - corners.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return _unit(normal)


def regularize_rectangle(
    front_left: np.ndarray,
    front_right: np.ndarray,
    back_left: np.ndarray,
    back_right: np.ndarray,
    mesh_centroid: np.ndarray,
    flip_normal: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Snap clicks to a clean rectangle; return corners dict + control_point, surface_dir, normal."""
    raw = np.stack([front_left, front_right, back_left, back_right], axis=0).astype(np.float64)
    plane_n = _fit_plane_normal(raw)
    # Orient normal away from mesh interior (toward contact / table side).
    to_centroid = mesh_centroid - raw.mean(axis=0)
    if float(np.dot(plane_n, to_centroid)) > 0.0:
        plane_n = -plane_n
    if flip_normal:
        plane_n = -plane_n

    origin = raw.mean(axis=0)
    proj = _project_to_plane(raw, origin, plane_n)

    fl, fr, bl, br = proj[0], proj[1], proj[2], proj[3]
    front_mid = 0.5 * (fl + fr)
    back_mid = 0.5 * (bl + br)

    length_vec = back_mid - front_mid
    length = float(np.linalg.norm(length_vec))
    if length < 1e-6:
        raise ValueError("Front and back edges are too close; re-click corners.")
    length_axis = _unit(length_vec)

    width_raw = fr - fl
    width_axis = width_raw - np.dot(width_raw, length_axis) * length_axis
    width_norm = float(np.linalg.norm(width_axis))
    if width_norm < 1e-6:
        # Fallback: in-plane perpendicular to length
        width_axis = np.cross(plane_n, length_axis)
        width_axis = _unit(width_axis)
    else:
        width_axis = _unit(width_axis)

    half_width = 0.5 * (
        abs(float(np.dot(fl - front_mid, width_axis)))
        + abs(float(np.dot(fr - front_mid, width_axis)))
    )
    half_width = max(half_width, 1e-4)

    front_mid_r = front_mid.copy()
    back_mid_r = front_mid_r + length_axis * length

    fl_r = front_mid_r - width_axis * half_width
    fr_r = front_mid_r + width_axis * half_width
    bl_r = back_mid_r - width_axis * half_width
    br_r = back_mid_r + width_axis * half_width

    corners_rect = {
        "front_left": fl_r,
        "front_right": fr_r,
        "back_left": bl_r,
        "back_right": br_r,
    }

    control_point = front_mid_r
    surface_dir = length_axis  # front -> back, in plane
    normal = plane_n

    return corners_rect, control_point, surface_dir, normal


def build_T_obj_from_contact(
    control_point: np.ndarray,
    surface_dir: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """4x4 row-major: columns [surface_dir, normal x surface_dir, normal], translation = control_point."""
    s = _unit(surface_dir)
    n = _unit(normal)
    y = _unit(np.cross(n, s))
    R = np.stack([s, y, n], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = control_point
    return T


def _vec3_list(v: np.ndarray) -> List[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _matrix_list(T: np.ndarray) -> List[List[float]]:
    return T.astype(np.float64).tolist()


@dataclass
class ComputedFrame:
    corners_clicked: Dict[str, np.ndarray]
    corners_rectangle: Dict[str, np.ndarray]
    control_point: np.ndarray
    surface_dir: np.ndarray
    normal: np.ndarray
    T_obj_from_contact: np.ndarray


@dataclass
class AnnotateArgs:
    object_name: str = "blue_brush"
    """Object name (used for default mesh path and output JSON filename)."""

    obj_path: Optional[Path] = None
    """Path to .obj mesh; default: simtoolreal/.../brush/{object_name}/{object_name}.obj"""

    port: int = 8080
    """Viser server port (bind 0.0.0.0 for port-forward)."""

    output_dir: Optional[Path] = None
    """Directory for saved JSON; default: generative_str_pipeline/assets/object_control_points"""

    arrow_scale: float = 0.08
    """Length of verification arrows (meters in mesh units)."""


class ObjectControlPointAnnotator:
    def __init__(self, args: AnnotateArgs) -> None:
        self.args = args
        self.objects = list_dextoolbench_objects()

        self.object_name = args.object_name
        if args.obj_path is not None:
            obj_path = Path(args.obj_path)
            # Register an explicit path so it shows up in the dropdown too.
            self.objects.setdefault(self.object_name, obj_path.resolve())
        else:
            obj_path = self.objects.get(self.object_name) or default_obj_path(self.object_name)
        if not obj_path.is_file():
            raise FileNotFoundError(f"Mesh not found: {obj_path}")

        self.obj_path = obj_path.resolve()
        self.mesh: trimesh.Trimesh = self._load_mesh(self.obj_path)
        self.mesh_centroid = np.asarray(self.mesh.centroid, dtype=np.float64)

        self.output_dir = (args.output_dir or default_output_dir()).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clicks: Dict[str, np.ndarray] = {}
        self.flip_normal = False
        self.computed: Optional[ComputedFrame] = None
        self._overlay_handles: List[Any] = []

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self._setup_scene()
        self._setup_gui()
        self._enable_mesh_click()
        self._refresh_prompt()

    @staticmethod
    def _load_mesh(obj_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(obj_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")
        return mesh

    def _setup_scene(self) -> None:
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0.25, -0.35, 0.12)
            client.camera.look_at = (0.0, 0.0, 0.0)

        self.server.scene.add_frame("/object", axes_length=0.05, axes_radius=0.002)
        self.mesh_handle = self.server.scene.add_mesh_simple(
            name="/object/mesh",
            vertices=np.asarray(self.mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.mesh.faces, dtype=np.int32),
            color=(100, 160, 220),
        )

        self._update_grid()

        print(f"Viser: http://0.0.0.0:{self.args.port}  (port-forward to your machine)")

    def _update_grid(self) -> None:
        min_z = float(self.mesh.bounds[0, 2]) if self.mesh.bounds is not None else 0.0
        self.server.scene.add_grid(
            "/grid",
            width=0.5,
            height=0.5,
            position=(0.0, 0.0, min_z - 0.002),
        )

    def _remove_annotation_overlays(self) -> None:
        """Remove all click markers and computed-frame overlays from the scene."""
        for key in CORNER_ORDER:
            for name in (f"/clicks/{key}", f"/clicks/{key}_label"):
                try:
                    self.server.scene.remove_by_name(name)
                except Exception:
                    pass
        for name in (
            "/frame/rect_0",
            "/frame/rect_1",
            "/frame/rect_2",
            "/frame/rect_3",
            "/frame/control_point",
            "/frame/surface_dir",
            "/frame/normal",
        ):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass

    def _select_object(self, object_name: str) -> None:
        """Swap the displayed mesh + reset annotation state for a new object."""
        obj_path = self.objects.get(object_name) or default_obj_path(object_name)
        if not obj_path.is_file():
            self.status_md.content = f"**Error:** Mesh not found for `{object_name}`."
            return
        try:
            new_mesh = self._load_mesh(obj_path)
        except (OSError, TypeError, ValueError) as exc:
            self.status_md.content = f"**Error loading `{object_name}`:** {exc}"
            return

        self.object_name = object_name
        self.obj_path = obj_path.resolve()
        self.mesh = new_mesh
        self.mesh_centroid = np.asarray(self.mesh.centroid, dtype=np.float64)

        self._remove_annotation_overlays()
        self.clicks.clear()
        self.flip_normal = False
        self.computed = None

        self.mesh_handle = self.server.scene.add_mesh_simple(
            name="/object/mesh",
            vertices=np.asarray(self.mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.mesh.faces, dtype=np.int32),
            color=(100, 160, 220),
        )
        self._enable_mesh_click()
        self._update_grid()
        self.object_md.content = f"**Object:** `{self.object_name}`"
        self._refresh_prompt()

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown("# Object control-point annotator")

        options = sorted(self.objects.keys())
        if self.object_name not in options:
            options = [self.object_name] + options
        self.object_dropdown = self.server.gui.add_dropdown(
            "Object mesh",
            options=tuple(options),
            initial_value=self.object_name,
        )
        self.object_dropdown.on_update(self._on_object_change)

        self.object_md = self.server.gui.add_markdown(f"**Object:** `{self.object_name}`")
        self.prompt_md = self.server.gui.add_markdown("*Loading…*")
        self.status_md = self.server.gui.add_markdown("**Status:** Click corners on the mesh.")

        self.server.gui.add_button("Undo last").on_click(lambda _: self._undo())
        self.server.gui.add_button("Clear").on_click(lambda _: self._clear())
        self.server.gui.add_button("Flip normal").on_click(lambda _: self._flip_normal())
        self.server.gui.add_button("Compute frame").on_click(lambda _: self._compute_frame())
        self.server.gui.add_button("Save JSON").on_click(lambda _: self._save())

    def _on_object_change(self, _event: Any) -> None:
        self._select_object(str(self.object_dropdown.value))

    def _next_corner_key(self) -> Optional[str]:
        for key in CORNER_ORDER:
            if key not in self.clicks:
                return key
        return None

    def _refresh_prompt(self) -> None:
        nxt = self._next_corner_key()
        if nxt is None:
            self.prompt_md.content = (
                "**All 4 corners set.** Click **Compute frame**, verify arrows, then **Save JSON**."
            )
        else:
            idx = CORNER_ORDER.index(nxt) + 1
            self.prompt_md.content = (
                f"**Step {idx}/4:** {CORNER_PROMPTS[nxt]}"
            )
        n = len(self.clicks)
        extra = ""
        if self.computed is not None:
            cp = self.computed.control_point
            extra = (
                f"\n\n**Frame:** control=({cp[0]:.4f}, {cp[1]:.4f}, {cp[2]:.4f})"
            )
        self.status_md.content = f"**Corners:** {n}/4{extra}"

    def _enable_mesh_click(self) -> None:
        @self.mesh_handle.on_click
        def on_click(event: viser.ScenePointerEvent) -> None:
            nxt = self._next_corner_key()
            if nxt is None:
                return

            ray_origin = np.array(event.ray_origin, dtype=np.float64)
            ray_dir = np.array(event.ray_direction, dtype=np.float64)
            ray_dir = _unit(ray_dir)

            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_dir],
            )
            if len(locations) == 0:
                self.status_md.content = "**Status:** No mesh hit — click on the brush surface."
                return

            dists = np.linalg.norm(locations - ray_origin[None, :], axis=1)
            point = locations[int(np.argmin(dists))].astype(np.float64)

            self.clicks[nxt] = point
            idx = CORNER_ORDER.index(nxt) + 1
            self.server.scene.add_icosphere(
                f"/clicks/{nxt}",
                radius=0.003,
                color=COLOR_CLICK,
                position=tuple(point.astype(float)),
            )
            self.server.scene.add_label(
                f"/clicks/{nxt}_label",
                text=str(idx),
                position=tuple(point.astype(float)),
            )
            self.computed = None
            self._clear_overlays()
            self._refresh_prompt()

    def _undo(self) -> None:
        if not self.clicks:
            return
        last_key = [k for k in CORNER_ORDER if k in self.clicks][-1]
        del self.clicks[last_key]
        self.computed = None
        self._clear_overlays()
        self._refresh_prompt()

    def _clear(self) -> None:
        self.clicks.clear()
        self.computed = None
        self._clear_overlays()
        self._refresh_prompt()

    def _flip_normal(self) -> None:
        self.flip_normal = not self.flip_normal
        if len(self.clicks) == 4:
            self._compute_frame()

    def _clear_overlays(self) -> None:
        self._overlay_handles.clear()
        # Overlays use fixed paths we overwrite on next compute
        pass

    def _draw_segment(
        self,
        name: str,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[int, int, int],
    ) -> None:
        pts = np.array([[start, end]], dtype=np.float32)
        colors = np.array([[color, color]], dtype=np.uint8)
        self.server.scene.add_line_segments(name, points=pts, colors=colors, line_width=2.0)

    def _draw_arrow_at(
        self,
        name: str,
        origin: np.ndarray,
        direction: np.ndarray,
        color: Tuple[int, int, int],
        length: float,
    ) -> None:
        end = origin + _unit(direction) * length
        self._draw_segment(name, origin, end, color)

    def _visualize_frame(self, frame: ComputedFrame) -> None:
        scale = self.args.arrow_scale
        cp = frame.control_point
        cr = frame.corners_rectangle

        # Rectangle outline (closed loop)
        loop_keys = ["front_left", "front_right", "back_right", "back_left", "front_left"]
        for i in range(4):
            a = cr[loop_keys[i]]
            b = cr[loop_keys[i + 1]]
            self._draw_segment(f"/frame/rect_{i}", a, b, COLOR_RECT)

        # Control point marker
        self.server.scene.add_icosphere(
            "/frame/control_point",
            radius=0.004,
            color=(255, 220, 50),
            position=tuple(cp.astype(float)),
        )

        # Arrows from control point
        self._draw_arrow_at(
            "/frame/surface_dir",
            cp,
            frame.surface_dir,
            COLOR_SURFACE,
            scale,
        )
        self._draw_arrow_at(
            "/frame/normal",
            cp,
            frame.normal,
            COLOR_NORMAL,
            scale * 0.85,
        )

    def _compute_frame(self) -> None:
        if len(self.clicks) != 4:
            self.status_md.content = "**Status:** Need exactly 4 corners before computing."
            return
        try:
            corners_clicked = {k: self.clicks[k].copy() for k in CORNER_ORDER}
            corners_rect, control_point, surface_dir, normal = regularize_rectangle(
                corners_clicked["front_left"],
                corners_clicked["front_right"],
                corners_clicked["back_left"],
                corners_clicked["back_right"],
                self.mesh_centroid,
                self.flip_normal,
            )
            T = build_T_obj_from_contact(control_point, surface_dir, normal)
            self.computed = ComputedFrame(
                corners_clicked=corners_clicked,
                corners_rectangle=corners_rect,
                control_point=control_point,
                surface_dir=surface_dir,
                normal=normal,
                T_obj_from_contact=T,
            )
            self._visualize_frame(self.computed)
            self._refresh_prompt()
            self.status_md.content = (
                "**Status:** Frame computed. Check rectangle + arrows, then Save."
            )
        except Exception as exc:
            self.status_md.content = f"**Error:** {exc}"

    def _save(self) -> None:
        if self.computed is None:
            self._compute_frame()
        if self.computed is None:
            return

        out_path = self.output_dir / f"{self.object_name}.json"
        frame = self.computed
        payload: Dict[str, Any] = {
            "object_name": self.object_name,
            "obj_path": str(self.obj_path),
            "frame": "object_local",
            "corners_clicked": {k: _vec3_list(v) for k, v in frame.corners_clicked.items()},
            "corners_rectangle": {
                k: _vec3_list(v) for k, v in frame.corners_rectangle.items()
            },
            "control_point": _vec3_list(frame.control_point),
            "surface_dir": _vec3_list(frame.surface_dir),
            "normal": _vec3_list(frame.normal),
            "T_obj_from_contact": _matrix_list(frame.T_obj_from_contact),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool_version": TOOL_VERSION,
            "notes": {
                "control_point": "Center of front edge (tool placement).",
                "surface_dir": "Unit vector from front-edge center toward back-edge center.",
                "normal": "Unit plane normal at control point (flip_normal applied if used).",
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")
        self.status_md.content = f"**Saved:** `{out_path}`"

    def run(self) -> None:
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting.")


def main() -> None:
    args = tyro.cli(AnnotateArgs)
    annotator = ObjectControlPointAnnotator(args)
    annotator.run()


if __name__ == "__main__":
    main()
