"""Interactive brush VLA visualization (Viser, no ROS).

Drag tool / cube / goal gizmos, pick a tool, see the predicted plan, and play it
forward in static-plan or closed-loop-push mode.

Usage (after ``pip install -e closed_loop/`` with viz extra):

    python -m closed_loop.tools.viz_interactive --device cpu --port 8080

All-tasks joint pretrain (epoch 10) with flip tool + pan in sim:

    CUDA_VISIBLE_DEVICES="" python -m closed_loop.tools.viz_interactive \\
        --device cpu --port 8080 \\
        --model all_tasks_joint_pretrain_epoch10 \\
        --control-frame flat_spatula

The ``--model`` flag picks the initial registry checkpoint; you can also switch
models live via the **Model** dropdown in the GUI (it reloads the policy and
restricts the Tool dropdown to that model's control frames).
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Python 3.8 compatibility shim for viser static file server.
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

from closed_loop.analytic_push import GOAL_REGION_RADIUS_M, execute_chunk
from closed_loop.frames import contact_frame_from_root
from closed_loop.inference import BrushPolicy
from closed_loop.orientation import align_waypoints_to_reference
from closed_loop.paths import list_control_frames
from closed_loop.registry import default_model_key, list_models, resolve_model
from closed_loop.scene import SceneState
from closed_loop.viz import (
    COLOR_PAN_RIM,
    COLOR_PAN_WALL,
    default_instruction_for_control_frame,
    make_pan_meshes,
    pan_center_for_scene,
)
TABLE_Z = 0.53
ZERO_SHIFT = np.zeros(3, dtype=np.float64)

# First-frame poses in model/world frame (robot poses + Y shift 0.8 applied).
DEFAULT_TOOL_XYZ = np.array([-0.03649737, 0.03031215, 0.54811580], dtype=np.float64)
DEFAULT_TOOL_QUAT_XYZW = np.array(
    [0.01837961, -0.03682509, -0.21215851, 0.97636820], dtype=np.float64
)
DEFAULT_CUBE_XYZ = np.array([-0.22108552, 0.03569249, 0.54887887], dtype=np.float64)
DEFAULT_GOAL_XYZ = np.array([-0.36499680, -0.05614140, 0.51701116], dtype=np.float64)

COLOR_NORMAL = (255, 60, 60)
COLOR_SURFACE = (255, 220, 50)
COLOR_PATH = (80, 200, 255)
COLOR_CONTACT = (180, 180, 255)


def quat_xyzw_to_wxyz(q: np.ndarray) -> Tuple[float, float, float, float]:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def quat_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    wxyz = np.asarray(wxyz, dtype=np.float64).reshape(4)
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return (v / n).astype(np.float64)


def load_control_frame_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class VizArgs:
    device: str = "cuda"
    """Torch device for VLA inference."""

    port: int = 8080
    """Viser server port (bind 0.0.0.0 for port-forward)."""

    model: str | None = None
    """Registry key (e.g. ``all_tasks_grpo_iter20``). Uses package default ckpt if unset."""

    control_frame: str = "blue_brush"
    """Initial tool control frame name."""

    instruction: str | None = None
    """Override prompt. When unset, the basic per-task default for ``control_frame`` is used."""

    debounce_s: float = 0.35
    """Seconds to wait after gizmo drag before re-predicting."""

    arrow_scale: float = 0.04
    """Arrow length for normal / surface_dir (meters)."""


class InteractiveBrushViz:
    def __init__(self, args: VizArgs) -> None:
        self.args = args
        self.all_control_frames = list_control_frames()
        if not self.all_control_frames:
            raise FileNotFoundError("No control frames under assets/control_frames/")

        self.model_keys = list_models()
        self.current_model_key = args.model or default_model_key()
        if self.current_model_key not in self.model_keys:
            self.current_model_key = default_model_key()

        model_entry = resolve_model(self.current_model_key)
        self.control_frames = self._filter_control_frames(model_entry)

        initial_name = args.control_frame
        if initial_name not in self.control_frames:
            initial_name = model_entry.default_control_frame
        if initial_name not in self.control_frames:
            initial_name = next(iter(self.control_frames))

        instruction = (
            args.instruction
            if args.instruction
            else default_instruction_for_control_frame(initial_name)
        )
        self.policy = self._build_policy(model_entry, initial_name, instruction)
        self.current_tool_name = initial_name
        self.table_z = self.policy.table_z
        self._pan_center_xy: np.ndarray | None = None
        self._instruction_customized = bool(args.instruction)
        self._current_instruction = str(self.policy.instruction)

        self._predict_lock = threading.Lock()
        self._debounce_timer: Optional[threading.Timer] = None
        self._play_stop = threading.Event()
        self._play_pause = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._playing = False
        self._play_finished_status: Optional[str] = None

        self._waypoints: Optional[np.ndarray] = None
        self._object_poses: List[Tuple[np.ndarray, np.ndarray]] = []

        self._initial_tool_xyz = DEFAULT_TOOL_XYZ.copy()
        self._initial_tool_quat = DEFAULT_TOOL_QUAT_XYZW.copy()
        self._initial_cube_xyz = DEFAULT_CUBE_XYZ.copy()
        self._initial_goal_xyz = DEFAULT_GOAL_XYZ.copy()

        self._chunk_size = 5
        self._max_replans = 30
        self._play_fps = 8.0
        self._play_mode = "static"
        self._show_all_ghosts = False
        self._unflip_orientation = True

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self._setup_scene(initial_name)
        self._setup_gui()
        self._wire_gizmo_callbacks()
        self._predict_and_draw()

        print(f"Viser: http://0.0.0.0:{args.port}  (port-forward to your machine)")

    @staticmethod
    def _allowed_frames_for_model(model_entry) -> set:
        """Control frames a model is allowed to drive (empty set => no restriction)."""
        allowed: set = set()
        for frames in model_entry.metadata.get("task_control_frames", {}).values():
            allowed.update(str(f) for f in frames)
        allowed.update(str(f) for f in model_entry.metadata.get("control_frames", []))
        return allowed

    def _filter_control_frames(self, model_entry) -> Dict[str, Path]:
        allowed = self._allowed_frames_for_model(model_entry)
        if not allowed:
            return dict(self.all_control_frames)
        filtered = {k: v for k, v in self.all_control_frames.items() if k in allowed}
        return filtered or dict(self.all_control_frames)

    def _build_policy(self, model_entry, control_frame: str, instruction: str) -> BrushPolicy:
        return BrushPolicy(
            device=self.args.device,
            control_frame=control_frame,
            instruction=instruction,
            checkpoint_path=model_entry.checkpoint_path,
            normalization_stats_path=model_entry.normalization_stats_path,
        )

    def _setup_scene(self, tool_name: str) -> None:
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0.35, -0.55, 0.30)
            client.camera.look_at = (-0.15, 0.0, TABLE_Z)

        self.server.scene.add_grid(
            "/table",
            width=1.2,
            height=1.2,
            position=(0.0, 0.0, TABLE_Z),
            plane="xy",
        )

        self.tool_gizmo = self.server.scene.add_transform_controls(
            "/tool",
            position=tuple(self._initial_tool_xyz.astype(float)),
            wxyz=quat_xyzw_to_wxyz(self._initial_tool_quat),
            scale=0.12,
        )
        self._load_tool_mesh(tool_name)

        self.cube_gizmo = self.server.scene.add_transform_controls(
            "/cube",
            position=tuple(self._initial_cube_xyz.astype(float)),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=0.08,
        )
        self.server.scene.add_box(
            "/cube/mesh",
            dimensions=(0.04, 0.04, 0.04),
            color=(200, 120, 60),
        )

        self.goal_gizmo = self.server.scene.add_transform_controls(
            "/goal",
            position=tuple(self._initial_goal_xyz.astype(float)),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=0.08,
        )
        self.server.scene.add_icosphere(
            "/goal/mesh",
            radius=0.025,
            color=(60, 220, 100),
            opacity=0.45,
        )
        self._update_pan()

    def _clear_pan(self) -> None:
        for name in ("/pan/wall", "/pan/rim"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass

    def _update_pan(self) -> None:
        """Draw or hide the handleless pan (flip/pour tasks only)."""
        self._clear_pan()
        _, _, cube_xyz, _ = self._read_gizmo_poses_or_defaults()
        draw, center = pan_center_for_scene(
            control_frame=self.current_tool_name,
            material_xyz_world=cube_xyz,
            pan_center_xy_world=self._pan_center_xy,
        )
        if not draw:
            return
        wall, rim = make_pan_meshes(self.table_z, center)
        self.server.scene.add_mesh_simple(
            "/pan/wall",
            vertices=np.asarray(wall.vertices, dtype=np.float32),
            faces=np.asarray(wall.faces, dtype=np.int32),
            color=COLOR_PAN_WALL,
            opacity=0.51,
        )
        self.server.scene.add_mesh_simple(
            "/pan/rim",
            vertices=np.asarray(rim.vertices, dtype=np.float32),
            faces=np.asarray(rim.faces, dtype=np.int32),
            color=COLOR_PAN_RIM,
        )

    def _load_tool_mesh(self, tool_name: str) -> None:
        meta = load_control_frame_meta(self.control_frames[tool_name])
        obj_path = Path(str(meta["obj_path"]))
        if not obj_path.is_file():
            raise FileNotFoundError(f"Tool mesh not found: {obj_path}")
        mesh: trimesh.Trimesh = trimesh.load(str(obj_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")
        self._tool_mesh = mesh
        self.server.scene.remove_by_name("/tool/mesh")
        self.server.scene.add_mesh_simple(
            "/tool/mesh",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            color=(100, 160, 220),
        )

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown("# Interactive brush VLA")
        self.server.gui.add_markdown(
            "**Frame:** model/world (`frame_shift=0`). Table at `z=0.53`. "
            "Gizmo coordinates are model inputs; the `+0.8` robot Y shift is not applied here."
        )

        self.model_dropdown = self.server.gui.add_dropdown(
            "Model",
            options=self.model_keys,
            initial_value=self.current_model_key,
        )
        self.model_dropdown.on_update(self._on_model_change)

        tool_names = list(self.control_frames.keys())
        self.tool_dropdown = self.server.gui.add_dropdown(
            "Tool",
            options=tool_names,
            initial_value=self.current_tool_name,
        )
        self.tool_dropdown.on_update(self._on_tool_change)

        self.instruction_input = self.server.gui.add_text(
            "Instruction",
            initial_value=self._current_instruction,
            hint="Basic per-task default; edit to override",
        )
        self.instruction_input.on_update(self._on_instruction_change)

        self.server.gui.add_button("Predict").on_click(lambda _: self._predict_and_draw())

        self.mode_dropdown = self.server.gui.add_dropdown(
            "Play mode",
            options=("static", "closed_loop"),
            initial_value="static",
        )
        self.mode_dropdown.on_update(self._on_mode_change)

        self.chunk_slider = self.server.gui.add_slider(
            "Chunk size",
            min=1,
            max=15,
            step=1,
            initial_value=self._chunk_size,
        )
        self.chunk_slider.on_update(self._on_chunk_change)

        self.fps_slider = self.server.gui.add_slider(
            "Play speed (fps)",
            min=1,
            max=30,
            step=1,
            initial_value=int(self._play_fps),
        )
        self.fps_slider.on_update(self._on_fps_change)

        self.replan_slider = self.server.gui.add_slider(
            "Replan cap",
            min=1,
            max=50,
            step=1,
            initial_value=self._max_replans,
        )
        self.replan_slider.on_update(self._on_replan_cap_change)

        self.ghost_toggle = self.server.gui.add_checkbox(
            "Show all 15 brush ghosts",
            initial_value=False,
        )
        self.ghost_toggle.on_update(self._on_ghost_toggle)

        self.unflip_toggle = self.server.gui.add_checkbox(
            "Un-flip orientation on replan",
            initial_value=self._unflip_orientation,
        )
        self.unflip_toggle.on_update(self._on_unflip_toggle)

        self.server.gui.add_button("Play").on_click(lambda _: self._start_play())
        self.server.gui.add_button("Pause").on_click(lambda _: self._pause_play())
        self.server.gui.add_button("Reset").on_click(lambda _: self._reset_scene())

        self.status_md = self.server.gui.add_markdown("**Status:** ready")
        self.pose_md = self.server.gui.add_markdown("*No prediction yet.*")

    def _wire_gizmo_callbacks(self) -> None:
        for gizmo in (self.tool_gizmo, self.cube_gizmo, self.goal_gizmo):
            gizmo.on_update(self._on_gizmo_update)

    def _sync_policy_instruction(self) -> None:
        """Push the active instruction text into the policy before inference."""
        text = str(self.instruction_input.value).strip()
        if text:
            self._current_instruction = text
            self.policy.instruction = text

    def _set_instruction_field(self, text: str, *, customized: bool) -> None:
        self._current_instruction = text
        self._instruction_customized = customized
        self.instruction_input.value = text
        self.policy.instruction = text

    def _on_instruction_change(self, _event: viser.GuiEvent) -> None:
        self._instruction_customized = True
        self._sync_policy_instruction()
        self._schedule_predict()

    def _on_model_change(self, _event: viser.GuiEvent) -> None:
        key = str(self.model_dropdown.value)
        if key == self.current_model_key:
            return
        try:
            model_entry = resolve_model(key)
        except Exception as exc:
            self.status_md.content = f"**Status:** model load failed — {exc}"
            return
        self._stop_play_thread()
        self.control_frames = self._filter_control_frames(model_entry)
        tool_name = self.current_tool_name
        if tool_name not in self.control_frames:
            tool_name = model_entry.default_control_frame
        if tool_name not in self.control_frames:
            tool_name = next(iter(self.control_frames))
        instruction = (
            self._current_instruction
            if self._instruction_customized
            else default_instruction_for_control_frame(tool_name)
        )
        self.status_md.content = f"**Status:** loading model {key}…"
        self.policy = self._build_policy(model_entry, tool_name, instruction)
        self.current_model_key = key
        self.current_tool_name = tool_name
        self.table_z = self.policy.table_z
        # Refresh the tool dropdown to the new model's allowed control frames.
        self.tool_dropdown.options = list(self.control_frames.keys())
        self.tool_dropdown.value = tool_name
        if not self._instruction_customized:
            self._set_instruction_field(
                default_instruction_for_control_frame(tool_name), customized=False
            )
        self._load_tool_mesh(tool_name)
        self._update_pan()
        self._predict_and_draw()

    def _on_tool_change(self, _event: viser.GuiEvent) -> None:
        name = str(self.tool_dropdown.value)
        if name == self.current_tool_name:
            return
        self.current_tool_name = name
        self.policy.set_control_frame(name)
        if not self._instruction_customized:
            self._set_instruction_field(
                default_instruction_for_control_frame(name),
                customized=False,
            )
        self._load_tool_mesh(name)
        self._update_pan()
        self._predict_and_draw()

    def _on_mode_change(self, _event: viser.GuiEvent) -> None:
        self._play_mode = str(self.mode_dropdown.value)

    def _on_chunk_change(self, _event: viser.GuiEvent) -> None:
        self._chunk_size = int(self.chunk_slider.value)

    def _on_fps_change(self, _event: viser.GuiEvent) -> None:
        self._play_fps = float(self.fps_slider.value)

    def _on_replan_cap_change(self, _event: viser.GuiEvent) -> None:
        self._max_replans = int(self.replan_slider.value)

    def _on_ghost_toggle(self, _event: viser.GuiEvent) -> None:
        self._show_all_ghosts = bool(self.ghost_toggle.value)
        if self._waypoints is not None:
            self._draw_plan(self._waypoints, self._object_poses)

    def _on_unflip_toggle(self, _event: viser.GuiEvent) -> None:
        self._unflip_orientation = bool(self.unflip_toggle.value)

    def _on_gizmo_update(self, event: viser.TransformControlsEvent) -> None:
        if self._playing:
            return
        # Pan tracks material/cube position (material-relative center).
        self._update_pan()
        if event.phase == "end":
            self._schedule_predict()
        elif event.phase == "update":
            self._schedule_predict()

    def _schedule_predict(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        self._debounce_timer = threading.Timer(self.args.debounce_s, self._predict_and_draw)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _read_gizmo_poses(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tool_xyz = np.asarray(self.tool_gizmo.position, dtype=np.float64)
        tool_quat = quat_wxyz_to_xyzw(np.asarray(self.tool_gizmo.wxyz, dtype=np.float64))
        cube_xyz = np.asarray(self.cube_gizmo.position, dtype=np.float64)
        goal_xyz = np.asarray(self.goal_gizmo.position, dtype=np.float64)
        return tool_xyz, tool_quat, cube_xyz, goal_xyz

    def _read_gizmo_poses_or_defaults(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "tool_gizmo"):
            return (
                self._initial_tool_xyz.copy(),
                self._initial_tool_quat.copy(),
                self._initial_cube_xyz.copy(),
                self._initial_goal_xyz.copy(),
            )
        return self._read_gizmo_poses()

    def _make_scene(self) -> SceneState:
        self._sync_policy_instruction()
        tool_xyz, tool_quat, cube_xyz, goal_xyz = self._read_gizmo_poses()
        contact, normal, sd = contact_frame_from_root(tool_xyz, tool_quat, self.policy.T_oc)
        return SceneState(
            instruction=self.policy.instruction,
            tool_label=self.policy.tool_label,
            tool_contact_xyz_world=contact,
            tool_current_normal=normal,
            tool_current_surface_dir=sd,
            material_xyz_world=cube_xyz.astype(np.float32),
            destination_xyz_world=goal_xyz.astype(np.float32),
            table_xyz_world=np.array([0.0, 0.0, self.table_z], dtype=np.float32),
            table_z=self.table_z,
        )

    def _predict_and_draw(self) -> None:
        if not self._predict_lock.acquire(blocking=False):
            return
        try:
            if self._playing:
                return
            self.status_md.content = "**Status:** predicting…"
            scene = self._make_scene()
            t0 = time.perf_counter()
            waypoints = self.policy.predict_waypoints(scene)
            gen_ms = (time.perf_counter() - t0) * 1e3
            object_poses = self.policy.waypoints_to_object_poses_robot(waypoints, ZERO_SHIFT)
            self._waypoints = waypoints
            self._object_poses = object_poses
            self._draw_plan(waypoints, object_poses)
            self._update_pose_panel(object_poses[: self._chunk_size])
            tool_xyz, _, cube_xyz, goal_xyz = self._read_gizmo_poses()
            dist = float(np.linalg.norm(cube_xyz[:2] - goal_xyz[:2]))
            self.status_md.content = (
                f"**Status:** plan updated · gen time {gen_ms:.0f} ms · "
                f"dist(cube, goal)={dist:.3f} m"
            )
        except Exception as exc:
            self.status_md.content = f"**Status:** predict failed — {exc}"
        finally:
            self._predict_lock.release()

    def _clear_plan_overlays(self) -> None:
        for i in range(20):
            for prefix in ("/plan/path", "/plan/contact", "/plan/normal", "/plan/surface", "/plan/ghost"):
                try:
                    self.server.scene.remove_by_name(f"{prefix}_{i}")
                except Exception:
                    pass
        try:
            self.server.scene.remove_by_name("/plan/spline")
        except Exception:
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

    def _draw_arrow(
        self,
        name: str,
        origin: np.ndarray,
        direction: np.ndarray,
        color: Tuple[int, int, int],
        length: float,
    ) -> None:
        d = _unit(direction)
        if float(np.linalg.norm(d)) < 1e-6:
            return
        self._draw_segment(name, origin, origin + d * length, color)

    def _draw_plan(
        self,
        waypoints: np.ndarray,
        object_poses: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self._clear_plan_overlays()
        contacts = waypoints[:, 0:3]
        normals = waypoints[:, 3:6]
        surface_dirs = waypoints[:, 6:9]

        if contacts.shape[0] >= 2:
            self.server.scene.add_spline_catmull_rom(
                "/plan/spline",
                positions=contacts.astype(np.float32),
                color=COLOR_PATH,
                line_width=2.5,
            )

        scale = self.args.arrow_scale
        n_show = contacts.shape[0]
        for i in range(n_show):
            c = contacts[i]
            self.server.scene.add_icosphere(
                f"/plan/contact_{i}",
                radius=0.006,
                color=COLOR_CONTACT,
                position=tuple(c.astype(float)),
            )
            self._draw_arrow(
                f"/plan/normal_{i}",
                c,
                normals[i],
                COLOR_NORMAL,
                scale,
            )
            self._draw_arrow(
                f"/plan/surface_{i}",
                c,
                surface_dirs[i],
                COLOR_SURFACE,
                scale * 0.9,
            )

        n_ghosts = n_show if self._show_all_ghosts else min(self._chunk_size, n_show)
        if self._tool_mesh is not None:
            verts = np.asarray(self._tool_mesh.vertices, dtype=np.float32)
            faces = np.asarray(self._tool_mesh.faces, dtype=np.int32)
            for i in range(n_ghosts):
                xyz, quat = object_poses[i]
                self.server.scene.add_mesh_simple(
                    f"/plan/ghost_{i}",
                    vertices=verts,
                    faces=faces,
                    color=(140, 180, 220),
                    opacity=0.22,
                    wxyz=quat_xyzw_to_wxyz(quat),
                    position=tuple(xyz.astype(float)),
                )

    def _update_pose_panel(self, poses: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        lines = [f"**Chunk poses ({len(poses)}):**"]
        for i, (xyz, quat) in enumerate(poses):
            lines.append(
                f"- `{i}` pos=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) "
                f"quat=({quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f})"
            )
        self.pose_md.content = "\n".join(lines)

    def _set_tool_pose(self, xyz: np.ndarray, quat_xyzw: np.ndarray) -> None:
        self.tool_gizmo.position = tuple(xyz.astype(float))
        self.tool_gizmo.wxyz = quat_xyzw_to_wxyz(quat_xyzw)

    def _set_cube_pose(self, xyz: np.ndarray) -> None:
        self.cube_gizmo.position = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def _snapshot_initial(self) -> None:
        self._initial_tool_xyz, self._initial_tool_quat, self._initial_cube_xyz, goal = (
            self._read_gizmo_poses()
        )
        self._initial_goal_xyz = goal.copy()

    def _restore_initial(self) -> None:
        self._set_tool_pose(self._initial_tool_xyz, self._initial_tool_quat)
        self._set_cube_pose(self._initial_cube_xyz)
        self.goal_gizmo.position = tuple(self._initial_goal_xyz.astype(float))

    def _pause_play(self) -> None:
        self._play_pause.set()

    def _stop_play_thread(self) -> None:
        self._play_stop.set()
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=2.0)
        self._play_thread = None
        self._playing = False
        self._play_stop.clear()
        self._play_pause.clear()

    def _reset_scene(self) -> None:
        self._stop_play_thread()
        self._restore_initial()
        self._predict_and_draw()

    def _start_play(self) -> None:
        if self._playing:
            self._play_pause.clear()
            self.status_md.content = "**Status:** playing (resumed)"
            return
        self._snapshot_initial()
        self._play_stop.clear()
        self._play_pause.clear()
        self._play_finished_status = None
        self._playing = True
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _sleep_fps(self) -> bool:
        """Sleep one frame; return False if stop requested."""
        dt = 1.0 / max(1.0, self._play_fps)
        end = time.monotonic() + dt
        while time.monotonic() < end:
            if self._play_stop.is_set():
                return False
            while self._play_pause.is_set() and not self._play_stop.is_set():
                time.sleep(0.05)
            time.sleep(0.01)
        return not self._play_stop.is_set()

    def _animate_poses(self, poses: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for xyz, quat in poses:
            if self._play_stop.is_set():
                return False
            self._set_tool_pose(xyz, quat)
            if not self._sleep_fps():
                return False
        return True

    def _play_loop(self) -> None:
        try:
            if self._play_mode == "static":
                self._play_static()
            else:
                self._play_closed_loop()
        except Exception as exc:
            traceback.print_exc()
            self._play_finished_status = f"**Status:** play failed — {exc}"
        finally:
            self._playing = False
            final_status = None
            if not self._play_stop.is_set():
                if self._play_finished_status is not None:
                    final_status = self._play_finished_status
                else:
                    final_status = "**Status:** play finished"
                self.status_md.content = final_status
            self._predict_and_draw()
            if final_status is not None:
                self.status_md.content = final_status

    def _play_static(self) -> None:
        self.status_md.content = "**Status:** static plan play"
        if self._waypoints is None or not self._object_poses:
            self._predict_and_draw()
        poses = list(self._object_poses)
        if not poses:
            self.status_md.content = "**Status:** no plan to play"
            return
        idx = 0
        while not self._play_stop.is_set():
            if idx >= len(poses):
                idx = 0
            xyz, quat = poses[idx]
            self._set_tool_pose(xyz, quat)
            idx += 1
            if not self._sleep_fps():
                return

    def _play_closed_loop(self) -> None:
        self.status_md.content = "**Status:** closed-loop push"
        self._sync_policy_instruction()
        tool_xyz, tool_quat, cube_xyz, goal_xyz = self._read_gizmo_poses()
        contact, normal, sd = contact_frame_from_root(tool_xyz, tool_quat, self.policy.T_oc)
        obj = cube_xyz.astype(np.float32).copy()
        dest = goal_xyz.astype(np.float32)
        in_contact = False
        chunk = self._chunk_size
        ref_quat: Optional[np.ndarray] = None

        for gen in range(self._max_replans):
            if self._play_stop.is_set():
                return

            self.status_md.content = f"**Status:** replanning gen {gen}…"
            scene = SceneState(
                instruction=self.policy.instruction,
                tool_label=self.policy.tool_label,
                tool_contact_xyz_world=contact.copy(),
                tool_current_normal=normal.copy(),
                tool_current_surface_dir=sd.copy(),
                material_xyz_world=obj.copy(),
                destination_xyz_world=dest.copy(),
                table_xyz_world=np.array([0.0, 0.0, self.table_z], dtype=np.float32),
                table_z=self.table_z,
            )
            t0 = time.perf_counter()
            waypoints = self.policy.predict_waypoints(scene)
            gen_ms = (time.perf_counter() - t0) * 1e3
            object_poses = self.policy.waypoints_to_object_poses_robot(waypoints, ZERO_SHIFT)
            flip_corrected = False
            if self._unflip_orientation and ref_quat is not None:
                waypoints, info = align_waypoints_to_reference(
                    waypoints, ref_quat, self.policy.T_oc
                )
                waypoints = waypoints.astype(np.float32)
                object_poses = self.policy.waypoints_to_object_poses_robot(
                    waypoints, ZERO_SHIFT
                )
                flip_corrected = bool(info["applied"])
            if object_poses:
                ref_quat = np.asarray(object_poses[0][1], dtype=np.float64).copy()
            self._waypoints = waypoints
            self._object_poses = object_poses
            self._draw_plan(waypoints, object_poses)
            self._update_pose_panel(object_poses[:chunk])

            chunk_poses = object_poses[:chunk]
            if not chunk_poses:
                self._play_finished_status = f"**Status:** no poses returned at gen {gen}"
                return
            if not self._animate_poses(chunk_poses):
                return

            new_brush, new_obj, in_contact = execute_chunk(
                waypoints,
                object_xyz=obj,
                in_contact=in_contact,
                destination_xyz=dest,
                table_z=self.table_z,
                chunk=chunk,
            )
            obj = new_obj
            self._set_cube_pose(obj)

            dist = float(np.linalg.norm(obj[:2] - dest[:2]))
            flip_note = " · flip corrected" if flip_corrected else ""
            self.status_md.content = (
                f"**Status:** gen {gen} · gen time {gen_ms:.0f} ms · "
                f"dist={dist:.3f} m · contact={in_contact}{flip_note}"
            )
            if dist <= GOAL_REGION_RADIUS_M:
                self._play_finished_status = (
                    f"**Status:** delivered at gen {gen} · gen time {gen_ms:.0f} ms · "
                    f"dist={dist:.3f} m"
                )
                return

            if chunk_poses:
                self._set_tool_pose(chunk_poses[-1][0], chunk_poses[-1][1])

            contact = new_brush[0:3].copy()
            normal = new_brush[3:6].copy()
            sd = new_brush[6:9].copy()

        self._play_finished_status = "**Status:** replan cap reached without delivery"

    def run(self) -> None:
        while True:
            time.sleep(1.0)


def main() -> None:
    args = tyro.cli(VizArgs)
    app = InteractiveBrushViz(args)
    app.run()


if __name__ == "__main__":
    main()
