"""Interactive viser viz of the analytic closed-loop spatula flip with a real tool mesh.

Rolls out the same reactive closed-loop flip trajectory used to build
dataset_0012, then animates the actual tool mesh (e.g. flat_spatula) following the
calculated plan. Contact-frame waypoints are converted to tool root poses via the
tool's annotated control frame (``T_obj_from_contact``).

This lives in the pipeline (NOT in the ``closed_loop`` package); it only reads the
control-frame JSON (and the mesh path inside it) from ``closed_loop/`` assets. No
pan is drawn.

Usage:
    python -m generative_str_pipeline.viz_spatula_flip_viser \\
        --control-frame flat_spatula --seed 3 --port 8080
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Python 3.8 compatibility shim for viser's static file server.
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
import viser

from generative_str_pipeline.build_dataset_0012_spatula_flip_reactive import (
    PAN_CENTER_XY_M,
    PAN_RADIUS_M,
    PAN_WALL_HEIGHT_M,
    FlipGenConfig,
    scene_to_datapoints,
)
from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    load_control_frame,
    waypoint_to_object_pose,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROL_FRAMES_DIR = REPO_ROOT / "closed_loop" / "closed_loop" / "assets" / "control_frames"

COLOR_TOOL = (100, 160, 220)
COLOR_MATERIAL = (220, 150, 70)
COLOR_PATH = (90, 200, 255)
COLOR_DEST = (60, 220, 100)
COLOR_PAN = (70, 70, 80)


def quat_xyzw_to_wxyz(q: np.ndarray) -> Tuple[float, float, float, float]:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def resolve_control_frame(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    cand = CONTROL_FRAMES_DIR / f"{name_or_path}.json"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Control frame not found: {name_or_path!r} (tried {cand})")


def list_control_frames() -> dict:
    out: dict = {}
    for path in sorted(CONTROL_FRAMES_DIR.glob("*.json")):
        out[path.stem] = path.resolve()
    return out


class FlipFrame:
    """One animation frame: tool root pose + material pose."""

    __slots__ = ("tool_xyz", "tool_quat", "contact", "mat_xyz", "mat_quat", "gen")

    def __init__(self, tool_xyz, tool_quat, contact, mat_xyz, mat_quat, gen):
        self.tool_xyz = tool_xyz
        self.tool_quat = tool_quat
        self.contact = contact
        self.mat_xyz = mat_xyz
        self.mat_quat = mat_quat
        self.gen = gen


def build_flip_frames(
    *, seed: int, T_oc: np.ndarray, chunk: int, entry_pitch_rad: Optional[float] = None
) -> Tuple[List[FlipFrame], dict]:
    """Roll out the analytic closed-loop flip and convert to tool/material poses.

    Slip / flip-failure rates are whatever the dataset generator encodes
    (``FlipGenConfig`` defaults); this just watches what the dataset produces.
    ``entry_pitch_rad`` overrides the scoop dig-in pitch so it can be tuned live.
    """
    rng = np.random.default_rng(int(seed))
    cfg = FlipGenConfig()
    if entry_pitch_rad is not None:
        cfg = replace(cfg, scoop_entry_pitch_rad=float(entry_pitch_rad))
    datapoints = scene_to_datapoints(
        rng, cfg, shard_id="viz", scene_index=0, base_datapoint_index=0
    )
    chunk = max(1, int(chunk))

    frames: List[FlipFrame] = []

    first = datapoints[0]
    start_contact = np.asarray(first["tool_contact_xyz_world"], dtype=np.float64)
    start_normal = np.asarray(first["tool_current_normal"], dtype=np.float64)
    start_sd = np.asarray(first["tool_current_surface_dir"], dtype=np.float64)
    txyz, tquat = waypoint_to_object_pose(start_contact, start_normal, start_sd, T_oc)
    frames.append(
        FlipFrame(
            txyz,
            tquat,
            start_contact,
            np.asarray(first["material_xyz_world"], dtype=np.float64),
            np.asarray(first["material_quat_world"], dtype=np.float64),
            0,
        )
    )

    for dp in datapoints:
        wps = np.asarray(dp["waypoints"], dtype=np.float64).reshape(-1, 9)
        mat_xyz = np.asarray(
            dp["material_xyz_executed_world"], dtype=np.float64
        ).reshape(-1, 3)
        mat_quat = np.asarray(
            dp["material_quat_executed_world"], dtype=np.float64
        ).reshape(-1, 4)
        gen = int(dp.get("window_index", 0))
        n_take = min(chunk, wps.shape[0], mat_xyz.shape[0])
        for i in range(n_take):
            contact = wps[i, 0:3]
            normal = wps[i, 3:6]
            sd = wps[i, 6:9]
            txyz, tquat = waypoint_to_object_pose(contact, normal, sd, T_oc)
            frames.append(
                FlipFrame(txyz, tquat, contact, mat_xyz[i], mat_quat[i], gen)
            )

    meta = {
        "instruction": str(first["instruction"]),
        "material_size": np.asarray(first["material_size"], dtype=np.float64),
        "destination": np.asarray(first["destination_xyz_world"], dtype=np.float64),
        "table_z": float(first["table_xyz_world"][2]),
        "num_generations": len(datapoints),
        "reached_goal": bool(datapoints[-1]["reached_goal"]),
    }
    return frames, meta


class SpatulaFlipViz:
    def __init__(
        self,
        *,
        control_frame: str,
        seed: int,
        port: int,
        chunk: int,
        fps: float,
        entry_pitch_rad: Optional[float] = None,
    ):
        self.control_frames = list_control_frames()
        cf_path = resolve_control_frame(control_frame)
        self.control_frame_name = cf_path.stem
        self.T_oc = load_control_frame(cf_path)
        self.obj_path = Path(json.loads(cf_path.read_text())["obj_path"])
        self.mesh: trimesh.Trimesh = self._load_mesh(self.obj_path)

        self.seed = int(seed)
        self.chunk = int(chunk)
        self.fps = float(fps)
        self.entry_pitch_rad = (
            float(entry_pitch_rad)
            if entry_pitch_rad is not None
            else float(FlipGenConfig().scoop_entry_pitch_rad)
        )

        self._play_stop = threading.Event()
        self._play_pause = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._playing = False

        self.frames, self.meta = build_flip_frames(
            seed=self.seed,
            T_oc=self.T_oc,
            chunk=self.chunk,
            entry_pitch_rad=self.entry_pitch_rad,
        )

        self.server = viser.ViserServer(host="0.0.0.0", port=int(port))
        self._setup_scene()
        self._setup_gui()
        self._show_frame(0)
        print(f"Viser: http://0.0.0.0:{port}  (port-forward to your machine)")

    @staticmethod
    def _load_mesh(obj_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(obj_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")
        return mesh

    def _setup_scene(self) -> None:
        table_z = float(self.meta["table_z"])

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (0.35, -0.55, table_z + 0.30)
            client.camera.look_at = (0.05, 0.0, table_z)

        self.table_handle = self.server.scene.add_grid(
            "/table", width=0.8, height=0.8, position=(0.0, 0.0, table_z), plane="xy"
        )
        self._add_pan(table_z)
        self._add_tool_mesh()

        size = self.meta["material_size"]
        self.material_handle = self.server.scene.add_box(
            "/material",
            dimensions=(float(size[0]), float(size[1]), float(size[2])),
            color=COLOR_MATERIAL,
        )
        dest = self.meta["destination"]
        self.server.scene.add_icosphere(
            "/destination",
            radius=0.022,
            color=COLOR_DEST,
            opacity=0.45,
            position=tuple(np.asarray(dest, dtype=float)),
        )
        self._draw_contact_path()

    def _add_pan(self, table_z: float) -> None:
        """Draw the handleless pan (rim wall + floor) matching the flip task geometry.

        Re-callable: replaces any existing pan so it tracks the (jittered) table
        height whenever the scene is re-rolled.
        """
        self.server.scene.remove_by_name("/pan_wall")
        self.server.scene.remove_by_name("/pan_floor")
        cx, cy = float(PAN_CENTER_XY_M[0]), float(PAN_CENTER_XY_M[1])
        radius = float(PAN_RADIUS_M)
        height = float(PAN_WALL_HEIGHT_M)
        thickness = 0.008

        # Hollow cylindrical rim wall: inner radius = pan radius, walls of
        # `thickness`, standing `height` tall on the table surface.
        wall = trimesh.creation.annulus(
            r_min=radius, r_max=radius + thickness, height=height
        )
        wall.apply_translation([cx, cy, table_z + 0.5 * height])
        self.server.scene.add_mesh_simple(
            "/pan_wall",
            vertices=np.asarray(wall.vertices, dtype=np.float32),
            faces=np.asarray(wall.faces, dtype=np.int32),
            color=COLOR_PAN,
            opacity=0.55,
        )

        # Thin floor disk so the pan reads as a vessel.
        floor = trimesh.creation.cylinder(radius=radius, height=0.004)
        floor.apply_translation([cx, cy, table_z + 0.002])
        self.server.scene.add_mesh_simple(
            "/pan_floor",
            vertices=np.asarray(floor.vertices, dtype=np.float32),
            faces=np.asarray(floor.faces, dtype=np.int32),
            color=COLOR_PAN,
            opacity=0.35,
        )

    def _add_tool_mesh(self) -> None:
        self.server.scene.remove_by_name("/tool")
        self.tool_handle = self.server.scene.add_mesh_simple(
            "/tool",
            vertices=np.asarray(self.mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.mesh.faces, dtype=np.int32),
            color=COLOR_TOOL,
        )

    def _draw_contact_path(self) -> None:
        self.server.scene.remove_by_name("/contact_path")
        pts = np.array([f.contact for f in self.frames], dtype=np.float32)
        if pts.shape[0] >= 2:
            self.server.scene.add_spline_catmull_rom(
                "/contact_path", positions=pts, color=COLOR_PATH, line_width=2.5
            )

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown("# Spatula flip — closed-loop plan (real tool)")
        self.server.gui.add_markdown(
            "Analytic reactive flip rollout; contact-frame waypoints mapped to the "
            "tool root via its control frame. Pan rim shown for clearance checks."
        )

        names = sorted(self.control_frames.keys())
        if self.control_frame_name not in names:
            names = [self.control_frame_name] + names
        self.tool_dropdown = self.server.gui.add_dropdown(
            "Tool", options=tuple(names), initial_value=self.control_frame_name
        )
        self.tool_dropdown.on_update(self._on_tool_change)

        self.seed_slider = self.server.gui.add_slider(
            "Seed", min=0, max=200, step=1, initial_value=self.seed
        )
        self.chunk_slider = self.server.gui.add_slider(
            "Chunk (executed/gen)", min=1, max=15, step=1, initial_value=self.chunk
        )
        self.pitch_slider = self.server.gui.add_slider(
            "Scoop dig-in pitch (deg)",
            min=0,
            max=70,
            step=1,
            initial_value=int(round(np.degrees(self.entry_pitch_rad))),
        )
        self.pitch_slider.on_update(self._on_pitch_change)
        self.fps_slider = self.server.gui.add_slider(
            "Play speed (fps)", min=1, max=30, step=1, initial_value=int(self.fps)
        )
        self.fps_slider.on_update(lambda _: setattr(self, "fps", float(self.fps_slider.value)))

        self.frame_slider = self.server.gui.add_slider(
            "Frame", min=0, max=max(1, len(self.frames) - 1), step=1, initial_value=0
        )
        self.frame_slider.on_update(self._on_frame_slider)

        self.server.gui.add_button("Re-roll trajectory").on_click(lambda _: self._reroll())
        self.server.gui.add_button("Play").on_click(lambda _: self._start_play())
        self.server.gui.add_button("Pause").on_click(lambda _: self._pause_play())
        self.server.gui.add_button("Reset").on_click(lambda _: self._reset())

        self.status_md = self.server.gui.add_markdown("**Status:** ready")
        self._refresh_status(0)

    def _refresh_status(self, idx: int) -> None:
        f = self.frames[idx]
        goal = "reached goal" if self.meta["reached_goal"] else "no goal"
        self.status_md.content = (
            f"**Status:** frame {idx}/{len(self.frames) - 1} · gen {f.gen}/"
            f"{self.meta['num_generations'] - 1} · {goal}\n\n"
            f"*\"{self.meta['instruction']}\"*"
        )

    def _show_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.frames) - 1))
        f = self.frames[idx]
        self.tool_handle.position = tuple(np.asarray(f.tool_xyz, dtype=float))
        self.tool_handle.wxyz = quat_xyzw_to_wxyz(f.tool_quat)
        self.material_handle.position = tuple(np.asarray(f.mat_xyz, dtype=float))
        self.material_handle.wxyz = quat_xyzw_to_wxyz(f.mat_quat)
        self._refresh_status(idx)

    def _on_frame_slider(self, _event: Any) -> None:
        if not self._playing:
            self._show_frame(int(self.frame_slider.value))

    def _on_tool_change(self, _event: Any) -> None:
        name = str(self.tool_dropdown.value)
        if name == self.control_frame_name:
            return
        cf_path = resolve_control_frame(name)
        self.control_frame_name = name
        self.T_oc = load_control_frame(cf_path)
        self.obj_path = Path(json.loads(cf_path.read_text())["obj_path"])
        self.mesh = self._load_mesh(self.obj_path)
        self._add_tool_mesh()
        self._reroll()

    def _rebuild(self) -> None:
        self.seed = int(self.seed_slider.value)
        self.chunk = int(self.chunk_slider.value)
        self.entry_pitch_rad = float(np.radians(float(self.pitch_slider.value)))
        self.frames, self.meta = build_flip_frames(
            seed=self.seed,
            T_oc=self.T_oc,
            chunk=self.chunk,
            entry_pitch_rad=self.entry_pitch_rad,
        )
        # Table height is jittered per scene, so move the table grid + pan to the
        # new surface on every re-roll (otherwise the pan stays at the old height
        # and everything else looks buried in / floating above it).
        table_z = float(self.meta["table_z"])
        self.table_handle.position = (0.0, 0.0, table_z)
        self._add_pan(table_z)
        self.frame_slider.max = max(1, len(self.frames) - 1)
        self.frame_slider.value = 0
        self._draw_contact_path()
        self._show_frame(0)

    def _on_pitch_change(self, _event: Any) -> None:
        self._reroll()

    def _reroll(self) -> None:
        self._stop_play_thread()
        self._rebuild()

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

    def _reset(self) -> None:
        self._stop_play_thread()
        self.frame_slider.value = 0
        self._show_frame(0)

    def _start_play(self) -> None:
        if self._playing:
            self._play_pause.clear()
            return
        self._play_stop.clear()
        self._play_pause.clear()
        self._playing = True
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _play_loop(self) -> None:
        try:
            start = int(self.frame_slider.value)
            if start >= len(self.frames) - 1:
                start = 0
            for idx in range(start, len(self.frames)):
                if self._play_stop.is_set():
                    return
                while self._play_pause.is_set() and not self._play_stop.is_set():
                    time.sleep(0.05)
                self._show_frame(idx)
                self.frame_slider.value = idx
                time.sleep(1.0 / max(1.0, self.fps))
        finally:
            self._playing = False

    def run(self) -> None:
        while True:
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser spatula flip closed-loop viz (real tool).")
    parser.add_argument("--control-frame", default="flat_spatula")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--entry-pitch-deg",
        type=float,
        default=None,
        help="Scoop dig-in pitch in degrees (overrides the dataset default).",
    )
    args = parser.parse_args()

    app = SpatulaFlipViz(
        control_frame=args.control_frame,
        seed=args.seed,
        port=args.port,
        chunk=args.chunk,
        fps=args.fps,
        entry_pitch_rad=(
            None if args.entry_pitch_deg is None else float(np.radians(args.entry_pitch_deg))
        ),
    )
    app.run()


if __name__ == "__main__":
    main()
