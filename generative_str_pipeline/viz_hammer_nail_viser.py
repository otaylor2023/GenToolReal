"""Interactive viser viz of the analytic closed-loop hammer-a-nail with a real tool mesh.

Rolls out the same reactive closed-loop hammering trajectory used to build
dataset_0014, then animates the actual hammer mesh (e.g. claw_hammer) striking a
nail that protrudes from a board. Each solid strike drives the nail head down a
step until it reaches the target depth, then the hammer lifts clear.

Contact-frame waypoints are converted to tool root poses via the tool's annotated
control frame (``T_obj_from_contact``). The board, nail (head + shaft), and the
target depth marker are drawn so the strike geometry is easy to eyeball.

Usage:
    python -m generative_str_pipeline.viz_hammer_nail_viser \\
        --control-frame claw_hammer --seed 3 --port 8080
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
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

from generative_str_pipeline.build_dataset_0014_hammer_nail_reactive import (
    HammerNailGenConfig,
    scene_to_datapoints,
)
from generative_str_pipeline.sim_rollout.waypoint_to_pose import (
    load_control_frame,
    waypoint_to_object_pose,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROL_FRAMES_DIR = REPO_ROOT / "closed_loop" / "closed_loop" / "assets" / "control_frames"

COLOR_TOOL = (110, 120, 130)
COLOR_NAIL = (200, 200, 210)
COLOR_NAIL_HEAD = (170, 170, 185)
COLOR_BOARD = (170, 120, 70)
COLOR_PATH = (90, 200, 255)
COLOR_TARGET = (60, 220, 100)


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


class HammerFrame:
    """One animation frame: tool root pose + nail-head position."""

    __slots__ = ("tool_xyz", "tool_quat", "contact", "head_xyz", "gen", "hits", "sink")

    def __init__(self, tool_xyz, tool_quat, contact, head_xyz, gen, hits, sink):
        self.tool_xyz = tool_xyz
        self.tool_quat = tool_quat
        self.contact = contact
        self.head_xyz = head_xyz
        self.gen = gen
        self.hits = hits
        self.sink = sink


def build_hammer_frames(
    *, seed: int, T_oc: np.ndarray, chunk: int, sink_target_m: Optional[float] = None
) -> Tuple[List[HammerFrame], dict]:
    """Roll out the analytic hammering and convert to tool/nail poses."""
    rng = np.random.default_rng(int(seed))
    cfg = HammerNailGenConfig()
    if sink_target_m is not None:
        s = float(sink_target_m)
        cfg = HammerNailGenConfig(sink_target_m_range=(s, s))
    datapoints = scene_to_datapoints(
        rng, cfg, shard_id="viz", scene_index=0, base_datapoint_index=0
    )
    chunk = max(1, int(chunk))

    frames: List[HammerFrame] = []
    first = datapoints[0]
    start_contact = np.asarray(first["tool_contact_xyz_world"], dtype=np.float64)
    start_normal = np.asarray(first["tool_current_normal"], dtype=np.float64)
    start_sd = np.asarray(first["tool_current_surface_dir"], dtype=np.float64)
    txyz, tquat = waypoint_to_object_pose(start_contact, start_normal, start_sd, T_oc)
    frames.append(
        HammerFrame(
            txyz,
            tquat,
            start_contact,
            np.asarray(first["material_xyz_world"], dtype=np.float64),
            0,
            0,
            0.0,
        )
    )

    for dp in datapoints:
        wps = np.asarray(dp["waypoints"], dtype=np.float64).reshape(-1, 9)
        head_trace = np.asarray(
            dp["material_xyz_executed_world"], dtype=np.float64
        ).reshape(-1, 3)
        gen = int(dp.get("window_index", 0))
        hits = int(dp.get("hits_done", 0))
        sink = float(dp.get("nail_sink_m", 0.0))
        n_take = min(chunk, wps.shape[0], head_trace.shape[0])
        for i in range(n_take):
            contact = wps[i, 0:3]
            normal = wps[i, 3:6]
            sd = wps[i, 6:9]
            txyz, tquat = waypoint_to_object_pose(contact, normal, sd, T_oc)
            frames.append(
                HammerFrame(txyz, tquat, contact, head_trace[i], gen, hits, sink)
            )

    meta = {
        "instruction": str(first["instruction"]),
        "table_z": float(first["table_xyz_world"][2]),
        "board_xyz": np.asarray(first["board_xyz_world"], dtype=np.float64),
        "board_size": np.asarray(first["board_size"], dtype=np.float64),
        "nail_head_size": np.asarray(first["nail_head_size"], dtype=np.float64),
        "nail_shaft_radius": float(first["nail_shaft_radius"]),
        "target_z": float(first["nail_target_z"]),
        "head_start_z": float(first["material_xyz_world"][2]),
        "num_generations": len(datapoints),
        "reached_goal": bool(datapoints[-1]["reached_goal"]),
        "total_hits": int(datapoints[-1].get("hits_done", 0)),
    }
    return frames, meta


class HammerNailViz:
    def __init__(
        self,
        *,
        control_frame: str,
        seed: int,
        port: int,
        chunk: int,
        fps: float,
        sink_target_m: Optional[float] = None,
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
        self.sink_target_m = sink_target_m

        self._play_stop = threading.Event()
        self._play_pause = threading.Event()
        self._play_thread: Optional[threading.Thread] = None
        self._playing = False

        self.frames, self.meta = build_hammer_frames(
            seed=self.seed, T_oc=self.T_oc, chunk=self.chunk, sink_target_m=self.sink_target_m
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
            client.camera.position = (0.35, -0.45, table_z + 0.35)
            client.camera.look_at = (0.0, 0.0, table_z + 0.05)

        self.server.scene.add_grid(
            "/table", width=0.8, height=0.8, position=(0.0, 0.0, table_z), plane="xy"
        )
        self._add_board(table_z)
        self._add_nail()
        self._add_target()
        self._add_tool_mesh()
        self._draw_contact_path()

    def _add_board(self, table_z: float) -> None:
        self.server.scene.remove_by_name("/board")
        bx, by, bz = (float(v) for v in self.meta["board_size"])
        bc = self.meta["board_xyz"]
        self.server.scene.add_box(
            "/board",
            dimensions=(bx, by, bz),
            color=COLOR_BOARD,
            position=(float(bc[0]), float(bc[1]), table_z + 0.5 * bz),
        )

    def _add_nail(self) -> None:
        # Nail = head box + a fixed shaft cylinder beneath it; the whole rigid
        # nail translates straight down as it is driven in (the board occludes
        # the buried shaft).
        hx, hy, hz = (float(v) for v in self.meta["nail_head_size"])
        self.nail_head_handle = self.server.scene.add_box(
            "/nail_head", dimensions=(hx, hy, hz), color=COLOR_NAIL_HEAD
        )
        shaft_r = float(self.meta["nail_shaft_radius"])
        self.shaft_len = 0.10
        shaft = trimesh.creation.cylinder(radius=shaft_r, height=self.shaft_len)
        self.nail_shaft_handle = self.server.scene.add_mesh_simple(
            "/nail_shaft",
            vertices=np.asarray(shaft.vertices, dtype=np.float32),
            faces=np.asarray(shaft.faces, dtype=np.int32),
            color=COLOR_NAIL,
        )
        self._head_half_z = 0.5 * hz

    def _add_target(self) -> None:
        # Translucent disk at the target head depth.
        hx, hy, _ = (float(v) for v in self.meta["nail_head_size"])
        bc = self.meta["board_xyz"]
        self.server.scene.add_box(
            "/target",
            dimensions=(hx * 1.4, hy * 1.4, 0.002),
            color=COLOR_TARGET,
            opacity=0.5,
            position=(float(bc[0]), float(bc[1]), float(self.meta["target_z"])),
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
        self.server.gui.add_markdown("# Hammer a nail — closed-loop plan (real tool)")
        self.server.gui.add_markdown(
            "Analytic reactive hammering: strike the nail head straight down, "
            "driving it a step per hit until it reaches the target depth, then lift."
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
        init_sink = (
            int(round(1000 * self.sink_target_m)) if self.sink_target_m is not None else 28
        )
        self.sink_slider = self.server.gui.add_slider(
            "Sink target (mm)", min=5, max=60, step=1, initial_value=init_sink
        )
        self.sink_slider.on_update(self._on_sink_change)
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
        goal = "reached target" if self.meta["reached_goal"] else "not reached"
        self.status_md.content = (
            f"**Status:** frame {idx}/{len(self.frames) - 1} · gen {f.gen}/"
            f"{self.meta['num_generations'] - 1} · hits {f.hits} · sink "
            f"{f.sink * 1000:.0f}mm/{(self.meta['head_start_z'] - self.meta['target_z']) * 1000:.0f}mm"
            f" · {goal}\n\n*\"{self.meta['instruction']}\"*"
        )

    def _show_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.frames) - 1))
        f = self.frames[idx]
        self.tool_handle.position = tuple(np.asarray(f.tool_xyz, dtype=float))
        self.tool_handle.wxyz = quat_xyzw_to_wxyz(f.tool_quat)
        head = np.asarray(f.head_xyz, dtype=float)
        self.nail_head_handle.position = (float(head[0]), float(head[1]), float(head[2]))
        # Shaft hangs directly beneath the head (its top at the head's underside).
        shaft_z = float(head[2]) - self._head_half_z - 0.5 * self.shaft_len
        self.nail_shaft_handle.position = (float(head[0]), float(head[1]), shaft_z)
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

    def _on_sink_change(self, _event: Any) -> None:
        self._reroll()

    def _rebuild(self) -> None:
        self.seed = int(self.seed_slider.value)
        self.chunk = int(self.chunk_slider.value)
        self.sink_target_m = float(self.sink_slider.value) / 1000.0
        self.frames, self.meta = build_hammer_frames(
            seed=self.seed, T_oc=self.T_oc, chunk=self.chunk, sink_target_m=self.sink_target_m
        )
        # Board / nail / target geometry depends on the scene; refresh them.
        self._add_board(float(self.meta["table_z"]))
        self.server.scene.remove_by_name("/target")
        self._add_target()
        self.frame_slider.max = max(1, len(self.frames) - 1)
        self.frame_slider.value = 0
        self._draw_contact_path()
        self._show_frame(0)

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
    parser = argparse.ArgumentParser(description="Viser hammer-a-nail closed-loop viz (real tool).")
    parser.add_argument("--control-frame", default="claw_hammer")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--sink-target-m",
        type=float,
        default=None,
        help="Fixed sink depth in meters (overrides the dataset range).",
    )
    args = parser.parse_args()

    app = HammerNailViz(
        control_frame=args.control_frame,
        seed=args.seed,
        port=args.port,
        chunk=args.chunk,
        fps=args.fps,
        sink_target_m=args.sink_target_m,
    )
    app.run()


if __name__ == "__main__":
    main()
