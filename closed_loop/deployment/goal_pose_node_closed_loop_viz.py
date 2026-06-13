#!/usr/bin/env python
"""Viser visualization of the closed-loop goal pose node.

Drop-in debugging twin of ``goal_pose_node_closed_loop.py``: it subscribes to the
SAME live ROS pose inputs (tool pose + material pose), runs the SAME closed-loop
predict/replan update loop, optionally re-publishes ``/robot_frame/goal_object_pose``,
and additionally renders the FULL 15-point generated trajectory in a viser scene
(contact points, spline, per-waypoint normal/surface_dir arrows, and tool-mesh
ghosts at every waypoint pose).

Everything is drawn in the ROBOT frame (the same frame as the published goal pose
and the live ROS inputs). ``policy.last_plan_object_poses`` are already robot-frame;
``policy.last_plan_waypoints`` are model-frame contact waypoints whose xyz are
shifted into the robot frame by subtracting ``frame_shift`` (directions are
frame-invariant).

This tool exists to debug a bug where the predicted tool orientation sometimes
flips ~180 degrees on replan; consecutive-plan orientation flips are detected,
logged, and visually flagged (spline + ghosts turn red, GUI status panel reports
"ORIENTATION FLIP DETECTED at replan N").

Install on the ROS python env::

    pip install -e /path/to/closed_loop            # core
    pip install -e "/path/to/closed_loop[viz]"     # trimesh + viser for this tool

Run::

    python deployment/goal_pose_node_closed_loop_viz.py \\
        --fixed-destination -0.365 -0.056 0.517 \\
        --control-frame blue_brush \\
        --port 8080
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tyro

try:
    import rospy
    from geometry_msgs.msg import Pose, PoseStamped

    _ROS_AVAILABLE = True
except ImportError:
    rospy = None  # type: ignore[assignment]
    Pose = PoseStamped = object  # type: ignore[assignment,misc]
    _ROS_AVAILABLE = False

try:
    from closed_loop import (
        ClosedLoopBrushPolicy,
        default_instruction_for_control_frame,
        forward_axis_from_quat,
        load_closed_loop_policy,
        orientation_flip,
    )
    from closed_loop.paths import resolve_control_frame
except ImportError as exc:
    raise SystemExit(
        "closed_loop package not found. Install with:\n"
        "  pip install -e /path/to/closed_loop\n"
        "from the SimToolReal machine (sibling of simtoolreal/)."
    ) from exc

import trimesh
import viser

try:
    from termcolor import colored
except ImportError:

    def colored(message: str, _color: str) -> str:  # type: ignore[misc]
        return message


COLOR_NORMAL = (255, 60, 60)
COLOR_SURFACE = (255, 220, 50)
COLOR_PATH = (80, 200, 255)
COLOR_PATH_FLIP = (255, 40, 40)
COLOR_CONTACT = (180, 180, 255)
COLOR_GHOST = (140, 180, 220)
COLOR_GHOST_FLIP = (235, 90, 90)
COLOR_GOAL = (60, 220, 100)
COLOR_MATERIAL = (200, 120, 60)


def info(message: str) -> None:
    print(colored(message, "green"))


def warn(message: str) -> None:
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None) -> None:
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}
    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)
    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def pose_to_xyzw(msg) -> np.ndarray:
    return np.array(
        [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ],
        dtype=np.float64,
    )


def xyzw_to_pose(xyzw: np.ndarray):
    p = Pose()
    p.position.x = float(xyzw[0])
    p.position.y = float(xyzw[1])
    p.position.z = float(xyzw[2])
    p.orientation.x = float(xyzw[3])
    p.orientation.y = float(xyzw[4])
    p.orientation.z = float(xyzw[5])
    p.orientation.w = float(xyzw[6])
    return p


def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray
) -> float:
    from isaacgymenvs.utils.observation_action_utils_sharpa import (
        _compute_keypoint_positions,
    )

    object_keypoint_positions = _compute_keypoint_positions(
        pose=pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    keypoint_distances_l2 = np.linalg.norm(keypoints_rel_goal, axis=-1).max(axis=-1)
    return float(keypoint_distances_l2[0])


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return v / n


def quat_xyzw_to_wxyz(q: np.ndarray) -> Tuple[float, float, float, float]:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class GoalPoseNodeClosedLoopViz:
    def __init__(
        self,
        policy: "ClosedLoopBrushPolicy",
        *,
        object_scales: np.ndarray,
        success_threshold: float,
        success_steps: int,
        control_frame: str,
        port: int,
        arrow_scale: float,
        publish: bool,
        tool_topic: str,
        material_topic: str,
    ):
        rospy.init_node("goal_pose_node_closed_loop_viz")

        KEYPOINT_SCALE = 1.5
        self.policy = policy
        self.object_scales = object_scales
        self.success_threshold = success_threshold
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SCALE
        self.success_steps = success_steps
        self.current_success_steps = 0
        self.publish = publish
        self.arrow_scale = float(arrow_scale)
        self.frame_shift = np.asarray(policy.frame_shift, dtype=np.float64).reshape(3)

        self.latest_tool_pose = None
        self.latest_material_pose = None
        self.goal_poses_robot: np.ndarray = np.zeros((0, 7), dtype=np.float64)
        self.current_goal_index = 0
        self.initialized = False

        self.replan_count = 0
        self._prev_plan_axis: Optional[np.ndarray] = None
        self._flip_detected = False
        self._last_flip_replan = -1
        self._last_dot = 1.0
        self._smoothing_on = bool(getattr(policy, "unflip_orientation", False))
        self._last_corrected_replan = -1

        self._tool_mesh = self._load_tool_mesh(control_frame)

        if self.publish:
            self.goal_object_pose_pub = rospy.Publisher(
                "/robot_frame/goal_object_pose", Pose, queue_size=1
            )
        else:
            self.goal_object_pose_pub = None

        rospy.Subscriber(tool_topic, PoseStamped, self._tool_pose_callback, queue_size=1)
        rospy.Subscriber(
            material_topic, PoseStamped, self._material_pose_callback, queue_size=1
        )

        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup_scene()
        self._setup_gui()
        info(f"Viser: http://0.0.0.0:{port}  (port-forward to your machine)")

    def _load_tool_mesh(self, control_frame: str):
        import json
        from pathlib import Path

        cf_path = resolve_control_frame(control_frame)
        meta = json.loads(Path(cf_path).read_text(encoding="utf-8"))
        obj_path = Path(str(meta["obj_path"]))
        if not obj_path.is_file():
            warn(f"Tool mesh not found, ghosts disabled: {obj_path}")
            return None
        mesh = trimesh.load(str(obj_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            warn(f"Tool mesh is not a Trimesh ({type(mesh)}); ghosts disabled")
            return None
        return mesh

    def _setup_scene(self) -> None:
        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.35, -0.55, 0.30)
            client.camera.look_at = (-0.15, 0.0, 0.53)

        self.server.scene.add_grid(
            "/table", width=1.2, height=1.2, position=(0.0, 0.0, 0.53), plane="xy"
        )

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown("# Closed-loop goal node (robot frame)")
        self.server.gui.add_markdown(
            "Full 15-point plan from the live VLA. Red spline/ghosts = "
            "**orientation flip** detected on replan."
        )
        self.status_md = self.server.gui.add_markdown("**Status:** waiting for poses…")
        self.flip_md = self.server.gui.add_markdown("**Flip:** none")

    def _set_status(self) -> None:
        dist = self._dist_to_goal()
        dist_s = f"{dist:.3f} m" if dist is not None else "n/a"
        self.status_md.content = (
            f"**Status:** replans={self.replan_count} · done={self.policy.done} · "
            f"dist(mat,goal)={dist_s}"
        )
        if self._last_corrected_replan >= 0:
            self.flip_md.content = (
                f"**Flip:** corrected at replan {self._last_corrected_replan} "
                f"(smoothing on, dot={self._last_dot:+.3f})"
            )
        elif self._flip_detected:
            self.flip_md.content = (
                f"**Flip:** ORIENTATION FLIP DETECTED at replan "
                f"{self._last_flip_replan} (dot={self._last_dot:+.3f})"
            )
        else:
            suffix = " · smoothing on" if self._smoothing_on else ""
            self.flip_md.content = f"**Flip:** none (dot={self._last_dot:+.3f}){suffix}"

    def _dist_to_goal(self) -> Optional[float]:
        if self.latest_material_pose is None:
            return None
        dest = self.policy._destination_model
        if dest is None:
            return None
        mat = pose_to_xyzw(self.latest_material_pose)[:3]
        mat_model = mat + self.frame_shift
        return float(np.linalg.norm(mat_model[:2] - np.asarray(dest)[:2]))

    def _tool_pose_callback(self, msg) -> None:
        self.latest_tool_pose = msg.pose

    def _material_pose_callback(self, msg) -> None:
        self.latest_material_pose = msg.pose

    def _wait_for_observations(self) -> None:
        while not rospy.is_shutdown():
            if self.latest_tool_pose is None or self.latest_material_pose is None:
                warn_every("Waiting for tool and material poses", n_seconds=1.0)
                time.sleep(0.1)
            else:
                info("Tool and material poses received")
                break

    def _initialize_policy(self) -> None:
        tool = pose_to_xyzw(self.latest_tool_pose)
        mat = pose_to_xyzw(self.latest_material_pose)[:3]
        self.policy.reset(tool[:3], tool[3:7], mat)
        self._on_replan()
        self._load_chunk_from_policy()
        self.initialized = True
        info(f"Policy initialized; chunk has {len(self.goal_poses_robot)} goals")

    def _load_chunk_from_policy(self) -> None:
        chunk = self.policy.plan_chunk()
        self.goal_poses_robot = np.array(
            [np.concatenate([xyz, quat]) for xyz, quat in chunk],
            dtype=np.float64,
        )
        self.current_goal_index = 0
        self.current_success_steps = 0

    def _maybe_replan(self) -> None:
        if self.policy.done:
            return
        tool = pose_to_xyzw(self.latest_tool_pose)
        mat = pose_to_xyzw(self.latest_material_pose)[:3]
        self.policy.observe(tool[:3], tool[3:7], mat)
        if not self.policy.done:
            self._on_replan()
            self._load_chunk_from_policy()
            info("Replanned new goal chunk from VLA")

    def _on_replan(self) -> None:
        """Capture the freshly stored full plan, run flip detection, redraw."""
        object_poses = self.policy.last_plan_object_poses
        waypoints = self.policy.last_plan_waypoints
        if not object_poses or waypoints is None:
            return
        self.replan_count += 1

        if getattr(self.policy, "last_replan_unflipped", False):
            self._last_corrected_replan = self.replan_count
            info(
                f"ORIENTATION FLIP CORRECTED at replan {self.replan_count} "
                f"(policy smoothing un-flipped the new plan)"
            )

        new_axis = forward_axis_from_quat(object_poses[0][1])
        if self._prev_plan_axis is not None:
            flipped, dot = orientation_flip(self._prev_plan_axis, new_axis)
            self._last_dot = dot
            if flipped:
                self._flip_detected = True
                self._last_flip_replan = self.replan_count
                warn(
                    f"ORIENTATION FLIP DETECTED at replan {self.replan_count}: "
                    f"forward-axis dot={dot:+.3f} (prev vs new waypoint 0)"
                )
            else:
                self._flip_detected = False
        self._prev_plan_axis = new_axis

        self._draw_full_plan(np.asarray(waypoints), object_poses)
        self._set_status()

    def _draw_segment(self, name, start, end, color) -> None:
        pts = np.array([[start, end]], dtype=np.float32)
        colors = np.array([[color, color]], dtype=np.uint8)
        self.server.scene.add_line_segments(name, points=pts, colors=colors, line_width=2.0)

    def _draw_arrow(self, name, origin, direction, color, length) -> None:
        d = _unit(direction)
        if float(np.linalg.norm(d)) < 1e-6:
            return
        self._draw_segment(name, origin, origin + d * length, color)

    def _clear_plan_overlays(self) -> None:
        for i in range(16):
            for prefix in (
                "/plan/contact",
                "/plan/normal",
                "/plan/surface",
                "/plan/ghost",
            ):
                try:
                    self.server.scene.remove_by_name(f"{prefix}_{i}")
                except Exception:
                    pass
        for name in ("/plan/spline", "/goal/marker", "/material/marker"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass

    def _draw_full_plan(
        self,
        waypoints: np.ndarray,
        object_poses: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self._clear_plan_overlays()

        contacts_robot = waypoints[:, 0:3].astype(np.float64) - self.frame_shift
        normals = waypoints[:, 3:6]
        surface_dirs = waypoints[:, 6:9]

        path_color = COLOR_PATH_FLIP if self._flip_detected else COLOR_PATH
        ghost_color = COLOR_GHOST_FLIP if self._flip_detected else COLOR_GHOST

        if contacts_robot.shape[0] >= 2:
            self.server.scene.add_spline_catmull_rom(
                "/plan/spline",
                positions=contacts_robot.astype(np.float32),
                color=path_color,
                line_width=2.5,
            )

        scale = self.arrow_scale
        for i in range(contacts_robot.shape[0]):
            c = contacts_robot[i]
            self.server.scene.add_icosphere(
                f"/plan/contact_{i}",
                radius=0.006,
                color=COLOR_CONTACT,
                position=tuple(c.astype(float)),
            )
            self._draw_arrow(f"/plan/normal_{i}", c, normals[i], COLOR_NORMAL, scale)
            self._draw_arrow(
                f"/plan/surface_{i}", c, surface_dirs[i], COLOR_SURFACE, scale * 0.9
            )

        if self._tool_mesh is not None:
            verts = np.asarray(self._tool_mesh.vertices, dtype=np.float32)
            faces = np.asarray(self._tool_mesh.faces, dtype=np.int32)
            for i, (xyz, quat) in enumerate(object_poses):
                self.server.scene.add_mesh_simple(
                    f"/plan/ghost_{i}",
                    vertices=verts,
                    faces=faces,
                    color=ghost_color,
                    opacity=0.22,
                    wxyz=quat_xyzw_to_wxyz(quat),
                    position=tuple(np.asarray(xyz, dtype=float)),
                )

        self._draw_markers()

    def _draw_markers(self) -> None:
        dest = self.policy._destination_model
        if dest is not None:
            goal_robot = np.asarray(dest, dtype=np.float64) - self.frame_shift
            self.server.scene.add_icosphere(
                "/goal/marker",
                radius=0.025,
                color=COLOR_GOAL,
                opacity=0.45,
                position=tuple(goal_robot.astype(float)),
            )
        if self.latest_material_pose is not None:
            mat = pose_to_xyzw(self.latest_material_pose)[:3]
            self.server.scene.add_box(
                "/material/marker",
                dimensions=(0.04, 0.04, 0.04),
                color=COLOR_MATERIAL,
                position=tuple(mat.astype(float)),
            )

    def update_goal_object_pose(self) -> None:
        if not self.initialized or self.goal_poses_robot.shape[0] == 0:
            return
        if self.policy.done:
            return

        if self.current_goal_index >= self.goal_poses_robot.shape[0]:
            self._maybe_replan()
            if self.goal_poses_robot.shape[0] == 0:
                return
            if self.current_goal_index >= self.goal_poses_robot.shape[0]:
                self.current_goal_index = self.goal_poses_robot.shape[0] - 1

        tool_pose = pose_to_xyzw(deepcopy(self.latest_tool_pose))
        goal_pose = self.goal_poses_robot[self.current_goal_index]
        distance = keypoint_distance(tool_pose, goal_pose, self.object_scales)
        n = self.goal_poses_robot.shape[0]
        print(
            f"Distance: {distance:.4f}, goal {self.current_goal_index}/{n - 1}, "
            f"policy_done={self.policy.done}"
        )

        if distance < self.keypoint_success_threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                self.current_success_steps = 0
                self.current_goal_index += 1
                if self.current_goal_index >= self.goal_poses_robot.shape[0]:
                    info("Chunk complete; replanning")
                    self._maybe_replan()
                else:
                    info(f"Advanced to goal index {self.current_goal_index}")
            else:
                info(
                    f"Success threshold reached, at {self.current_success_steps}/{self.success_steps}"
                )

    def publish_goal_object_pose(self) -> None:
        if self.goal_object_pose_pub is None or self.goal_poses_robot.shape[0] == 0:
            return
        idx = min(self.current_goal_index, self.goal_poses_robot.shape[0] - 1)
        self.goal_object_pose_pub.publish(xyzw_to_pose(self.goal_poses_robot[idx]))

    def run(self) -> None:
        self._wait_for_observations()
        self._initialize_policy()

        while not rospy.is_shutdown():
            if self.latest_tool_pose is not None and self.latest_material_pose is not None:
                self.update_goal_object_pose()
                self.publish_goal_object_pose()
                self._draw_markers()
                self._set_status()
            self.rate.sleep()


@dataclass
class GoalPoseNodeClosedLoopVizArgs:
    tool_topic: str = "/robot_frame/current_object_pose"
    material_topic: str = "/robot_frame/current_object_pose_2"
    fixed_destination: tuple[float, float, float] = (-0.365, -0.056, 0.517)
    model: Optional[str] = None
    """closed_loop model_registry key. When unset, uses the package default model."""
    control_frame: str = "blue_brush"
    instruction: Optional[str] = None
    """Override prompt. When unset, the basic per-task default for ``control_frame`` is used."""
    y_shift: float = 0.8
    chunk_size: int = 5
    device: str = "cuda"
    success_threshold: float = 0.02
    success_steps: int = 1
    tool_pose_is_root: bool = True
    unflip_orientation: bool = True
    """Un-flip a replanned tool orientation that is ~180 deg flipped from the
    previous plan, keeping published goal poses continuous. Disable with
    ``--no-unflip-orientation``."""
    port: int = 8080
    """Viser server port (bind 0.0.0.0 for port-forward)."""
    arrow_scale: float = 0.04
    """Arrow length for normal / surface_dir (meters)."""
    publish: bool = True
    """Re-publish /robot_frame/goal_object_pose so this is a drop-in debug replacement."""


def main() -> None:
    if not _ROS_AVAILABLE:
        raise SystemExit(
            "rospy / geometry_msgs not found. Run this node from a sourced ROS "
            "environment (it subscribes to live tool/material PoseStamped topics)."
        )

    args: GoalPoseNodeClosedLoopVizArgs = tyro.cli(GoalPoseNodeClosedLoopVizArgs)

    instruction = (
        args.instruction
        if args.instruction
        else default_instruction_for_control_frame(args.control_frame)
    )
    info(
        f"model={args.model or '<default>'!r} "
        f"control_frame={args.control_frame!r} instruction={instruction!r}"
    )

    policy = load_closed_loop_policy(
        args.model,
        device=args.device,
        control_frame=args.control_frame,
        instruction=instruction,
        frame_shift=(0.0, args.y_shift, 0.0),
        chunk_size=args.chunk_size,
        tool_pose_is_root=args.tool_pose_is_root,
        unflip_orientation=args.unflip_orientation,
    )
    policy.set_destination(np.asarray(args.fixed_destination, dtype=np.float64))

    object_scales = np.array([0.141, 0.03025, 0.0271]) * 25

    try:
        node = GoalPoseNodeClosedLoopViz(
            policy,
            object_scales=object_scales,
            success_threshold=args.success_threshold,
            success_steps=args.success_steps,
            control_frame=args.control_frame,
            port=args.port,
            arrow_scale=args.arrow_scale,
            publish=args.publish,
            tool_topic=args.tool_topic,
            material_topic=args.material_topic,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
