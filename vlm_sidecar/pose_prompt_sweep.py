"""Sweep-focused prompt profile: no ghost-goal language."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from pose_prompt import (
    WORLD_FRAME_AND_CAMERA_NOTE,
    applied_delta6_to_assistant_json,
    build_history_turn_user_text,
)

SYSTEM_PROMPT_FIVE_IMAGE_APPEND = (
    "**This turn’s user message has five RGB images in order:** "
    "(1) Full scene (solid tool, arm/table, bin, debris). "
    "(2) Zoomed render of **only** the tool mesh at the **current / unmodified** pose. "
    "(3–5) The same tool **position** with hypothetical orientations: +45° about body +X, +Y, +Z. "
    "Use (2–5) as rotation references for choosing drx/dry/drz."
)

FIVE_IMAGE_STACK_USER_APPEND = (
    "\n**Image order in this message:** "
    "1=full scene, 2=current tool-only, 3=+45deg body+X, 4=+45deg body+Y, 5=+45deg body+Z.\n"
)


def _relative_body_axis_angle_deg(
    pose7_xyzw: np.ndarray, orientation_reference_quat_xyzw: np.ndarray
) -> np.ndarray:
    p = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    q_cur = np.asarray(p[3:7], dtype=np.float64).reshape(4)
    q_ref = np.asarray(orientation_reference_quat_xyzw, dtype=np.float64).reshape(4)
    r_cur = SciRotation.from_quat(q_cur)
    r_ref = SciRotation.from_quat(q_ref)
    r_rel = r_ref.inv() * r_cur
    rv_world = r_rel.as_rotvec()
    n = float(np.linalg.norm(rv_world))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    # Express relative axis-angle components in current tool body frame.
    R_cur = r_cur.as_matrix()
    rv_body = R_cur.T @ rv_world.reshape(3)
    return np.degrees(rv_body)


def format_object_pose_for_vlm_prompt(
    pose7_xyzw: np.ndarray, orientation_reference_quat_xyzw: np.ndarray
) -> str:
    p = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    xyz = p[:3]
    r_body_deg = _relative_body_axis_angle_deg(p, orientation_reference_quat_xyzw)
    return (
        f"Position in **world** frame (meters): x={xyz[0]:.2f}, y={xyz[1]:.2f}, z={xyz[2]:.2f}\n"
        f"Orientation as **axis–angle in object body frame**: "
        f"rx={r_body_deg[0]:.2f}, ry={r_body_deg[1]:.2f}, rz={r_body_deg[2]:.2f}\n"
        f"(rotation axis direction = normalize([rx,ry,rz]) if norm > 0; angle in degrees = norm([rx,ry,rz]))"
    )


def prompt_pose_values_2dp(
    pose7_xyzw: np.ndarray, *, orientation_reference_quat_xyzw: np.ndarray
) -> dict:
    p = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    xyz = p[:3]
    r_body_deg = _relative_body_axis_angle_deg(p, orientation_reference_quat_xyzw)
    return {
        "x_m": round(float(xyz[0]), 2),
        "y_m": round(float(xyz[1]), 2),
        "z_m": round(float(xyz[2]), 2),
        "rx_deg": round(float(r_body_deg[0]), 2),
        "ry_deg": round(float(r_body_deg[1]), 2),
        "rz_deg": round(float(r_body_deg[2]), 2),
    }


def build_llm_user_text(
    *,
    object_pose_xyzw,
    tool_name: str = "",
    task_description: str = "",
    turn_index_1based: int = 1,
    prior_turns_in_context: int = 0,
    five_image_stack: bool = False,
    render_feedback_note: str = "",
    phase: str = "direction",
    history_summary: str = "",
    rotation_context_json: str = "",
    orientation_reference_quat_xyzw=None,
) -> str:
    _ = (task_description or "").strip()
    tool = (tool_name or "").strip()
    tool_line = (
        f"**Tool:** {tool}. Use only real scene objects (brush, balls, bin, table) as cues.\n\n"
        if tool and tool != "-- Select --"
        else ""
    )
    phase = (phase or "direction").strip().lower()
    if phase not in ("plan", "rotation", "rotation_review", "direction"):
        phase = "direction"
    session_ctx = (
        f"This is **turn {turn_index_1based}** with **{prior_turns_in_context}** prior chat turns in context.\n\n"
        if prior_turns_in_context > 0
        else "This is **turn 1**.\n\n"
    )
    feedback_line = (
        f"**Image framing feedback:** {(render_feedback_note or '').strip()}\n\n"
        if (render_feedback_note or "").strip()
        else ""
    )
    history_summary_block = (
        f"**Long-horizon summary:**\n{history_summary.strip()}\n\n"
        if (history_summary or "").strip()
        else ""
    )
    rotation_context_block = (
        f"**Rotation refinement context (JSON):**\n{rotation_context_json.strip()}\n\n"
        if (rotation_context_json or "").strip()
        else ""
    )
    ref_q = (
        np.asarray(orientation_reference_quat_xyzw, dtype=np.float64).reshape(4)
        if orientation_reference_quat_xyzw is not None
        else np.asarray(object_pose_xyzw, dtype=np.float64).reshape(7)[3:7]
    )
    pose_block = format_object_pose_for_vlm_prompt(object_pose_xyzw, ref_q)
    shared_goal = (
        "Task: sweep clustered red balls into the green bin using the brush. "
        "Focus on desired brush motion, not low-level execution.\n\n"
    )
    base = tool_line + shared_goal + history_summary_block + rotation_context_block + session_ctx + feedback_line
    if phase == "direction":
        base += (
            "**Direction stage only**\n"
            "- Use the plan stage that matches the current sweep progress (approach / engage / push / collect / reset).\n"
            "- Translate according to that stage intent (e.g., get behind balls, maintain contact, drive toward bin, recover).\n"
            "- Keep orientation fixed; do not reinterpret rotation during this stage.\n"
            "- Output translation only: dx,dy,dz (rotation ignored).\n"
            "- Return one JSON object with `phase`, `reasoning_summary`, `motion_delta` and use `phase=\"direction\"`.\n\n"
            "**World frame:** +Z up, +X/+Y table-plane. Fixed camera: pos=(0,-1,1), look_at=(0,0,0.5).\n\n"
        )
    elif phase == "plan":
        base += (
            "**Planning stage only**\n"
            "- Produce a clear multi-stage sweep plan from current state to balls-in-bin completion.\n"
            "- For each stage, describe BOTH:\n"
            "  1) desired brush orientation (qualitative, no numeric angles), and\n"
            "  2) desired translation behavior (qualitative path/contact intent, no numeric deltas).\n"
            "- Use concrete stage labels such as: approach, orient, engage, push, collect, reset.\n"
            "- Be explicit about when orientation should stay fixed vs. when it should change.\n"
            "- Do not output numeric dx/dy/dz/drx/dry/drz in this phase.\n"
            "- Return one JSON object with `phase`, `reasoning_summary`, `plan_steps` and use `phase=\"plan\"`.\n\n"
        )
    else:
        base += (
            "**Rotation stage only**\n"
            "- Refer to the relevant plan stage and choose rotation that matches its stated orientation goal.\n"
            "- Do not solve translation here; only orientation alignment for the current stage.\n"
            "- Output rotation delta only: drx,dry,drz (body-frame axis-angle degrees).\n"
            "- Return one JSON object with `phase`, `reasoning_summary`, `rotation_satisfied`, `motion_delta`.\n\n"
        )
        if phase == "rotation_review":
            base += (
                "**Rotation review clarification**\n"
                "- Any new rotation delta you provide is applied from the `original_pose_xyzw` in the review context.\n"
                "- Do not chain from the previous proposal; always correct relative to that original reference.\n\n"
            )
        base += WORLD_FRAME_AND_CAMERA_NOTE
        base += f"**Current** brush pose:\n{pose_block}\n"
    if five_image_stack:
        base += FIVE_IMAGE_STACK_USER_APPEND
    return base

