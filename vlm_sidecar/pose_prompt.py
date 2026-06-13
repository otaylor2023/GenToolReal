"""Build VLM user text from object pose. Keep in sync with simtoolreal/llm_goal_env/vlm_pose_math.py."""

from __future__ import annotations

import json
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


def quat_xyzw_to_axis_angle_in_body_frame(quat_xyzw: np.ndarray) -> np.ndarray:
    """Map current orientation to axis–angle with axis **components in object body frame**."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    rot = SciRotation.from_quat(q)
    rotvec_world = rot.as_rotvec()
    n = float(np.linalg.norm(rotvec_world))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    R = rot.as_matrix()
    return (R.T @ rotvec_world.reshape(3)).astype(np.float64)


def format_object_pose_for_vlm_prompt(pose7_xyzw: np.ndarray) -> str:
    """Human-readable pose for the VLM: world xyz + body-frame axis–angle (6 numbers, 2 dp)."""
    p = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    xyz = p[:3]
    # Show axis-angle components in degrees to match sidecar output contract.
    r_body_deg = np.degrees(quat_xyzw_to_axis_angle_in_body_frame(p[3:7]))
    return (
        f"Position in **world** frame (meters): x={xyz[0]:.2f}, y={xyz[1]:.2f}, z={xyz[2]:.2f}\n"
        f"Orientation as **axis–angle in object body frame**: "
        f"rx={r_body_deg[0]:.2f}, ry={r_body_deg[1]:.2f}, rz={r_body_deg[2]:.2f}\n"
        f"(rotation axis direction = normalize([rx,ry,rz]) if norm > 0; angle in degrees = norm([rx,ry,rz]))"
    )


def prompt_pose_values_2dp(pose7_xyzw: np.ndarray) -> dict:
    """Return exactly the rounded pose numbers shown to the model."""
    p = np.asarray(pose7_xyzw, dtype=np.float64).reshape(7)
    xyz = p[:3]
    r_body_deg = np.degrees(quat_xyzw_to_axis_angle_in_body_frame(p[3:7]))
    return {
        "x_m": round(float(xyz[0]), 2),
        "y_m": round(float(xyz[1]), 2),
        "z_m": round(float(xyz[2]), 2),
        "rx_deg": round(float(r_body_deg[0]), 2),
        "ry_deg": round(float(r_body_deg[1]), 2),
        "rz_deg": round(float(r_body_deg[2]), 2),
    }


def format_applied_delta6_line(delta6: np.ndarray) -> str:
    """One-line summary of an applied motion_delta (2 dp) for history text."""
    d = np.asarray(delta6, dtype=np.float64).reshape(6)
    return (
        f"dx={d[0]:.2f}, dy={d[1]:.2f}, dz={d[2]:.2f} (world, m); "
        f"rx={d[3]:.2f}, ry={d[4]:.2f}, rz={d[5]:.2f} (body axis–angle, deg)"
    )


def build_history_turn_user_text(
    *,
    step_index_1based: int,
    object_pose_xyzw: np.ndarray,
    delta6_applied: np.ndarray,
) -> str:
    """User text for a **past** interaction (paired with that step's RGB in the message)."""
    pose_block = format_object_pose_for_vlm_prompt(object_pose_xyzw)
    delta_line = format_applied_delta6_line(delta6_applied)
    return (
        f"--- **History turn {step_index_1based}** (older than the current user message) ---\n"
        "Structure: the **assistant** message **immediately above** this block is **your JSON from that step** "
        "(already applied in sim). This **user** block pairs this RGB with the outcome **after** that JSON’s "
        "``motion_delta`` was integrated.\n"
        "Use this pair when planning **future** ``drx,dry,drz``: did that step’s rotation move the solid tool "
        "toward the intended sweep objective, or should the next turn correct axis / angle?\n"
        "Tool pose in this screenshot (world + body axis–angle, deg):\n"
        f"{pose_block}\n"
        f"``motion_delta`` taken from your assistant JSON on that step (we applied exactly): {delta_line}\n"
    )


WORLD_FRAME_AND_CAMERA_NOTE = (
    "**World frame (matches pose numbers and ``dx,dy,dz``):** **+Z** is up (vertical); **+X** and **+Y** are "
    "horizontal on the table (meters). **Default RGB view:** camera sits near **−Y** looking toward **+Y**, "
    "with **+Z** upward on screen—so **+dz** is “up” in the image and **+dy** often moves deeper into the "
    "workspace (approximate; use image + pose numbers together).\n\n"
)

# Appended to system prompt when the client sends five images (main + four probes).
SYSTEM_PROMPT_FIVE_IMAGE_APPEND = (
    "**This turn’s user message has five RGB images in order:** "
    "(1) Full scene (solid tool, arm/table, bin, debris). "
    "(2) Zoomed render of **only** the tool mesh at the **current / unmodified** pose "
    "(this is the object’s actual current rotation; same orientation as the numbers below). "
    "(3–5) The same tool **position** with **hypothetical** orientations: +45° about the tool’s body **+X**, "
    "then +45° about body **+Y**, then +45° about body **+Z** (each is a visual hint, not the live sim). "
    "Treat (2–5) as **rotation reference views** for this exact tool geometry: use them to infer how "
    "body-axis rotations change appearance, then choose ``rx,ry,rz`` for the live scene in (1). "
    "Compare (2–5) to the desired brush orientation in (1) when choosing rotation. "
    "When this turn requests rotation-only behavior, output rotational components only and ignore translation."
)

FIVE_IMAGE_STACK_USER_APPEND = (
    "\n**Image order in this message:** "
    "**1** = full scene. **2** = zoomed tool-only at **current / unmodified** pose "
    "(actual current object rotation). "
    "**3** = tool-only, current position, orientation rotated **+45°** about body **+X**. "
    "**4** = same, **+45°** about body **+Y**. **5** = same, **+45°** about body **+Z**. "
    "Use (1) for scene layout and sweep context; use (2–5) as rotation references to judge how body-axis "
    "rotation relates to the goal.\n"
)


def build_llm_user_text(
    *,
    object_pose_xyzw: np.ndarray,
    tool_name: str = "",
    task_description: str = "",
    turn_index_1based: int = 1,
    prior_turns_in_context: int = 0,
    five_image_stack: bool = False,
    render_feedback_note: str = "",
    phase: str = "direction",
    history_summary: str = "",
    rotation_context_json: str = "",
) -> str:
    """Text for the VLM: optional **tool** label + **current** pose (no task wording).

    ``task_description`` is ignored (API compatibility).
    ``turn_index_1based`` / ``prior_turns_in_context`` describe the multi-turn session (see system prompt).
    When ``five_image_stack`` is True, the user message includes five images (see ``FIVE_IMAGE_STACK_USER_APPEND``).
    """
    _ = (task_description or "").strip()
    tool = (tool_name or "").strip()
    if tool in ("", "-- Select --"):
        tool = ""
    tool_line = ""
    if tool:
        tool_line = (
            f"**Tool:** {tool} — names the rigid body only (solid mesh colors vary). "
            "Use only the solid tool and scene objects (balls/bin/table) as task cues.\n\n"
        )
    pose_block = format_object_pose_for_vlm_prompt(object_pose_xyzw)

    if prior_turns_in_context > 0:
        session_ctx = (
            f"This is **turn {turn_index_1based}** with **{prior_turns_in_context}** prior chat turns in context.\n\n"
        )
    else:
        session_ctx = "This is **turn 1**.\n\n"

    feedback_line = ""
    note = (render_feedback_note or "").strip()
    if note:
        feedback_line = (
            f"**Image framing feedback from previous position preview:** {note} "
            "If you adjust motion this turn, prefer changes that keep the tool fully in frame.\n\n"
        )

    phase = (phase or "direction").strip().lower()
    if phase not in ("plan", "rotation", "rotation_review", "direction"):
        phase = "direction"

    phase_block = ""
    if phase == "plan":
        phase_block = (
            "**Planning stage only**\n"
            "- You are a brush pose planner only.\n"
            "- Decide a detailed step-by-step strategy.\n"
            "- Do not output numeric actions.\n"
            "- Do not output or reason about dx,dy,dz,drx,dry,drz.\n"
            "- Return exactly one JSON object with: `phase`, `reasoning_summary`, `plan_steps`.\n"
            "- Use `phase=\"plan\"`.\n\n"
        )
    elif phase in ("rotation", "rotation_review"):
        phase_block = (
            "**Rotation stage only**\n"
            "- You are only selecting desired brush orientation.\n"
            "- Output rotation delta only: drx,dry,drz.\n"
            "- Do not provide dx,dy,dz (ignored).\n"
            "- drx,dry,drz are BODY-frame axis-angle components in degrees.\n"
            "- Auxiliary refs: aux0=0deg baseline, aux1=+45deg body+X, aux2=+45deg body+Y, aux3=+45deg body+Z.\n"
            "- Return exactly one JSON object with: `phase`, `reasoning_summary`, `rotation_satisfied`, `motion_delta`.\n"
            "- Use `phase=\"rotation\"`.\n"
        )
        if phase == "rotation_review":
            phase_block += (
                "- This is a rotation review turn: judge proposed orientation render.\n"
                "- If orientation is correct, set rotation_satisfied=true.\n"
                "- Else set false and provide new rotation delta from original reference.\n"
            )
        phase_block += "\n"
    else:
        phase_block = (
            "**Direction stage only**\n"
            "- Rotation is locked and handled separately.\n"
            "- Output translation delta only: dx,dy,dz.\n"
            "- Any rotation values are ignored.\n"
            "- World movement axes: +Z is up, +X/+Y are table-plane.\n"
            "- Fixed camera: position=(0.0,-1.0,1.0), look_at=(0.0,0.0,0.5).\n"
            "- In image terms: +Y moves deeper into scene, -Y toward camera, +Z upward.\n"
            "- Return exactly one JSON object with: `phase`, `reasoning_summary`, `motion_delta`.\n"
            "- Use `phase=\"direction\"`.\n\n"
        )

    history_summary_block = ""
    if (history_summary or "").strip():
        history_summary_block = (
            "**Long-horizon summary of earlier conversation (compressed):**\n"
            f"{history_summary.strip()}\n\n"
        )

    rotation_context_block = ""
    if (rotation_context_json or "").strip():
        rotation_context_block = (
            "**Rotation refinement context (JSON):**\n"
            f"{rotation_context_json.strip()}\n\n"
        )

    shared_goal = (
        "Task: sweep clustered red balls into the green bin using the brush. "
        "Focus only on desired brush pose (position/rotation), not low-level execution.\n\n"
    )
    base = (
        tool_line
        + phase_block
        + shared_goal
        + history_summary_block
        + rotation_context_block
        + session_ctx
        + feedback_line
    )
    if phase == "direction":
        base += (
            "**World frame for translation deltas (`dx,dy,dz`):** +Z is vertical up; +X and +Y are horizontal.\n"
            "**Fixed camera:** position=(0.0,-1.0,1.0), look_at=(0.0,0.0,0.5).\n"
            "Therefore in image terms: +Y moves deeper into the scene, -Y moves toward camera, +Z moves upward.\n\n"
        )
    elif phase != "plan":
        base += WORLD_FRAME_AND_CAMERA_NOTE
        base += (
            "**Current** brush pose for this turn:\n"
            f"{pose_block}\n"
        )
    if five_image_stack:
        base = base + FIVE_IMAGE_STACK_USER_APPEND
    return base


def applied_delta6_to_assistant_json(delta6: np.ndarray) -> str:
    """Minimal assistant reply replayed into chat history (must match output contract)."""
    d = np.asarray(delta6, dtype=np.float64).reshape(6)
    row = {
        "reasoning_summary": "",
        "motion_delta": {
            "dx": round(float(d[0]), 2),
            "dy": round(float(d[1]), 2),
            "dz": round(float(d[2]), 2),
            "rx": round(float(d[3]), 2),
            "ry": round(float(d[4]), 2),
            "rz": round(float(d[5]), 2),
        },
    }
    return json.dumps(row, separators=(",", ":"))
