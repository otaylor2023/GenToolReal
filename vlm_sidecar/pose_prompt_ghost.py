"""Legacy ghost-goal prompt profile for older alignment experiments."""

from __future__ import annotations

from pose_prompt import (
    WORLD_FRAME_AND_CAMERA_NOTE,
    applied_delta6_to_assistant_json,
    build_history_turn_user_text,
    format_object_pose_for_vlm_prompt,
    prompt_pose_values_2dp,
)

SYSTEM_PROMPT_FIVE_IMAGE_APPEND = (
    "**This turn’s user message has five RGB images in order:** "
    "(1) Full scene with solid tool and green ghost goal. "
    "(2) Current tool-only render. (3–5) +45° body-axis probes (+X,+Y,+Z). "
    "Use probes to align solid tool orientation to the green ghost."
)

FIVE_IMAGE_STACK_USER_APPEND = (
    "\n**Image order:** 1=full scene, 2=current tool-only, 3=+45deg body+X, 4=+45deg body+Y, 5=+45deg body+Z.\n"
)


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
) -> str:
    _ = (task_description or "").strip()
    tool = (tool_name or "").strip()
    tool_line = (
        f"**Tool:** {tool}. Align the solid tool to the green ghost goal pose.\n\n"
        if tool and tool != "-- Select --"
        else ""
    )
    session_ctx = (
        f"This is **turn {turn_index_1based}** with **{prior_turns_in_context}** prior turns in context.\n\n"
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
    pose_block = format_object_pose_for_vlm_prompt(object_pose_xyzw)
    base = (
        tool_line
        + "Goal: align solid tool to green ghost pose.\n\n"
        + history_summary_block
        + rotation_context_block
        + session_ctx
        + feedback_line
        + WORLD_FRAME_AND_CAMERA_NOTE
        + f"**Current** tool pose:\n{pose_block}\n"
    )
    if five_image_stack:
        base += FIVE_IMAGE_STACK_USER_APPEND
    return base

