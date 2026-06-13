from __future__ import annotations


def build_predict25_self_planned_debug_prompt() -> str:
    return (
        "You are generating a short, controlled debug video from a single fixed workspace frame.\n"
        "Do not infer a new camera shot. Preserve the exact input framing.\n\n"
        "Absolute constraints (highest priority):\n"
        "- CAMERA MUST BE COMPLETELY STATIC FOR ALL FRAMES, INCLUDING THE FINAL FRAME: no zoom, no crop change, no pan, no tilt, no orbit, no reframing.\n"
        "- ROBOT/HAND IDENTITY MUST BE INVARIANT: preserve the same hand/end-effector seen in the input frame.\n"
        "- NEVER replace the hand with a different gripper/tool type. No end-effector swap is allowed.\n"
        "- Do not morph fingers into a claw, pincer, or mechanical gripper; keep human-like finger proportions.\n"
        "- BRUSH IDENTITY MUST BE INVARIANT: preserve a long straight handle with the same visible length/thickness, plus the same brush head shape and size.\n"
        "- NEVER shorten, remove, bend, or morph the brush handle.\n"
        "- Do not redraw, replace, or swap the brush for a different tool; treat it as a rigid object.\n"
        "- During grasping, do not shrink the handle or clip it out of frame; keep the full handle visible whenever possible.\n"
        "- Keep all static scene objects fixed in place (especially the green bin and table background).\n"
        "- Respect collisions and physical constraints: no penetration, no teleportation, no impossible geometry changes.\n\n"
        "Debug objective:\n"
        "Create a short 4-second clip that performs one simple, smooth sweep toward placing red balls into the green bin,\n"
        "with minimal articulation and no dramatic motion.\n\n"
        "Execution plan (use exactly this sequence):\n"
        "1) Move gripper to brush handle and establish stable grasp without changing brush geometry.\n"
        "2) Lift slightly and translate brush to just behind the red balls.\n"
        "3) Perform one smooth sweep arc pushing balls toward/into the green bin.\n"
        "4) Stop motion and hold final pose for the last portion of the clip with zero camera or object drift.\n\n"
        "End-of-clip lock:\n"
        "- In the final segment, freeze camera and scene framing exactly; do not add any late drift or reframing.\n\n"
        "If any motion conflicts with camera lock or identity consistency, preserve camera/identity and reduce motion."
    )

