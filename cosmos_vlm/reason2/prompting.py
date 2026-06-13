from __future__ import annotations

from pathlib import Path


SYSTEM_PROMPT = (
    "You are a planning model, not a video generator. "
    "Your plan will be consumed by Cosmos Predict 2.5 (image-to-video), which will execute it visually. "
    "Given a workspace image, generate a detailed future trajectory plan for an embodied manipulator. "
    "Be explicit, ordered, physically plausible, and easy for a video model to follow. "
    "Preserve object identity and keep brush geometry/scale consistent across the sequence. "
    "Respect workspace physics constraints: do not move through solid objects, and rise/move around obstacles when needed."
)


def build_reason_messages(image_path: Path, task_description: str | None = None) -> list[dict]:
    task = task_description or "Sweep the red balls into the green bin."
    user_text = (
        f"Task: {task}\n\n"
        "Primary objective: sweep all visible red balls into the green bin.\n\n"
        "You are producing execution steps for a downstream image-to-video model. "
        "Do not describe video rendering behavior; describe only executable physical actions.\n\n"
        "Constraints:\n"
        "- Pick up and control the brush by its handle (not the brush head/bristles).\n"
        "- Keep brush size/proportions constant.\n"
        "- The green bin is stationary and must not be moved, pushed, rotated, or repositioned.\n"
        "- No object penetration: the brush and gripper must not pass through objects.\n"
        "- If blocked, lift and move above/around obstacles before continuing.\n"
        "- Use smooth, temporally coherent step transitions.\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "scene_summary": "short scene description",\n'
        '  "trajectory_steps": ["step 1", "step 2", "..."],\n'
        '  "motion_prompt_for_video": "single detailed motion prompt suitable for image-to-video generation"\n'
        "}\n"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": user_text},
            ],
        },
    ]

