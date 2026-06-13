from __future__ import annotations

from typing import Sequence


def build_predict25_prompt_from_trajectory(
    trajectory_steps: Sequence[str],
    motion_prompt_for_video: str | None = None,
) -> str:
    steps = [s.strip() for s in trajectory_steps if isinstance(s, str) and s.strip()]
    if not steps:
        raise ValueError("trajectory_steps is empty; cannot build video prompt")
    motion = (motion_prompt_for_video or "").strip()

    numbered_steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))

    sections = [
        "Your goal is to visualize exactly the trajectory provided by the planner, as faithfully as possible.\n"
        "Follow the listed trajectory steps in order and preserve temporal continuity between steps.\n\n"
        "Hard constraints (absolute, non-negotiable):\n"
        "- CAMERA LOCK IS TOP PRIORITY. Keep the exact same camera intrinsics/extrinsics for every frame.\n"
        "- Absolutely no zoom-in or zoom-out, no crop change, no reframing, no panning, no tilting, no orbiting, and no viewpoint drift.\n"
        "- Maintain identical framing/scale of static scene elements across frames.\n"
        "- The tool shape and size must remain completely consistent at all times: no growth, shrinkage, deformation, or identity drift.\n"
        "- Keep the green bin fixed in place; do not move, rotate, or reposition it.\n"
        "- The brush is controlled via its handle; do not depict handling by the bristles/head.\n"
        "- Respect physical collisions: no object penetration, teleportation, or impossible motion.\n"
        "- Use smooth, physically plausible motion and stable object identities.\n\n"
        "Priority rule: if any requested motion conflicts with camera lock, preserve camera lock and adjust object motion instead.\n\n"
    ]
    if motion:
        sections.append(f"Planner motion intent:\n{motion}\n\n")
    sections.append(
        "Trajectory steps to execute:\n"
        f"{numbered_steps}\n\n"
        "Render only the action implied by these steps with consistent appearance and physics."
    )
    return "".join(sections)

