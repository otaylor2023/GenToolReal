"""
Quick test of Gemini Robotics-ER 1.6 for embodied reasoning — specifically
grasp prediction + trajectory planning on a scene image.

Scene (00_main.png): a tabletop with a robot arm, a broom, a white platform
with red cherries on top, and a green bin off to the right.

The script:
  1. Loads GEMINI_API_KEY from .env (same folder)
  2. Loads 00_main.png
  3. Asks the model to:
       (a) point to the key objects (broom, cherries, green bin)
       (b) predict a top-down grasp on the broom handle
       (c) plan a trajectory for the brush to sweep the cherries into the green bin
  4. Parses the JSON from each response
  5. Saves all raw + parsed responses to gemini_robotics_er_output.json
  6. Draws all three on the image and saves gemini_robotics_er_output.png

Points are returned in [y, x] format normalized 0..1000 (the Robotics-ER
convention). We denormalize to pixel coords before drawing.

Run:
    pip install google-genai python-dotenv pillow
    python gemini_robotics_er.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
IMAGE_PATH = HERE / "00_main.png"
OUTPUT_IMAGE = HERE / "gemini_robotics_er_output.png"
OUTPUT_JSON = HERE / "gemini_robotics_er_output.json"

# The embodied-reasoning model. ER 1.6 is the latest as of April 2026; fall
# back to 1.5 if the preview alias ever changes.
MODEL = "gemini-robotics-er-1.6-preview"

# ER uses thinking by default. A small budget is plenty for pointing; bump
# it for multi-step reasoning like trajectory planning.
THINK_SMALL = 512
THINK_LARGE = 2048


# ---------------------------------------------------------------------------
# Prompts — the model expects specific JSON schemas for each task type.
# Formats below follow the Robotics-ER cookbook conventions.
# ---------------------------------------------------------------------------

POINT_PROMPT = """\
Point to the following items in the image: the broom/brush, the red cherries
on the white platform, and the empty green bin.

The answer should follow the JSON format:
[{"point": [y, x], "label": <name>}, ...]

The points are in [y, x] format normalized to 0-1000. Return only JSON."""

GRASP_PROMPT = """\
Predict a top-down grasp for picking up the broom by its handle.

Return a JSON object with two points: one at each end of the gripper's closing
line across the handle. The grasp should be perpendicular to the handle's long
axis so two-finger closure captures it securely.

Format:
{"grasp": [[y1, x1], [y2, x2]], "label": "broom handle grasp"}

Points are in [y, x] format normalized to 0-1000. Return only JSON."""

TRAJECTORY_PROMPT = """\
Plan a collision-free trajectory for the brush to sweep the cherries from the white
platform into the empty green bin. Produce 12 waypoints, labeled in order
from "0" (start, at the brush handle) to "11" (end, above the green bin). Keep
the path arcing upward so it clears the platform edge. Assume the brush rotates as required to sweep the cherries into the green bin, just focus on the handle position.

The answer should follow the JSON format:
[{"point": [y, x], "label": "0"},
 {"point": [y, x], "label": "1"},
 ...,
 {"point": [y, x], "label": "11"}]

Points are in [y, x] format normalized to 0-1000. Return only JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_json(text: str):
    """Strip code fences and parse the first JSON blob in the model's reply."""
    s = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Find the outermost JSON array or object
    start = min(
        (i for i in (s.find("["), s.find("{")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError(f"No JSON found in response:\n{text}")
    # Try parsing progressively — in case there's trailing commentary
    for end in range(len(s), start, -1):
        try:
            return json.loads(s[start:end])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from:\n{text}")


def denorm(point, width, height):
    """Convert [y, x] in 0..1000 to (px, py) pixel coords."""
    y, x = point
    return (int(x / 1000 * width), int(y / 1000 * height))


def call_er(client, prompt, image_bytes, thinking_budget):
    """Send prompt + image to Gemini Robotics-ER.

    Returns (parsed_json, raw_text). If JSON parsing fails, parsed_json is
    None and the raw text is still returned so it can be saved for debugging.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
        ),
    )
    text = response.text or ""
    print("\n--- raw response ---")
    print(text)
    print("--- end raw ---\n")
    try:
        return extract_json(text), text
    except ValueError as e:
        print(f"[warn] JSON parse failed: {e}")
        return None, text


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize(image_path, points, grasp, trajectory, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    # --- Object points (cyan) ---
    for item in points or []:
        px, py = denorm(item["point"], w, h)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(0, 200, 255, 255))
        draw.text((px + 9, py - 7), item.get("label", ""), fill=(0, 200, 255), font=font)

    # --- Grasp line (magenta) ---
    if grasp:
        g = grasp.get("grasp") if isinstance(grasp, dict) else grasp
        if g and len(g) >= 2:
            p1 = denorm(g[0], w, h)
            p2 = denorm(g[1], w, h)
            draw.line([p1, p2], fill=(255, 0, 200, 255), width=4)
            for p in (p1, p2):
                draw.ellipse(
                    (p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5),
                    fill=(255, 0, 200, 255),
                )

    # --- Trajectory (yellow polyline + numbered dots) ---
    if trajectory:
        traj_pts = [denorm(item["point"], w, h) for item in trajectory]
        if len(traj_pts) >= 2:
            draw.line(traj_pts, fill=(255, 230, 0, 220), width=3)
        for i, (px, py) in enumerate(traj_pts):
            draw.ellipse(
                (px - 5, py - 5, px + 5, py + 5),
                fill=(255, 230, 0, 255),
                outline=(0, 0, 0, 255),
            )
            draw.text((px + 7, py + 4), str(i), fill=(0, 0, 0), font=font)

    img.save(out_path)
    print(f"Saved visualization -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    load_dotenv(HERE / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Missing GEMINI_API_KEY in .env")

    if not IMAGE_PATH.exists():
        sys.exit(f"Image not found: {IMAGE_PATH}")

    image_bytes = IMAGE_PATH.read_bytes()
    client = genai.Client(api_key=api_key)

    print(f"Model: {MODEL}")
    print(f"Image: {IMAGE_PATH.name} ({len(image_bytes):,} bytes)\n")

    print("=" * 60)
    print("1) Pointing at key objects")
    print("=" * 60)
    points, points_raw = call_er(client, POINT_PROMPT, image_bytes, THINK_SMALL)
    print(json.dumps(points, indent=2))

    print("\n" + "=" * 60)
    print("2) Grasp prediction — broom handle")
    print("=" * 60)
    grasp, grasp_raw = call_er(client, GRASP_PROMPT, image_bytes, THINK_SMALL)
    print(json.dumps(grasp, indent=2))

    print("\n" + "=" * 60)
    print("3) Trajectory — cherries -> green bin")
    print("=" * 60)
    trajectory, trajectory_raw = call_er(
        client, TRAJECTORY_PROMPT, image_bytes, THINK_LARGE
    )
    print(json.dumps(trajectory, indent=2))

    # --- Save all outputs to a single JSON file ---
    output = {
        "model": MODEL,
        "image": IMAGE_PATH.name,
        "points": {"parsed": points, "raw": points_raw},
        "grasp": {"parsed": grasp, "raw": grasp_raw},
        "trajectory": {"parsed": trajectory, "raw": trajectory_raw},
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nSaved JSON output -> {OUTPUT_JSON}")

    # --- Draw overlays on the image and save ---
    visualize(IMAGE_PATH, points, grasp, trajectory, OUTPUT_IMAGE)


if __name__ == "__main__":
    main()
