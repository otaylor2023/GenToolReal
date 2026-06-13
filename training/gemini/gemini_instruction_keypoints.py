"""Generate instruction/tool-keypoint pairs and scene keypoints with Gemini.

This module:
1) Sends a scene image to Gemini Robotics-ER
2) Requests JSON with:
   - instructions: [{"instruction": str, "tool_keypoint": str}, ...]
   - keypoints: [{"label": str, "point": [y, x]}, ...] in 0..1000
3) Validates and saves the JSON artifact
4) Draws labeled keypoints on the image

Usage:
  python isaacsim_envs/gemini_instruction_keypoints.py \
    --image 00_main.png \
    --num-instructions 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from google import genai
from google.genai import types


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_NUM_INSTRUCTIONS = 50


@dataclass
class GenerationResult:
    model: str
    image: str
    instructions: List[Dict[str, str]]
    keypoints: List[Dict[str, Any]]
    raw_text: str


@dataclass
class ScoreResult:
    model: str
    image: str
    instruction: str
    tool_visible: bool
    position_score: float
    orientation_score: float
    notes: str
    raw_text: str


def _load_api_key_from_repo_env() -> str:
    # Reuse sidecar logic so we follow existing .env conventions.
    sidecar_dir = REPO_ROOT / "vlm_sidecar"
    if str(sidecar_dir) not in sys.path:
        sys.path.insert(0, str(sidecar_dir))
    from gemini_backend import _hydrate_gemini_env_from_dotenv, resolve_gemini_api_key

    _hydrate_gemini_env_from_dotenv()
    api_key = resolve_gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY in .env")
    return api_key


def _extract_json_blob(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")
    for end in range(len(s), start, -1):
        try:
            obj = json.loads(s[start:end])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to parse JSON object from model response")


def _validate_point(point: Any, label: str) -> List[float]:
    if not isinstance(point, Sequence) or len(point) != 2:
        raise ValueError(f"Keypoint '{label}' must have [y, x]")
    y, x = float(point[0]), float(point[1])
    if y < 0 or y > 1000 or x < 0 or x > 1000:
        raise ValueError(f"Keypoint '{label}' must be normalized to 0..1000")
    return [y, x]


def _validate_response(obj: Dict[str, Any], expected_count: int) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    instructions_raw = obj.get("instructions")
    keypoints_raw = obj.get("keypoints")
    if not isinstance(instructions_raw, list):
        raise ValueError("Response missing 'instructions' list")
    if not isinstance(keypoints_raw, list):
        raise ValueError("Response missing 'keypoints' list")
    if len(instructions_raw) != expected_count:
        raise ValueError(
            f"Expected {expected_count} instructions, got {len(instructions_raw)}"
        )

    keypoint_labels: set[str] = set()
    keypoints: List[Dict[str, Any]] = []
    for item in keypoints_raw:
        if not isinstance(item, dict):
            raise ValueError("Each keypoint entry must be an object")
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("Each keypoint requires a non-empty label")
        point = _validate_point(item.get("point"), label)
        keypoint_labels.add(label)
        keypoints.append({"label": label, "point": point})

    instructions: List[Dict[str, str]] = []
    for idx, item in enumerate(instructions_raw):
        if not isinstance(item, dict):
            raise ValueError(f"instructions[{idx}] must be an object")
        instruction = str(item.get("instruction", "")).strip()
        tool_keypoint = str(item.get("tool_keypoint", "")).strip()
        if not instruction:
            raise ValueError(f"instructions[{idx}].instruction is required")
        if not tool_keypoint:
            raise ValueError(f"instructions[{idx}].tool_keypoint is required")
        if tool_keypoint not in keypoint_labels:
            raise ValueError(
                f"instructions[{idx}] tool_keypoint '{tool_keypoint}' not found in keypoints"
            )
        instructions.append(
            {
                "instruction": instruction,
                "tool_keypoint": tool_keypoint,
            }
        )

    return instructions, keypoints


def _prompt(num_instructions: int) -> str:
    return f"""Analyze this tabletop manipulation scene with many tools/objects.

Generate exactly {num_instructions} instruction variants and a scene keypoint catalog.

Output ONLY valid JSON with this exact schema:
{{
  "instructions": [
    {{
      "instruction": "move X object to Y relative to Z object in A orientation",
      "tool_keypoint": "name_of_tool_keypoint_label"
    }}
  ],
  "keypoints": [
    {{
      "label": "semantic_keypoint_name",
      "point": [y, x]
    }}
  ]
}}

Requirements:
- instructions must be diverse variants of object placement/manipulation with relative position + orientation intent
- examples: "move the brush above the red block with bristles facing down", "put the head of the screwdriver on top of the box"
- "tool_keypoint" must refer to one of the keypoint labels you return
- keypoints should include tool parts and relevant object anchors when visible (for example screwdriver_head, screwdriver_tip, screwdriver_handle, red_block, box_of_cheetos)
- each point is [y, x], normalized to 0..1000
- include enough keypoints for all referenced objects/tool parts
- do not include markdown or any explanation, only JSON
"""


def _score_prompt(instruction: str, target_point_yx: Sequence[float] | None) -> str:
    target_text = "null"
    if target_point_yx is not None:
        target_text = json.dumps([float(target_point_yx[0]), float(target_point_yx[1])])
    return f"""You are grading the final tool placement in a tabletop robot workspace image.

You are given:
- the original instruction
- an image where the placed tool is visible in the workspace render
- a red dot marking the intended target position (if present)

Instruction:
{instruction}

Target point [y, x] normalized 0..1000 (if provided externally):
{target_text}

Return ONLY valid JSON with this exact schema:
{{
  "tool_visible": true,
  "position_score": 0.00,
  "orientation_score": 0.00,
  "notes": "short explanation"
}}

Scoring rules:
- tool_visible:
  - false if the tool is fully off-screen, severely clipped, buried/intersecting table geometry, or otherwise not plausibly placed.
  - true if tool is at least meaningfully visible and plausibly in-scene.
- position_score (1.00 to 10.00):
  - grade only spatial placement of the instructed tool keypoint relative to targets/reference objects and the red target dot.
  - 10.00 = target alignment is excellent; 1.00 = very poor placement.
- orientation_score (1.00 to 10.00):
  - grade only rotational alignment implied by instruction (e.g., facing down, parallel, angled toward object).
  - 10.00 = orientation matches instruction perfectly; 1.00 = major mismatch.
- Use up to 2 decimal places.
- Do NOT score robot arm style, lighting, texture quality, or unrelated object motion.
- Do NOT reward/penalize occlusions unless they indicate physically invalid placement.
- If instruction orientation is ambiguous, score orientation based on best visible intent match and say so in notes.
"""


def generate_instruction_keypoints(
    *,
    image_path: Path,
    model: str = DEFAULT_MODEL,
    num_instructions: int = DEFAULT_NUM_INSTRUCTIONS,
) -> GenerationResult:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if num_instructions <= 0:
        raise ValueError("--num-instructions must be > 0")

    image_bytes = image_path.read_bytes()
    api_key = _load_api_key_from_repo_env()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _prompt(num_instructions),
        ],
        config=types.GenerateContentConfig(
            temperature=0.35,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )

    raw_text = response.text or ""
    parsed = _extract_json_blob(raw_text)
    instructions, keypoints = _validate_response(parsed, num_instructions)
    return GenerationResult(
        model=model,
        image=image_path.name,
        instructions=instructions,
        keypoints=keypoints,
        raw_text=raw_text,
    )


def _clamp_score_1_10(value: Any, field: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if f < 1.0 or f > 10.0:
        raise ValueError(f"{field} must be in [1.0, 10.0]")
    return round(f, 2)


def _validate_score_response(obj: Dict[str, Any]) -> Tuple[bool, float, float, str]:
    if "tool_visible" not in obj:
        raise ValueError("Response missing tool_visible")
    tool_visible = bool(obj["tool_visible"])
    position_score = _clamp_score_1_10(obj.get("position_score"), "position_score")
    orientation_score = _clamp_score_1_10(
        obj.get("orientation_score"), "orientation_score"
    )
    notes = str(obj.get("notes", "")).strip()
    if not notes:
        raise ValueError("Response missing notes")
    return tool_visible, position_score, orientation_score, notes


def score_rendered_tool_placement(
    *,
    image_path: Path,
    instruction: str,
    target_point_yx: Sequence[float] | None = None,
    model: str = DEFAULT_MODEL,
) -> ScoreResult:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not instruction.strip():
        raise ValueError("instruction is required")
    if target_point_yx is not None:
        _validate_point(target_point_yx, "target_point")

    image_bytes = image_path.read_bytes()
    api_key = _load_api_key_from_repo_env()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _score_prompt(instruction=instruction.strip(), target_point_yx=target_point_yx),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    raw_text = response.text or ""
    parsed = _extract_json_blob(raw_text)
    tool_visible, position_score, orientation_score, notes = _validate_score_response(
        parsed
    )
    return ScoreResult(
        model=model,
        image=image_path.name,
        instruction=instruction.strip(),
        tool_visible=tool_visible,
        position_score=position_score,
        orientation_score=orientation_score,
        notes=notes,
        raw_text=raw_text,
    )


def _denorm_yx(point_yx: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    y, x = float(point_yx[0]), float(point_yx[1])
    px = int((x / 1000.0) * width)
    py = int((y / 1000.0) * height)
    return px, py


def draw_keypoints(image_path: Path, keypoints: List[Dict[str, Any]], output_image: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    for item in keypoints:
        label = str(item["label"])
        px, py = _denorm_yx(item["point"], w, h)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(0, 220, 255, 255))
        draw.text((px + 8, py - 8), label, fill=(0, 220, 255, 255), font=font)

    img.save(output_image)


def draw_target_dot(
    image_path: Path, target_point_yx: Sequence[float], output_image: Path
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    px, py = _denorm_yx(target_point_yx, w, h)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(255, 0, 0, 255))
    draw.ellipse((px - 14, py - 14, px + 14, py + 14), outline=(255, 0, 0, 220), width=2)
    img.save(output_image)


def write_output_json(output_json: Path, result: GenerationResult) -> None:
    payload = {
        "model": result.model,
        "image": result.image,
        "instructions": result.instructions,
        "keypoints": result.keypoints,
        "raw_response_text": result.raw_text,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_score_json(output_json: Path, result: ScoreResult) -> None:
    payload = {
        "model": result.model,
        "image": result.image,
        "instruction": result.instruction,
        "tool_visible": result.tool_visible,
        "position_score": result.position_score,
        "orientation_score": result.orientation_score,
        "notes": result.notes,
        "raw_response_text": result.raw_text,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate instruction/keypoints or score tool placement with Gemini."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate instruction/tool_keypoint samples and keypoints.",
    )
    gen_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input scene image (PNG).",
    )
    gen_parser.add_argument(
        "--num-instructions",
        type=int,
        default=DEFAULT_NUM_INSTRUCTIONS,
        help=f"Number of instruction samples to generate (default: {DEFAULT_NUM_INSTRUCTIONS}).",
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model id (default: {DEFAULT_MODEL}).",
    )
    gen_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <image_stem>_instruction_keypoints.json).",
    )
    gen_parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Output annotated image path (default: <image_stem>_keypoints.png).",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score rendered tool placement from instruction + image.",
    )
    score_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Rendered workspace image containing final tool pose and target red dot.",
    )
    score_parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Original instruction used to place the tool.",
    )
    score_parser.add_argument(
        "--target-y",
        type=float,
        default=None,
        help="Optional target point y in normalized 0..1000.",
    )
    score_parser.add_argument(
        "--target-x",
        type=float,
        default=None,
        help="Optional target point x in normalized 0..1000.",
    )
    score_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model id (default: {DEFAULT_MODEL}).",
    )
    score_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <image_stem>_placement_score.json).",
    )
    score_parser.add_argument(
        "--score-overlay-image",
        type=Path,
        default=None,
        help="Optional image path to save red-dot target overlay for debugging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "generate":
        image_path: Path = args.image
        output_json = args.output_json or image_path.with_name(
            f"{image_path.stem}_instruction_keypoints.json"
        )
        output_image = args.output_image or image_path.with_name(
            f"{image_path.stem}_keypoints.png"
        )

        result = generate_instruction_keypoints(
            image_path=image_path,
            model=args.model,
            num_instructions=args.num_instructions,
        )
        write_output_json(output_json, result)
        draw_keypoints(image_path, result.keypoints, output_image)

        print(f"Model: {result.model}")
        print(f"Image: {image_path}")
        print(f"Instructions generated: {len(result.instructions)}")
        print(f"Keypoints generated: {len(result.keypoints)}")
        print(f"Saved JSON: {output_json}")
        print(f"Saved keypoint image: {output_image}")
        return

    target_point = None
    if args.target_y is not None or args.target_x is not None:
        if args.target_y is None or args.target_x is None:
            raise ValueError("Provide both --target-y and --target-x, or neither.")
        target_point = [args.target_y, args.target_x]

    score = score_rendered_tool_placement(
        image_path=args.image,
        instruction=args.instruction,
        target_point_yx=target_point,
        model=args.model,
    )
    output_json = args.output_json or args.image.with_name(
        f"{args.image.stem}_placement_score.json"
    )
    write_score_json(output_json, score)
    if target_point is not None:
        overlay_image = args.score_overlay_image or args.image.with_name(
            f"{args.image.stem}_target_dot.png"
        )
        draw_target_dot(args.image, target_point, overlay_image)
    print(f"Model: {score.model}")
    print(f"Image: {args.image}")
    print(f"Tool visible: {score.tool_visible}")
    print(f"Position score: {score.position_score:.2f}/10")
    print(f"Orientation score: {score.orientation_score:.2f}/10")
    print(f"Saved JSON: {output_json}")
    if target_point is not None:
        print(f"Saved target-dot overlay: {overlay_image}")


if __name__ == "__main__":
    main()
