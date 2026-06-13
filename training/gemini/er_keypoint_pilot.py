"""Run a fixed-size Gemini ER keypoint pilot and save artifacts.

This utility is designed for large runs (e.g., runs_0036) where we want:
- a sample subset (e.g., 50 images),
- raw + validated keypoint JSON per scene,
- quick keypoint overlays,
- token-usage based cost estimates.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from training.gemini.gemini_position_dataset import (
    _extract_json_blob,
    _load_api_key_from_repo_env,
    _validate_model_keypoints,
)

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"
INPUT_PRICE_PER_1M = 1.0
OUTPUT_PRICE_PER_1M = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt() -> str:
    return """Analyze the image and return ONLY visible object keypoints.

Output ONLY valid JSON:
{
  "objects": [
    {
      "object_id": "obj_name_index_with_underscores",
      "name": "object plain name",
      "keypoints": [
        {"id": "kp_id_with_underscores", "label": "Plain language label", "point": [y, x]}
      ]
    }
  ]
}

Rules:
- Return keypoints only for clearly visible objects/parts.
- If uncertain or occluded, omit instead of guessing.
- Aim for denser but useful coverage:
  - For non-tool objects: return at least 1-2 stable keypoints each.
  - For tools: return at least 2-3 meaningful keypoints (tip/head/handle/end where visible).
- keypoint id must use lowercase letters/numbers/underscores only.
- labels should be plain language.
- point is [y, x] normalized in [0, 1000].
- Do not include scene_id or any extra fields.
"""


def _draw_overlay(image_path: Path, objects: List[Dict[str, Any]], out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    for obj in objects:
        obj_name = str(obj.get("name", "")).strip()
        for kp in obj.get("keypoints", []):
            label = str(kp.get("label", "")).strip()
            point = kp.get("point", [0, 0])
            y, x = float(point[0]), float(point[1])
            u = int((x / 1000.0) * w)
            v = int((y / 1000.0) * h)
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=(255, 255, 255, 240))
            txt = f"{obj_name}: {label}" if obj_name else label
            draw.text((u + 7, v - 8), txt, fill=(255, 255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _extract_usage_counts(resp: Any) -> Tuple[int, int]:
    usage = getattr(resp, "usage_metadata", None)
    if usage is None:
        return 0, 0
    # Handle either pydantic-like model_dump() or attr objects.
    if hasattr(usage, "model_dump"):
        d = usage.model_dump()
        in_t = int(d.get("prompt_token_count", 0) or d.get("input_token_count", 0) or 0)
        out_t = int(
            d.get("candidates_token_count", 0)
            or d.get("output_token_count", 0)
            or d.get("response_token_count", 0)
            or 0
        )
        return in_t, out_t
    in_t = int(
        getattr(usage, "prompt_token_count", 0)
        or getattr(usage, "input_token_count", 0)
        or 0
    )
    out_t = int(
        getattr(usage, "candidates_token_count", 0)
        or getattr(usage, "output_token_count", 0)
        or getattr(usage, "response_token_count", 0)
        or 0
    )
    return in_t, out_t


def run_pilot(
    *,
    runs_root: Path,
    output_root: Path,
    sample_count: int,
    seed: int,
    model: str,
) -> Dict[str, Any]:
    rgb_files = sorted(runs_root.glob("scene_*/rgb.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No scene_*/rgb.png files found in {runs_root}")
    if sample_count <= 0:
        raise ValueError("--sample-count must be > 0")

    rng = random.Random(seed)
    selected = rgb_files[:] if sample_count >= len(rgb_files) else rng.sample(rgb_files, sample_count)
    selected = sorted(selected)

    raw_dir = output_root / "keypoints_raw"
    validated_dir = output_root / "keypoints_validated"
    overlays_dir = output_root / "overlays"
    logs_dir = output_root / "logs"
    for d in (raw_dir, validated_dir, overlays_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Prefer shared camera_k at run root when available.
    camera_k_path = runs_root / "camera_k.txt"
    shared_camera_k: Dict[str, Any] | None = None
    if camera_k_path.is_file():
        shared_camera_k = {"camera_k_path": str(camera_k_path), "camera_k_text": camera_k_path.read_text(encoding="utf-8")}
    else:
        first_cam = selected[0].parent / "camera.json"
        if first_cam.is_file():
            cam_obj = json.loads(first_cam.read_text(encoding="utf-8"))
            shared_camera_k = {
                "camera_k_path": None,
                "derived_from": str(first_cam),
                "intrinsics_fx_fy_cx_cy_px": cam_obj.get("intrinsics_fx_fy_cx_cy_px"),
                "intrinsics_matrix_3x3": cam_obj.get("intrinsics_matrix_3x3"),
            }

    client = genai.Client(api_key=_load_api_key_from_repo_env())
    per_image: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for rgb_path in selected:
        scene_id = rgb_path.parent.name
        img_bytes = rgb_path.read_bytes()
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"), _prompt()],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )
        text = resp.text or ""
        parsed = _extract_json_blob(text)
        validated = _validate_model_keypoints(parsed)

        in_t, out_t = _extract_usage_counts(resp)
        total_input_tokens += in_t
        total_output_tokens += out_t

        raw_out = raw_dir / f"{scene_id}.json"
        val_out = validated_dir / f"{scene_id}.json"
        overlay_out = overlays_dir / f"{scene_id}.png"
        raw_out.write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "rgb_path": str(rgb_path),
                    "model": model,
                    "raw_response_text": text,
                    "parsed_json": parsed,
                    "usage": {"input_tokens": in_t, "output_tokens": out_t},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        val_out.write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "rgb_path": str(rgb_path),
                    "model": model,
                    "objects": validated,
                    "usage": {"input_tokens": in_t, "output_tokens": out_t},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _draw_overlay(rgb_path, validated, overlay_out)

        per_image.append(
            {
                "scene_id": scene_id,
                "rgb_path": str(rgb_path),
                "raw_json_path": str(raw_out),
                "validated_json_path": str(val_out),
                "overlay_path": str(overlay_out),
                "input_tokens": in_t,
                "output_tokens": out_t,
            }
        )

    est_input_cost = (total_input_tokens / 1_000_000.0) * INPUT_PRICE_PER_1M
    est_output_cost = (total_output_tokens / 1_000_000.0) * OUTPUT_PRICE_PER_1M
    est_total_cost = est_input_cost + est_output_cost

    summary = {
        "created_at": _now_iso(),
        "runs_root": str(runs_root),
        "sample_count": len(selected),
        "seed": seed,
        "model": model,
        "shared_camera_k": shared_camera_k,
        "pricing_usd_per_1m_tokens": {
            "input": INPUT_PRICE_PER_1M,
            "output": OUTPUT_PRICE_PER_1M,
        },
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_input_cost_usd": est_input_cost,
            "estimated_output_cost_usd": est_output_cost,
            "estimated_total_cost_usd": est_total_cost,
        },
        "images": per_image,
    }
    (logs_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": output_root.name,
                "dataset_root": str(output_root),
                "runs_root": str(runs_root),
                "schema_version": "er_pilot_v1",
                "artifacts": {
                    "raw_dir": str(raw_dir),
                    "validated_dir": str(validated_dir),
                    "overlays_dir": str(overlays_dir),
                    "logs_dir": str(logs_dir),
                    "pilot_summary": str(logs_dir / "pilot_summary.json"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run Gemini ER pilot sampling on scene rgb images.")
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--sample-count", type=int, default=50)
    p.add_argument("--seed", type=int, default=36)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = p.parse_args()
    summary = run_pilot(
        runs_root=args.runs_root,
        output_root=args.output_root,
        sample_count=args.sample_count,
        seed=args.seed,
        model=args.model,
    )
    print(json.dumps(summary["totals"], indent=2))
    print(f"Saved summary: {args.output_root / 'logs' / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()

