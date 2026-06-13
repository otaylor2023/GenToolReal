"""Build a Gemini-driven position dataset from RGBD scene folders.

Input layout:
  data/
    camera_k.txt
    scene_0001/
      rgb/
        0000.png
      depth/
        0000.png | 0000.npy | 0000.npz | 0000.exr
      camera_pose_0000.txt|json (or camera_pose.txt|json)

Output (per scene):
  - scene shard JSON with non-redundant top-level scene metadata
  - optional parquet datapoints file
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_KEYPOINT_MODEL = "gemini-robotics-er-1.6-preview"
DEFAULT_INSTRUCTION_MODEL = "gemini-2.5-flash"
SCENE_GLOB = "scene_*"
TOOL_WORDS = (
    "tool",
    "brush",
    "screwdriver",
    "hammer",
    "wrench",
    "spatula",
    "knife",
    "scissors",
    "plier",
    "drill",
)
ALIAS_ABOVE_V0 = {"near", "next_to", "above", "over"}
MOVEMENT_TOKENS = (
    "near",
    "next_to",
    "above",
    "exact_above",
    "over",
    "left",
    "right",
    "in_front",
    "behind",
    "between",
)


@dataclass
class FrameRecord:
    frame_id: str
    rgb_path: Path
    depth_path: Path
    world_from_camera: np.ndarray
    intrinsics: Dict[str, float]


def _load_api_key_from_repo_env() -> str:
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


def _parse_matrix44(obj: Any, field_name: str) -> np.ndarray:
    arr = np.array(obj, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{field_name} must be a 4x4 matrix")
    return arr


def _load_camera_k(camera_k_path: Path) -> Dict[str, float]:
    if not camera_k_path.is_file():
        raise FileNotFoundError(f"Missing intrinsics file: {camera_k_path}")
    raw = camera_k_path.read_text(encoding="utf-8").strip()
    nums = [
        float(x)
        for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    ]
    if len(nums) >= 9:
        k = np.array(nums[:9], dtype=np.float64).reshape(3, 3)
        return {
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
        }
    if len(nums) >= 4:
        return {"fx": nums[0], "fy": nums[1], "cx": nums[2], "cy": nums[3]}
    raise ValueError(f"Could not parse camera intrinsics from {camera_k_path}")


def _load_world_from_camera(scene_dir: Path, frame_id: str) -> np.ndarray:
    candidates = [
        scene_dir / f"camera_pose_{frame_id}.json",
        scene_dir / f"camera_pose_{frame_id}.txt",
        scene_dir / "camera_pose.json",
        scene_dir / "camera_pose.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                if "world_from_camera" in obj:
                    return _parse_matrix44(obj["world_from_camera"], "world_from_camera")
                if "camera_from_world" in obj:
                    cfw = _parse_matrix44(obj["camera_from_world"], "camera_from_world")
                    return np.linalg.inv(cfw)
            return _parse_matrix44(obj, "camera pose json")
        nums = [
            float(x)
            for x in re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                path.read_text(encoding="utf-8"),
            )
        ]
        if len(nums) >= 16:
            return np.array(nums[:16], dtype=np.float64).reshape(4, 4)
    raise FileNotFoundError(
        f"No camera pose file for frame '{frame_id}' in {scene_dir}"
    )


def _look_at_world_from_camera(
    camera_pos: Sequence[float], look_at: Sequence[float]
) -> np.ndarray:
    cam = np.array(camera_pos, dtype=np.float64)
    tgt = np.array(look_at, dtype=np.float64)
    forward = tgt - cam
    n = np.linalg.norm(forward)
    if n < 1e-8:
        raise ValueError("camera position and look_at are identical")
    forward /= n
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
    right /= max(rn, 1e-8)
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-8)
    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = right
    m[:3, 1] = up
    m[:3, 2] = forward
    m[:3, 3] = cam
    return m


def _load_camera_json(scene_dir: Path) -> Tuple[Dict[str, float], np.ndarray]:
    path = scene_dir / "camera.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera.json in {scene_dir}")
    obj = json.loads(path.read_text(encoding="utf-8"))

    if "intrinsics_fx_fy_cx_cy_px" in obj:
        fx, fy, cx, cy = [float(v) for v in obj["intrinsics_fx_fy_cx_cy_px"][:4]]
    elif "intrinsics_matrix_3x3" in obj:
        k = np.array(obj["intrinsics_matrix_3x3"], dtype=np.float64)
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    else:
        raise ValueError(f"camera.json missing intrinsics fields in {scene_dir}")
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    if "world_from_camera_exported" in obj:
        wfc = _parse_matrix44(obj["world_from_camera_exported"], "world_from_camera_exported")
    elif "world_from_camera" in obj:
        wfc = _parse_matrix44(obj["world_from_camera"], "world_from_camera")
    elif "camera_from_world" in obj:
        cfw = _parse_matrix44(obj["camera_from_world"], "camera_from_world")
        wfc = np.linalg.inv(cfw)
    elif "world_position_xyz_m" in obj and "look_at_xyz_m" in obj:
        wfc = _look_at_world_from_camera(obj["world_position_xyz_m"], obj["look_at_xyz_m"])
    else:
        raise ValueError(f"camera.json missing pose fields in {scene_dir}")
    return intr, wfc


def _discover_scene_frames(
    scene_dir: Path,
    *,
    global_intrinsics: Dict[str, float] | None = None,
    global_world_from_camera: np.ndarray | None = None,
) -> List[FrameRecord]:
    # New runs_0001 schema: scene_xxxx/{rgb.png, depth.npy|depth.png, camera.json}
    direct_rgb = scene_dir / "rgb.png"
    if direct_rgb.is_file():
        depth_candidates = [
            scene_dir / "depth.npy",
            scene_dir / "depth.npz",
            scene_dir / "depth.exr",
            scene_dir / "depth.png",
            scene_dir / "depth.tif",
            scene_dir / "depth.tiff",
        ]
        depth = next((d for d in depth_candidates if d.is_file()), None)
        if depth is None:
            raise FileNotFoundError(f"Missing depth file in {scene_dir}")
        if global_intrinsics is not None and global_world_from_camera is not None:
            intr = dict(global_intrinsics)
            wfc = np.array(global_world_from_camera, dtype=np.float64)
        else:
            intr, wfc = _load_camera_json(scene_dir)
        return [
            FrameRecord(
                frame_id="0000",
                rgb_path=direct_rgb,
                depth_path=depth,
                world_from_camera=wfc,
                intrinsics=intr,
            )
        ]

    # Legacy schema: scene_xxxx/rgb/* and depth/*
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing rgb/depth folders in {scene_dir}")

    rgb_files = sorted(
        p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    frames: List[FrameRecord] = []
    shared_intr: Dict[str, float] | None = None
    shared_intr_path = scene_dir.parent / "camera_k.txt"
    if shared_intr_path.is_file():
        shared_intr = _load_camera_k(shared_intr_path)
    for rgb in rgb_files:
        frame_id = rgb.stem
        depth_candidates = [
            depth_dir / f"{frame_id}.npy",
            depth_dir / f"{frame_id}.npz",
            depth_dir / f"{frame_id}.exr",
            depth_dir / f"{frame_id}.png",
            depth_dir / f"{frame_id}.tif",
            depth_dir / f"{frame_id}.tiff",
        ]
        depth = next((d for d in depth_candidates if d.is_file()), None)
        if depth is None:
            continue
        wfc = _load_world_from_camera(scene_dir, frame_id)
        intr = shared_intr if shared_intr is not None else {}
        frames.append(
            FrameRecord(
                frame_id=frame_id,
                rgb_path=rgb,
                depth_path=depth,
                world_from_camera=wfc,
                intrinsics=intr,
            )
        )
    return frames


def _load_depth(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(depth_path)
    elif suffix == ".npz":
        npz = np.load(depth_path)
        key = "depth" if "depth" in npz else list(npz.keys())[0]
        arr = npz[key]
    else:
        arr = np.array(Image.open(depth_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _gemini_keypoint_prompt() -> str:
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
- labels should be plain language (e.g., "Brush tip", "Center of platform").
- point is [y, x] normalized in [0, 1000].
- Do not include scene_id or any extra fields.
"""


def _instruction_prompt(
    *,
    relation_string: str,
    movement_token: str,
    labels: Sequence[str],
) -> str:
    labels_json = json.dumps(list(labels))
    return f"""Generate instruction variants for a robot-manipulation training dataset.

Canonical relation string:
{relation_string}

Movement token:
{movement_token}

Referenced labels:
{labels_json}

Return ONLY valid JSON:
{{
  "instructions": [
    "instruction 1",
    "instruction 2",
    "instruction 3",
    "instruction 4"
  ]
}}

Rules:
- Generate exactly 4 instructions.
- Use moderate variation in wording while preserving the same meaning.
- Keep imperative manipulation language.
- Preserve the entities and relation from the canonical relation string.
- Do not add unseen objects, targets, or relations.
- Keep each instruction concise and non-identical.
"""


def _validate_model_keypoints(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Gemini output missing 'objects' list")
    out: List[Dict[str, Any]] = []
    id_re = re.compile(r"^[a-z0-9_]+$")
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id", "")).strip()
        name = str(obj.get("name", "")).strip()
        kps = obj.get("keypoints")
        if not object_id or not id_re.match(object_id) or not name or not isinstance(kps, list):
            continue
        clean_kps = []
        for kp in kps:
            if not isinstance(kp, dict):
                continue
            kp_id = str(kp.get("id", "")).strip()
            label = str(kp.get("label", "")).strip()
            point = kp.get("point")
            if not kp_id or not id_re.match(kp_id) or not label:
                continue
            if not isinstance(point, Sequence) or len(point) != 2:
                continue
            y, x = float(point[0]), float(point[1])
            if y < 0 or y > 1000 or x < 0 or x > 1000:
                continue
            clean_kps.append({"id": kp_id, "label": label, "point": [y, x]})
        if clean_kps:
            out.append({"object_id": object_id, "name": name, "keypoints": clean_kps})
    return out


def _call_gemini_objects(client: genai.Client, model: str, rgb_path: Path) -> List[Dict[str, Any]]:
    image_bytes = rgb_path.read_bytes()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _gemini_keypoint_prompt(),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    parsed = _extract_json_blob(resp.text or "")
    objects = _validate_model_keypoints(parsed)
    if not objects:
        raise ValueError(f"No valid keypoints from Gemini for {rgb_path}")
    return objects


def _call_gemini_objects_with_retry(
    client: genai.Client,
    model: str,
    rgb_path: Path,
    max_attempts: int = 4,
    base_backoff_s: float = 1.5,
) -> List[Dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_gemini_objects(client, model, rgb_path)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_s = min(base_backoff_s * (2 ** (attempt - 1)), 30.0) + random.uniform(0.0, 0.6)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _call_gemini_objects_with_raw(
    client: genai.Client, model: str, rgb_path: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    image_bytes = rgb_path.read_bytes()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _gemini_keypoint_prompt(),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    text = resp.text or ""
    parsed = _extract_json_blob(text)
    objects = _validate_model_keypoints(parsed)
    if not objects:
        raise ValueError(f"No valid keypoints from Gemini for {rgb_path}")
    return parsed, objects, text


def _validate_instruction_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError("instructions must be a list")
    cleaned: List[str] = []
    seen = set()
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
    if len(cleaned) != 4:
        raise ValueError(f"Expected 4 unique instructions, got {len(cleaned)}")
    return cleaned


def _generate_instruction_variants(
    *,
    client: genai.Client,
    instruction_model: str,
    relation_string: str,
    movement_token: str,
    labels: Sequence[str],
    max_attempts: int = 2,
) -> List[str]:
    last_error = None
    for _ in range(max_attempts):
        resp = client.models.generate_content(
            model=instruction_model,
            contents=[
                _instruction_prompt(
                    relation_string=relation_string,
                    movement_token=movement_token,
                    labels=labels,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.55,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=768),
            ),
        )
        try:
            parsed = _extract_json_blob(resp.text or "")
            return _validate_instruction_list(parsed.get("instructions"))
        except (ValueError, TypeError) as exc:
            last_error = exc
            continue
    raise ValueError(
        f"Failed to generate valid instruction variants for relation '{relation_string}': {last_error}"
    )


def _lift_to_world(
    point_yx: Sequence[float],
    depth_map: np.ndarray,
    intr: Dict[str, float],
    world_from_camera: np.ndarray,
) -> Tuple[Optional[List[float]], bool, List[int]]:
    h, w = depth_map.shape[:2]
    y, x = float(point_yx[0]), float(point_yx[1])
    u = int(np.clip((x / 1000.0) * w, 0, w - 1))
    v = int(np.clip((y / 1000.0) * h, 0, h - 1))
    z = float(depth_map[v, u])
    if not np.isfinite(z) or z <= 0.0:
        return None, False, [u, v]

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    # Correct Isaac/Replicator convention:
    # - image y is down, camera y is up  => flip sign on y term
    # - camera forward is -Z
    # - depth is distance_to_camera range along the view ray
    dir_c = np.array([(u - cx) / fx, -((v - cy) / fy), -1.0], dtype=np.float64)
    dir_c /= max(np.linalg.norm(dir_c), 1e-12)
    R = world_from_camera[:3, :3]
    t = world_from_camera[:3, 3]
    world_xyz = R @ (dir_c * z) + t
    world_point = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
    return [float(world_point[0]), float(world_point[1]), float(world_point[2])], True, [u, v]


def _draw_mini_overlay(
    *,
    image_path: Path,
    keypoints: Dict[str, Dict[str, Any]],
    tool_keypoint_id: str,
    ref_keypoint_ids: Sequence[str],
    goal_xyz_world: Sequence[float],
    intr: Dict[str, float],
    world_from_camera: np.ndarray,
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    def project_world(xyz: Sequence[float]) -> Tuple[int, int] | None:
        p_w = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        cfw = np.linalg.inv(world_from_camera)
        p_c = cfw @ p_w
        # In our convention, points in front have negative camera-z.
        d = -float(p_c[2])
        if d <= 1e-6:
            return None
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        u = int((p_c[0] / d) * fx + cx)
        v = int(cy - (p_c[1] / d) * fy)
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return u, v

    # all Gemini keypoints: white
    for kp in keypoints.values():
        yx = kp.get("point_yx_0_1000")
        if not yx:
            continue
        py, px = float(yx[0]), float(yx[1])
        u = int((px / 1000.0) * w)
        v = int((py / 1000.0) * h)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 255, 255, 240))

    # tool keypoint: blue
    t = keypoints[tool_keypoint_id]["point_yx_0_1000"]
    tu = int((float(t[1]) / 1000.0) * w)
    tv = int((float(t[0]) / 1000.0) * h)
    draw.ellipse((tu - 6, tv - 6, tu + 6, tv + 6), fill=(0, 170, 255, 255))

    # refs: cyan/yellow
    ref_colors = [(0, 255, 255, 255), (255, 230, 0, 255)]
    for i, rid in enumerate(ref_keypoint_ids):
        r = keypoints[rid]["point_yx_0_1000"]
        ru = int((float(r[1]) / 1000.0) * w)
        rv = int((float(r[0]) / 1000.0) * h)
        c = ref_colors[i % len(ref_colors)]
        draw.ellipse((ru - 6, rv - 6, ru + 6, rv + 6), fill=c)

    # goal: red projected from world xyz
    gp = project_world(goal_xyz_world)
    if gp is not None:
        gu, gv = gp
        draw.ellipse((gu - 7, gv - 7, gu + 7, gv + 7), fill=(255, 0, 0, 255))
        draw.ellipse((gu - 11, gv - 11, gu + 11, gv + 11), outline=(255, 0, 0, 230), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _movement_alias_group(token: str) -> str:
    if token == "exact_above":
        return "exact_above_cylinder_v1"
    if token in ALIAS_ABOVE_V0:
        return "above_v0"
    if token == "between":
        return "between_midpoint_v0"
    return "directional_v0"


def _goal_offset_for_token(token: str, dx: float, dy: float, dz: float) -> List[float]:
    """Token-aware offsets in canonical world frame (+X right, +Y away, +Z up)."""
    z_lift = abs(float(dz))
    x_step = abs(float(dx))
    y_step = abs(float(dy))

    # Keep practical fallbacks if caller passes zeros.
    if x_step <= 1e-8:
        x_step = 0.05
    if y_step <= 1e-8:
        y_step = 0.05
    if z_lift <= 1e-8:
        z_lift = 0.05

    if token == "left":
        return [-x_step, 0.0, z_lift]
    if token == "right":
        return [x_step, 0.0, z_lift]
    if token == "in_front":
        # +Y is away from camera, so "in_front" is toward camera => -Y.
        return [0.0, -y_step, z_lift]
    if token == "behind":
        # "behind" is away/back => +Y.
        return [0.0, y_step, z_lift]
    # near/next_to/above/over/exact_above default to same XY and lifted Z.
    return [0.0, 0.0, z_lift]


def _normalize_object_name(name: str) -> str:
    s = str(name).strip().lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _object_name_from_object_id(object_id: str) -> str:
    s = str(object_id).strip().lower()
    if not s:
        return ""
    s = re.sub(r"[_\-\s]*\d+$", "", s)
    return _normalize_object_name(s)


def _kp_phrase(kp: Dict[str, Any]) -> str:
    label = str(kp.get("label", "")).strip()
    obj_name = str(kp.get("object_name", "")).strip()
    if not obj_name:
        obj_name = _object_name_from_object_id(str(kp.get("object_id", "")))
    if label and obj_name:
        return f"{label} of {_normalize_object_name(obj_name)}"
    return label


def _is_tool_object_name(name: str) -> bool:
    s = name.lower()
    return any(w in s for w in TOOL_WORDS)


def _build_scene_shard(
    *,
    scene_id: str,
    frame: FrameRecord,
    intr: Dict[str, float],
    client: genai.Client,
    keypoint_model: str,
    instruction_model: str,
    num_datapoints: int,
    delta_world: Sequence[float],
    seed: int,
    debug_overlay_dir: Path | None = None,
    generate_instructions: bool = True,
    precomputed_objects: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    objects = (
        precomputed_objects
        if precomputed_objects is not None
        else _call_gemini_objects_with_retry(client, keypoint_model, frame.rgb_path)
    )
    depth_map = _load_depth(frame.depth_path)

    keypoints: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for obj in objects:
        object_id = obj["object_id"]
        object_name = obj["name"]
        for kp in obj["keypoints"]:
            kp_id = kp["id"]
            unique_id = f"{frame.frame_id}__{kp_id}"
            if unique_id in keypoints:
                duplicates += 1
                continue
            xyz, valid, uv = _lift_to_world(kp["point"], depth_map, intr, frame.world_from_camera)
            keypoints[unique_id] = {
                "frame_id": frame.frame_id,
                "object_id": object_id,
                "object_name": object_name,
                "label": kp["label"],
                "point_yx_0_1000": kp["point"],
                "uv_px": uv,
                "xyz_world": xyz,
                "valid": valid,
            }

    rng = random.Random(seed)
    all_ids = list(keypoints.keys())
    valid_ids = [k for k in all_ids if keypoints[k]["valid"]]
    tool_candidates = [
        k
        for k in valid_ids
        if _is_tool_object_name(keypoints[k]["object_name"])
        or "tool" in keypoints[k]["label"].lower()
    ]
    # Keep a practical fallback when names do not include obvious tool words.
    if not tool_candidates:
        tool_candidates = valid_ids[:]

    datapoints: List[Dict[str, Any]] = []
    movement_hist = Counter()
    dx, dy, dz = float(delta_world[0]), float(delta_world[1]), float(delta_world[2])

    for _ in range(num_datapoints):
        if not tool_candidates:
            break
        tool_id = rng.choice(tool_candidates)
        tool_obj = keypoints[tool_id]
        token = rng.choice(MOVEMENT_TOKENS)
        alias_group = _movement_alias_group(token)

        refs = [
            k
            for k in valid_ids
            if keypoints[k]["object_id"] != tool_obj["object_id"]
        ]
        if len(refs) < 1:
            break

        if token == "between" and len(refs) >= 2:
            ref_a, ref_b = rng.sample(refs, 2)
            a = np.array(keypoints[ref_a]["xyz_world"], dtype=np.float64)
            b = np.array(keypoints[ref_b]["xyz_world"], dtype=np.float64)
            between_goal = (a + b) * 0.5
            # Keep all goals at a common lifted Z level.
            between_goal[2] = between_goal[2] + abs(dz)
            goal = between_goal.tolist()
            relation_string = (
                f"[{_kp_phrase(tool_obj)}] [between] [{_kp_phrase(keypoints[ref_a])}] "
                f"[{_kp_phrase(keypoints[ref_b])}]"
            )
            instructions = (
                _generate_instruction_variants(
                    client=client,
                    instruction_model=instruction_model,
                    relation_string=relation_string,
                    movement_token=token,
                    labels=[
                        tool_obj["label"],
                        keypoints[ref_a]["label"],
                        keypoints[ref_b]["label"],
                    ],
                )
                if generate_instructions
                else []
            )
            dp = {
                "tool_keypoint_id": tool_id,
                "ref_keypoint_ids": [ref_a, ref_b],
                "movement_token": token,
                "movement_alias_group": alias_group,
                "constraint_type": "between_midpoint_v0",
                "constraint_params": {},
                "goal_tool_keypoint_xyz_world": [float(goal[0]), float(goal[1]), float(goal[2])],
                "relation_string": relation_string,
                "instructions": instructions,
            }
            datapoints.append(dp)
            if debug_overlay_dir is not None:
                _draw_mini_overlay(
                    image_path=frame.rgb_path,
                    keypoints=keypoints,
                    tool_keypoint_id=tool_id,
                    ref_keypoint_ids=[ref_a, ref_b],
                    goal_xyz_world=dp["goal_tool_keypoint_xyz_world"],
                    intr=intr,
                    world_from_camera=frame.world_from_camera,
                    out_path=debug_overlay_dir / f"{scene_id}_{len(datapoints):03d}.png",
                )
        else:
            ref = rng.choice(refs)
            ref_xyz = keypoints[ref]["xyz_world"]
            constraint_type = "point_goal_v0"
            constraint_params: Dict[str, Any] = {}
            if token == "exact_above":
                z_target = _goal_offset_for_token(token, dx, dy, dz)[2]
                z_min_offset_m = max(0.0, z_target - 0.02)
                z_max_offset_m = z_target + 0.02
                goal = [ref_xyz[0], ref_xyz[1], ref_xyz[2] + z_target]
                constraint_type = "exact_above_cylinder"
                constraint_params = {
                    "reference_keypoint_id": ref,
                    "xy_radius_m": 0.02,
                    "z_min_offset_m": z_min_offset_m,
                    "z_max_offset_m": z_max_offset_m,
                }
            else:
                off = _goal_offset_for_token(token, dx, dy, dz)
                goal = [ref_xyz[0] + off[0], ref_xyz[1] + off[1], ref_xyz[2] + off[2]]
            relation_string = (
                f"[{_kp_phrase(tool_obj)}] [{token}] [{_kp_phrase(keypoints[ref])}]"
            )
            instructions = (
                _generate_instruction_variants(
                    client=client,
                    instruction_model=instruction_model,
                    relation_string=relation_string,
                    movement_token=token,
                    labels=[tool_obj["label"], keypoints[ref]["label"]],
                )
                if generate_instructions
                else []
            )
            dp = {
                "tool_keypoint_id": tool_id,
                "ref_keypoint_ids": [ref],
                "movement_token": token,
                "movement_alias_group": alias_group,
                "constraint_type": constraint_type,
                "constraint_params": constraint_params,
                "goal_tool_keypoint_xyz_world": [float(goal[0]), float(goal[1]), float(goal[2])],
                "relation_string": relation_string,
                "instructions": instructions,
            }
            datapoints.append(dp)
            if debug_overlay_dir is not None:
                _draw_mini_overlay(
                    image_path=frame.rgb_path,
                    keypoints=keypoints,
                    tool_keypoint_id=tool_id,
                    ref_keypoint_ids=[ref],
                    goal_xyz_world=dp["goal_tool_keypoint_xyz_world"],
                    intr=intr,
                    world_from_camera=frame.world_from_camera,
                    out_path=debug_overlay_dir / f"{scene_id}_{len(datapoints):03d}.png",
                )
        movement_hist[token] += 1

    valid_count = sum(1 for v in keypoints.values() if v["valid"])
    total_count = len(keypoints)
    instruction_count_ok = sum(
        1
        for dp in datapoints
        if isinstance(dp.get("instructions"), list) and len(dp["instructions"]) == 4
    )
    instruction_unique_ok = sum(
        1
        for dp in datapoints
        if isinstance(dp.get("instructions"), list)
        and len({str(x).strip().lower() for x in dp["instructions"]}) == 4
    )
    qa = {
        "total_keypoints": total_count,
        "valid_lift_keypoints": valid_count,
        "lift_validity_rate": float(valid_count / total_count) if total_count else 0.0,
        "duplicate_keypoint_ids": duplicates,
        "movement_histogram": dict(movement_hist),
        "datapoint_count": len(datapoints),
        "instruction_count_ok": instruction_count_ok,
        "instruction_unique_ok": instruction_unique_ok,
    }

    return {
        "scene_id": scene_id,
        "image": str(frame.rgb_path),
        "depth": str(frame.depth_path),
        "camera": {
            "intrinsics": {
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "width": int(depth_map.shape[1]),
                "height": int(depth_map.shape[0]),
            },
            "world_from_camera": frame.world_from_camera.tolist(),
        },
        "movement_rules": {
            "version": "position_dataset_v0.1",
            "delta_world": [dx, dy, dz],
            "alias_groups": {
                "above_v0": sorted(ALIAS_ABOVE_V0),
                "between_midpoint_v0": ["between"],
                "exact_above_cylinder_v1": ["exact_above"],
            },
        },
        "keypoints": keypoints,
        "datapoints": datapoints,
        "qa": qa,
    }


def _write_outputs(
    shard: Dict[str, Any],
    output_dir: Path,
    write_parquet: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = shard["scene_id"]
    shard_json = output_dir / f"{scene_id}_position_dataset_v0_1.json"
    shard_json.write_text(json.dumps(shard, indent=2), encoding="utf-8")

    if not write_parquet:
        return
    try:
        import pandas as pd
    except ImportError:
        return
    rows = []
    for dp in shard["datapoints"]:
        rows.append(
            {
                "scene_id": scene_id,
                "image": shard["image"],
                "depth": shard["depth"],
                "tool_keypoint_id": dp["tool_keypoint_id"],
                "ref_keypoint_ids": json.dumps(dp["ref_keypoint_ids"]),
                "movement_token": dp["movement_token"],
                "movement_alias_group": dp["movement_alias_group"],
                "goal_tool_keypoint_xyz_world": json.dumps(dp["goal_tool_keypoint_xyz_world"]),
                "relation_string": dp["relation_string"],
                "instructions": json.dumps(dp["instructions"]),
            }
        )
    if rows:
        pd.DataFrame(rows).to_parquet(
            output_dir / f"{scene_id}_position_datapoints_v0_1.parquet",
            index=False,
        )


def build_dataset(
    *,
    data_root: Path,
    output_dir: Path,
    keypoint_model: str,
    instruction_model: str,
    num_datapoints: int,
    delta_world: Sequence[float],
    seed: int,
    write_parquet: bool,
    debug_overlay_dir: Path | None = None,
    generate_instructions: bool = True,
    precomputed_keypoints_dir: Path | None = None,
    max_workers: int = 1,
    max_scenes: int | None = None,
    skip_existing: bool = True,
    scene_list: Sequence[str] | None = None,
    global_camera_json: Path | None = None,
) -> Dict[str, Any]:
    api_key = _load_api_key_from_repo_env()
    global_intrinsics: Dict[str, float] | None = None
    global_world_from_camera: np.ndarray | None = None

    # Prefer a runs_00XX root camera.json when available to enforce one canonical camera transform.
    selected_global_camera = global_camera_json
    if selected_global_camera is None:
        candidate = data_root / "camera.json"
        if candidate.is_file():
            selected_global_camera = candidate
    if selected_global_camera is not None:
        gobj = json.loads(Path(selected_global_camera).read_text(encoding="utf-8"))
        if "intrinsics_fx_fy_cx_cy_px" in gobj:
            fx, fy, cx, cy = [float(v) for v in gobj["intrinsics_fx_fy_cx_cy_px"][:4]]
        elif "intrinsics_matrix_3x3" in gobj:
            k = np.array(gobj["intrinsics_matrix_3x3"], dtype=np.float64)
            fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        else:
            raise ValueError(f"Global camera missing intrinsics fields: {selected_global_camera}")
        global_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        if "world_from_camera_exported" in gobj:
            global_world_from_camera = _parse_matrix44(gobj["world_from_camera_exported"], "world_from_camera_exported")
        elif "world_from_camera" in gobj:
            global_world_from_camera = _parse_matrix44(gobj["world_from_camera"], "world_from_camera")
        elif "camera_from_world" in gobj:
            global_world_from_camera = np.linalg.inv(_parse_matrix44(gobj["camera_from_world"], "camera_from_world"))
        else:
            raise ValueError(f"Global camera missing pose fields: {selected_global_camera}")

    scene_dirs = sorted(
        d for d in data_root.glob(SCENE_GLOB) if d.is_dir()
    )
    if scene_list:
        wanted = {str(x).strip() for x in scene_list if str(x).strip()}
        scene_dirs = [d for d in scene_dirs if d.name in wanted]
    if max_scenes is not None and max_scenes > 0:
        scene_dirs = scene_dirs[: int(max_scenes)]
    if not scene_dirs:
        raise FileNotFoundError(f"No scene_* folders found in {data_root}")

    summary = {
        "scenes_total": len(scene_dirs),
        "scenes_processed": 0,
        "scenes_failed": 0,
        "scenes_skipped_existing": 0,
        "datapoints_total": 0,
        "mean_lift_validity_rate": 0.0,
        "scene_reports": [],
        "failed_scenes": [],
    }
    rates = []

    def process_one_scene(i: int, scene_dir: Path) -> Dict[str, Any] | None:
        scene_id = scene_dir.name
        shard_out = output_dir / f"{scene_id}_position_dataset_v0_1.json"
        if skip_existing and shard_out.is_file():
            return {"scene_id": scene_id, "status": "skipped_existing"}
        frames = _discover_scene_frames(
            scene_dir,
            global_intrinsics=global_intrinsics,
            global_world_from_camera=global_world_from_camera,
        )
        if not frames:
            return None
        frame = frames[0]  # v0.1 fixed-camera assumption; use first paired frame.
        precomputed_objects = None
        if precomputed_keypoints_dir is not None:
            kp_path = precomputed_keypoints_dir / f"{scene_id}.json"
            if not kp_path.is_file():
                return None
            kp_obj = json.loads(kp_path.read_text(encoding="utf-8"))
            precomputed_objects = kp_obj.get("objects")
            if not isinstance(precomputed_objects, list):
                precomputed_objects = kp_obj.get("raw_json", {}).get("objects")
            if not isinstance(precomputed_objects, list):
                return None
        client = genai.Client(api_key=api_key)
        try:
            shard = _build_scene_shard(
                scene_id=scene_id,
                frame=frame,
                intr=frame.intrinsics,
                client=client,
                keypoint_model=keypoint_model,
                instruction_model=instruction_model,
                num_datapoints=num_datapoints,
                delta_world=delta_world,
                seed=seed + i,
                debug_overlay_dir=(
                    debug_overlay_dir / scene_id if debug_overlay_dir is not None else None
                ),
                generate_instructions=generate_instructions,
                precomputed_objects=precomputed_objects,
            )
            _write_outputs(shard, output_dir=output_dir, write_parquet=write_parquet)
            return {
                "scene_id": scene_id,
                "status": "ok",
                "datapoints": int(shard["qa"]["datapoint_count"]),
                "lift_validity_rate": float(shard["qa"]["lift_validity_rate"]),
                "duplicate_keypoint_ids": int(shard["qa"]["duplicate_keypoint_ids"]),
                "instruction_count_ok": int(shard["qa"]["instruction_count_ok"]),
                "instruction_unique_ok": int(shard["qa"]["instruction_unique_ok"]),
            }
        except Exception as exc:
            return {"scene_id": scene_id, "status": "failed", "error": str(exc)}

    workers = max(1, int(max_workers))
    reports: List[Dict[str, Any]] = []
    if workers == 1:
        for i, scene_dir in enumerate(scene_dirs):
            rep = process_one_scene(i, scene_dir)
            if rep is not None:
                reports.append(rep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_one_scene, i, scene_dir): i
                for i, scene_dir in enumerate(scene_dirs)
            }
            for fut in as_completed(futures):
                rep = fut.result()
                if rep is not None:
                    reports.append(rep)

    reports.sort(key=lambda r: r["scene_id"])
    for rep in reports:
        status = rep.get("status", "ok")
        if status == "skipped_existing":
            summary["scenes_skipped_existing"] += 1
            continue
        if status == "failed":
            summary["scenes_failed"] += 1
            summary["failed_scenes"].append({"scene_id": rep["scene_id"], "error": rep.get("error", "")})
            continue
        summary["scenes_processed"] += 1
        summary["datapoints_total"] += int(rep["datapoints"])
        rates.append(float(rep["lift_validity_rate"]))
        clean = dict(rep)
        clean.pop("status", None)
        summary["scene_reports"].append(clean)

    summary["mean_lift_validity_rate"] = float(np.mean(rates)) if rates else 0.0
    summary["max_workers"] = workers
    summary["global_camera_json"] = str(selected_global_camera.resolve()) if selected_global_camera is not None else None
    summary_path = output_dir / "dataset_build_summary_v0_1.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_mini_verification(
    *,
    runs_root: Path,
    reports_dir: Path,
    keypoint_model: str,
    instruction_model: str,
    delta_world: Sequence[float],
    seed: int,
) -> None:
    api_key = _load_api_key_from_repo_env()
    client = genai.Client(api_key=api_key)
    reports_dir.mkdir(parents=True, exist_ok=True)
    global_intrinsics: Dict[str, float] | None = None
    global_world_from_camera: np.ndarray | None = None
    root_cam = runs_root / "camera.json"
    if root_cam.is_file():
        gobj = json.loads(root_cam.read_text(encoding="utf-8"))
        fx, fy, cx, cy = [float(v) for v in gobj["intrinsics_fx_fy_cx_cy_px"][:4]]
        global_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        if "world_from_camera_exported" in gobj:
            global_world_from_camera = _parse_matrix44(gobj["world_from_camera_exported"], "world_from_camera_exported")
        else:
            global_world_from_camera = _parse_matrix44(gobj["world_from_camera"], "world_from_camera")

    scene_dirs = sorted(d for d in runs_root.glob("scene_*") if d.is_dir())
    ingest = {"scenes": []}
    for s in scene_dirs:
        frames = _discover_scene_frames(
            s,
            global_intrinsics=global_intrinsics,
            global_world_from_camera=global_world_from_camera,
        )
        if not frames:
            continue
        f = frames[0]
        ingest["scenes"].append(
            {
                "scene_id": s.name,
                "rgb": str(f.rgb_path),
                "depth": str(f.depth_path),
                "intrinsics": f.intrinsics,
            }
        )
    (reports_dir / "scene_ingest_report.json").write_text(
        json.dumps(ingest, indent=2), encoding="utf-8"
    )

    scene1 = runs_root / "scene_0001"
    frame = _discover_scene_frames(
        scene1,
        global_intrinsics=global_intrinsics,
        global_world_from_camera=global_world_from_camera,
    )[0]
    raw, objects, raw_text = _call_gemini_objects_with_raw(client, keypoint_model, frame.rgb_path)
    (reports_dir / "scene_0001_keypoints.json").write_text(
        json.dumps({"raw_json": raw, "validated_objects": objects, "raw_text": raw_text}, indent=2),
        encoding="utf-8",
    )

    depth = _load_depth(frame.depth_path)
    lifted = []
    keypoint_map: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        for kp in obj["keypoints"]:
            xyz, valid, uv = _lift_to_world(kp["point"], depth, frame.intrinsics, frame.world_from_camera)
            kid = f"{frame.frame_id}__{kp['id']}"
            keypoint_map[kid] = {
                "label": kp["label"],
                "object_id": obj["object_id"],
                "point_yx_0_1000": kp["point"],
                "xyz_world": xyz,
                "valid": valid,
            }
            lifted.append({"id": kid, "label": kp["label"], "uv_px": uv, "xyz_world": xyz, "valid": valid})
    (reports_dir / "scene_0001_lifted_keypoints.json").write_text(
        json.dumps({"scene_id": "scene_0001", "lifted_keypoints": lifted}, indent=2),
        encoding="utf-8",
    )

    # Single-scene overlay datapoint
    valid_ids = [k for k, v in keypoint_map.items() if v["valid"]]
    if len(valid_ids) >= 2:
        rng = random.Random(seed)
        tool_id = valid_ids[0]
        ref_id = valid_ids[1]
        token = "near"
        off = _goal_offset_for_token(
            token,
            float(delta_world[0]),
            float(delta_world[1]),
            float(delta_world[2]),
        )
        goal = [
            keypoint_map[ref_id]["xyz_world"][0] + off[0],
            keypoint_map[ref_id]["xyz_world"][1] + off[1],
            keypoint_map[ref_id]["xyz_world"][2] + off[2],
        ]
        relation_string = f"[{_kp_phrase(keypoint_map[tool_id])}] [{token}] [{_kp_phrase(keypoint_map[ref_id])}]"
        _draw_mini_overlay(
            image_path=frame.rgb_path,
            keypoints=keypoint_map,
            tool_keypoint_id=tool_id,
            ref_keypoint_ids=[ref_id],
            goal_xyz_world=goal,
            intr=frame.intrinsics,
            world_from_camera=frame.world_from_camera,
            out_path=reports_dir / "scene_0001_overlay_001.png",
        )
        (reports_dir / "scene_0001_overlay_001.json").write_text(
            json.dumps(
                {
                    "movement_token": token,
                    "relation_string": relation_string,
                    "goal_xyz_world": goal,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Mini pass over all scenes
    out_dir = reports_dir / "mini_dataset"
    summary = build_dataset(
        data_root=runs_root,
        output_dir=out_dir,
        keypoint_model=keypoint_model,
        instruction_model=instruction_model,
        num_datapoints=3,
        delta_world=delta_world,
        seed=seed,
        write_parquet=False,
        debug_overlay_dir=reports_dir / "overlays",
        generate_instructions=False,
    )
    (reports_dir / "runs_0001_mini_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Gemini position dataset shards from data/scene_XXXX RGBD."
    )
    p.add_argument("--data-root", type=Path, required=True, help="Root folder (contains camera_k.txt and scene_* dirs).")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for scene shards.")
    p.add_argument(
        "--keypoint-model",
        type=str,
        default=DEFAULT_KEYPOINT_MODEL,
        help=f"Gemini model for image keypoint extraction (default: {DEFAULT_KEYPOINT_MODEL}).",
    )
    p.add_argument(
        "--instruction-model",
        type=str,
        default=DEFAULT_INSTRUCTION_MODEL,
        help=f"Gemini model for relation-string instruction variants (default: {DEFAULT_INSTRUCTION_MODEL}).",
    )
    p.add_argument("--num-datapoints", type=int, default=50, help="Datapoints to sample per scene (default: 50).")
    p.add_argument("--delta-world", type=float, nargs=3, default=[0.0, 0.0, 0.05], metavar=("DX", "DY", "DZ"), help="Fixed world offset for v0.1 alias rules.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--write-parquet", action="store_true", help="Also write flattened parquet datapoints per scene.")
    p.add_argument("--debug-overlay-dir", type=Path, default=None, help="Optional overlay directory; writes one overlay per datapoint with keypoint/ref/goal dots.")
    p.add_argument("--mini-verify", action="store_true", help="Run staged runs_0001 mini verification and write reports.")
    p.add_argument("--mini-reports-dir", type=Path, default=Path("mini_reports"), help="Output directory for mini verification artifacts.")
    p.add_argument("--skip-instructions", action="store_true", help="Skip instruction generation (useful for quota-limited visual QA).")
    p.add_argument("--max-workers", type=int, default=1, help="Parallel workers for per-scene processing.")
    p.add_argument("--max-scenes", type=int, default=None, help="Optional cap on number of scene_* dirs to process.")
    p.add_argument("--no-skip-existing", action="store_true", help="Process scenes even if output shard already exists.")
    p.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Optional newline-delimited scene IDs to process (e.g., scene_00014).",
    )
    p.add_argument(
        "--precomputed-keypoints-dir",
        type=Path,
        default=None,
        help="Optional dir of per-scene keypoint JSON files (<scene_id>.json) to avoid ER calls.",
    )
    p.add_argument(
        "--global-camera-json",
        type=Path,
        default=None,
        help=(
            "Optional root camera.json used for all scenes. If unset and <data-root>/camera.json exists, "
            "that file is used automatically."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    scene_list = None
    if args.scene_list_file is not None:
        scene_list = [
            ln.strip()
            for ln in args.scene_list_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    if args.mini_verify:
        run_mini_verification(
            runs_root=args.data_root,
            reports_dir=args.mini_reports_dir,
            keypoint_model=args.keypoint_model,
            instruction_model=args.instruction_model,
            delta_world=args.delta_world,
            seed=args.seed,
        )
        print(f"Mini verification artifacts written to: {args.mini_reports_dir}")
        return

    summary = build_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        keypoint_model=args.keypoint_model,
        instruction_model=args.instruction_model,
        num_datapoints=args.num_datapoints,
        delta_world=args.delta_world,
        seed=args.seed,
        write_parquet=bool(args.write_parquet),
        debug_overlay_dir=args.debug_overlay_dir,
        generate_instructions=not bool(args.skip_instructions),
        precomputed_keypoints_dir=args.precomputed_keypoints_dir,
        max_workers=int(args.max_workers),
        max_scenes=args.max_scenes,
        skip_existing=not bool(args.no_skip_existing),
        scene_list=scene_list,
        global_camera_json=args.global_camera_json,
    )
    print(f"Scenes processed: {summary['scenes_processed']}/{summary['scenes_total']}")
    print(f"Total datapoints: {summary['datapoints_total']}")
    print(f"Mean lift validity: {summary['mean_lift_validity_rate']:.4f}")
    print(f"Summary JSON: {args.output_dir / 'dataset_build_summary_v0_1.json'}")


if __name__ == "__main__":
    main()
