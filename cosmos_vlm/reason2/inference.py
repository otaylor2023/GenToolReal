from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from cosmos_vlm.reason2.prompting import build_reason_messages


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    loose = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if loose:
        return json.loads(loose.group(1))

    raise ValueError("model output did not contain parseable JSON")


def run_reason_plan(
    image_path: Path,
    model_id_or_path: str,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
    task_description: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    torch_dtype = _dtype_from_name(dtype_name)

    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_id_or_path)

    messages = build_reason_messages(image_path=image_path, task_description=task_description)
    prompt_text = ""
    for msg in messages:
        role = msg.get("role", "")
        text_parts: list[str] = []
        for c in msg.get("content", []):
            if c.get("type") == "text":
                text_parts.append(c.get("text", ""))
            if c.get("type") == "image":
                text_parts.append(f"[image] {c.get('image', '')}")
        if text_parts:
            prompt_text += f"[{role}]\n" + "\n".join(text_parts) + "\n\n"
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    elapsed = time.perf_counter() - t0

    parse_error = ""
    try:
        parsed = _extract_json(decoded)
    except ValueError as exc:
        parsed = {}
        parse_error = str(exc)

    normalized = {
        "scene_summary": parsed.get("scene_summary", ""),
        "trajectory_steps": parsed.get("trajectory_steps", []),
        "motion_prompt_for_video": parsed.get("motion_prompt_for_video", ""),
        "parse_error": parse_error,
    }
    meta = {
        "model_id_or_path": model_id_or_path,
        "device": str(model.device),
        "dtype": dtype_name,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": elapsed,
        "prompt_messages": messages,
    }
    return decoded, normalized, meta, prompt_text.rstrip() + "\n"

