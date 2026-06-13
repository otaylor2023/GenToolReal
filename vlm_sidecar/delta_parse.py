"""Parse 6D delta JSON from model output. Keep logic aligned with SimToolReal expectations."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, Optional

import numpy as np


# Qwen3 / Qwen3.5 templates may still emit empty thinking wrappers when enable_thinking is false.
_THINK_STRIP_PATTERNS = (
    r"<think>[\s\S]*?</think>",
    r"<thinking>[\s\S]*?</thinking>",
)


def normalize_model_output_text(text: str) -> str:
    """Strip Qwen/HF thinking wrappers and markdown fences so JSON extraction runs on the payload."""
    t = text
    for _ in range(8):
        prev = t
        for pat in _THINK_STRIP_PATTERNS:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        if t == prev:
            break
    t = t.strip()
    while True:
        m = re.match(r"^```(?:json)?\s*", t, flags=re.IGNORECASE)
        if not m:
            break
        t = t[m.end() :].lstrip()
    while True:
        m = re.search(r"\s*```\s*$", t)
        if not m:
            break
        t = t[: m.start()].rstrip()
    return t.strip()


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing `` ``` `` / `` ```json `` wrappers (also used internally)."""
    t = text.strip()
    while True:
        m = re.match(r"^```(?:json)?\s*", t, flags=re.IGNORECASE)
        if not m:
            break
        t = t[m.end() :].lstrip()
    while True:
        m = re.search(r"\s*```\s*$", t)
        if not m:
            break
        t = t[: m.start()].rstrip()
    return t.strip()


def _iter_decoded_json_objects(text: str) -> Iterator[Any]:
    """Yield JSON values found by scanning for ``{`` / ``[`` and ``JSONDecoder.raw_decode``."""
    t = text.strip()
    dec = json.JSONDecoder()
    n = len(t)
    i = 0
    while i < n:
        while i < n and t[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(t, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj
        i = end
        while i < n and t[i] in " \t\r\n,":
            i += 1


def _iter_json_objects_brace_scan(text: str) -> Iterator[Any]:
    """Fallback: every top-level balanced ``{...}`` substring that parses as JSON."""
    t = text.strip()
    n = len(t)
    i = 0
    while i < n:
        if t[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    chunk = t[i : j + 1]
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break


def _repair_common_delta_typos(text: str) -> str:
    """Repair common malformed key-value tokens from model outputs before parsing.

    Examples repaired:
    - ``"rx:-10.0"`` -> ``"rx":-10.0``
    - ``"drz: 5"`` -> ``"drz": 5``
    """
    t = text
    # Broken quoted key+value token (missing quote before colon), often emitted inside motion_delta.
    t = re.sub(
        r'"(d?rx|d?ry|d?rz|dx|dy|dz)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"',
        r'"\1":\2',
        t,
        flags=re.IGNORECASE,
    )
    return t


def _delta_dict_from_parsed(obj: Any) -> Optional[dict]:
    """Return the inner dict holding dx..rz (flat or under ``motion_delta`` / common aliases)."""
    if isinstance(obj, dict):
        if all(k in obj for k in ("dx", "dy", "dz", "rx", "ry", "rz")):
            return obj
        if all(k in obj for k in ("dx", "dy", "dz", "drx", "dry", "drz")):
            return obj
        for nest_key in ("motion_delta", "delta", "action", "tool_delta", "pose_delta"):
            sub = obj.get(nest_key)
            if isinstance(sub, dict) and all(
                k in sub for k in ("dx", "dy", "dz", "rx", "ry", "rz")
            ):
                return sub
            if isinstance(sub, dict) and all(
                k in sub for k in ("dx", "dy", "dz", "drx", "dry", "drz")
            ):
                return sub
    if isinstance(obj, list):
        for el in obj:
            d = _delta_dict_from_parsed(el)
            if d is not None:
                return d
    return None


def _regex_extract_motion_delta_numbers(text: str) -> Optional[dict]:
    """Last resort: ``"dx": -1.23``-style pairs; use **last** match per key."""
    keys = ("dx", "dy", "dz", "rx", "ry", "rz")
    out: Dict[str, float] = {}
    for k in keys:
        aliases = (k,)
        if k == "rx":
            aliases = ("rx", "drx")
        elif k == "ry":
            aliases = ("ry", "dry")
        elif k == "rz":
            aliases = ("rz", "drz")
        pat = r"|".join(
            [rf'"{ak}"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)' for ak in aliases]
        )
        ms = list(re.finditer(pat, text))
        if not ms:
            return None
        g = next((x for x in ms[-1].groups() if x is not None), None)
        if g is None:
            return None
        out[k] = float(g)
    return out


def _extract_delta_dict(text: str) -> Optional[dict]:
    """Find a JSON object (noisy / fenced output) that contains dx..rz under ``motion_delta`` or flat."""
    t = _repair_common_delta_typos(
        _strip_markdown_fences(normalize_model_output_text(text))
    )
    for parsed in _iter_decoded_json_objects(t):
        d = _delta_dict_from_parsed(parsed)
        if d is not None:
            return d
    for parsed in _iter_json_objects_brace_scan(t):
        d = _delta_dict_from_parsed(parsed)
        if d is not None:
            return d
    d = _regex_extract_motion_delta_numbers(t)
    if d is not None:
        return d
    # Some model outputs escape object keys inside ``motion_delta`` like ``\\\"dx\\\"``.
    # Recover by unescaping a copy and re-running extraction.
    t_unescaped = (
        t.replace(r"\\\"", '"')
        .replace(r"\"", '"')
        .replace(r"\\n", "\n")
        .replace(r"\n", "\n")
    )
    t_unescaped = _repair_common_delta_typos(t_unescaped)
    for parsed in _iter_decoded_json_objects(t_unescaped):
        d = _delta_dict_from_parsed(parsed)
        if d is not None:
            return d
    for parsed in _iter_json_objects_brace_scan(t_unescaped):
        d = _delta_dict_from_parsed(parsed)
        if d is not None:
            return d
    return _regex_extract_motion_delta_numbers(t_unescaped)


def parse_delta_json(text: str) -> np.ndarray:
    """Extract JSON object with dx..rz from model output (tolerates preamble / markdown / thinking tags)."""
    d = _extract_delta_dict(text)
    if d is not None:
        try:
            arr = np.array(
                [
                    float(d["dx"]),
                    float(d["dy"]),
                    float(d["dz"]),
                    float(d["rx"] if "rx" in d else d["drx"]),
                    float(d["ry"] if "ry" in d else d["dry"]),
                    float(d["rz"] if "rz" in d else d["drz"]),
                ],
                dtype=np.float64,
            )
            return arr
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Missing or invalid keys in JSON: {d}") from e
    raise ValueError(
        "No valid 6D delta JSON found (need motion_delta with all of dx..rz, not empty {}). "
        f"Raw head: {text[:400]!r}"
    )


def extract_motion_delta_snippet_for_feedback(text: str, max_chars: int = 600) -> str:
    """Return a short ``motion_delta`` object substring for retry prompts when JSON is invalid.

    Uses brace matching from the first ``"motion_delta"`` key; tolerates truncated / malformed objects.
    """
    t = normalize_model_output_text(text)
    m = re.search(r'"motion_delta"\s*:', t, flags=re.IGNORECASE)
    if not m:
        return ""
    i = m.end()
    while i < len(t) and t[i] in " \t\r\n":
        i += 1
    if i >= len(t):
        return ""
    if t[i] != "{":
        tail = t[m.start() :].strip()
        return tail[:max_chars] + ("..." if len(tail) > max_chars else "")
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                chunk = t[i : j + 1]
                return chunk if len(chunk) <= max_chars else chunk[: max_chars - 3] + "..."
    chunk = t[i : min(len(t), i + max_chars)]
    return chunk + ("..." if i + max_chars < len(t) else "")


def parse_reasoning_summary(text: str) -> str:
    """Best-effort extraction of optional reasoning_summary from the outer JSON object."""
    t = _strip_markdown_fences(normalize_model_output_text(text))
    for obj in _iter_decoded_json_objects(t):
        if isinstance(obj, dict):
            v = obj.get("reasoning_summary")
            if isinstance(v, str) and v.strip():
                return v.strip()[:1000]
    for obj in _iter_json_objects_brace_scan(t):
        if isinstance(obj, dict):
            v = obj.get("reasoning_summary")
            if isinstance(v, str) and v.strip():
                return v.strip()[:1000]
    return ""


def parse_phase_label(text: str) -> str:
    """Best-effort extraction of top-level phase; defaults to 'direction'."""
    t = _strip_markdown_fences(normalize_model_output_text(text))
    for obj in _iter_decoded_json_objects(t):
        if isinstance(obj, dict):
            v = obj.get("phase")
            if isinstance(v, str) and v.strip():
                p = v.strip().lower()
                if p in ("plan", "rotation", "direction"):
                    return p
    for obj in _iter_json_objects_brace_scan(t):
        if isinstance(obj, dict):
            v = obj.get("phase")
            if isinstance(v, str) and v.strip():
                p = v.strip().lower()
                if p in ("plan", "rotation", "direction"):
                    return p
    return "direction"


def parse_plan_steps(text: str) -> list:
    """Extract optional plan_steps list of short strings."""
    t = _strip_markdown_fences(normalize_model_output_text(text))
    for obj in _iter_decoded_json_objects(t):
        if isinstance(obj, dict):
            v = obj.get("plan_steps")
            if isinstance(v, list):
                out = [str(s).strip() for s in v if str(s).strip()]
                return out[:12]
    for obj in _iter_json_objects_brace_scan(t):
        if isinstance(obj, dict):
            v = obj.get("plan_steps")
            if isinstance(v, list):
                out = [str(s).strip() for s in v if str(s).strip()]
                return out[:12]
    return []


def parse_rotation_satisfied(text: str) -> bool:
    """Extract optional rotation_satisfied boolean; defaults False."""
    t = _strip_markdown_fences(normalize_model_output_text(text))
    for obj in _iter_decoded_json_objects(t):
        if isinstance(obj, dict):
            v = obj.get("rotation_satisfied")
            if isinstance(v, bool):
                return v
    for obj in _iter_json_objects_brace_scan(t):
        if isinstance(obj, dict):
            v = obj.get("rotation_satisfied")
            if isinstance(v, bool):
                return v
    return False
