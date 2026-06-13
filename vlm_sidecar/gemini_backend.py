"""Google Gemini — Vertex AI (service account JSON) preferred; Developer API (API key) optional.

Vertex auth uses ``google.oauth2.service_account`` and the JSON file contents — not Application
Default Credentials discovery. See ``_resolve_vertex_service_account_json`` for path resolution.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# GA flagship multimodal; override with VLM_GEMINI_MODEL (e.g. gemini-3-flash-preview when available).
DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-pro"

_DOTENV_KEYS_FOR_GEMINI = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VLM_GEMINI_SERVICE_ACCOUNT_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "VLM_GEMINI_FORCE_DEVELOPER_API",
)


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _gemini_dotenv_paths() -> Tuple[Path, ...]:
    here = Path(__file__).resolve().parent
    return (here.parent / ".env", here / ".env")


def _hydrate_gemini_env_from_dotenv() -> None:
    """If auth-related env vars are missing or blank-only, fill from repo or sidecar ``.env``."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for key in _DOTENV_KEYS_FOR_GEMINI:
        if (os.environ.get(key) or "").strip():
            continue
        for p in _gemini_dotenv_paths():
            if not p.is_file():
                continue
            vals = dotenv_values(p)
            raw = vals.get(key)
            if raw is None:
                continue
            v = str(raw).strip()
            if v:
                os.environ[key] = v
                break


def resolve_gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def _gac_path() -> str:
    return (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()


def _repo_root_vertex_credentials_path() -> Path:
    """Default service-account JSON checked when ``GOOGLE_APPLICATION_CREDENTIALS`` is unset."""
    return Path(__file__).resolve().parent.parent / "simtoolreal-93aa22063ba0.json"


def _resolve_vertex_service_account_json() -> Optional[Path]:
    """Resolve a Vertex service-account JSON path (explicit file preferred over ``GOOGLE_APPLICATION_CREDENTIALS``).

    Order: ``VLM_GEMINI_SERVICE_ACCOUNT_JSON``, repo-root ``simtoolreal-93aa22063ba0.json``,
    then ``GOOGLE_APPLICATION_CREDENTIALS`` when that file exists. Loads credentials from the
    file directly (no ADC / ``google.auth.default``).
    """
    explicit = (os.environ.get("VLM_GEMINI_SERVICE_ACCOUNT_JSON") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise RuntimeError(
                "VLM_GEMINI_SERVICE_ACCOUNT_JSON is set but the file was not found: "
                f"{p}"
            )
        return p
    d = _repo_root_vertex_credentials_path()
    if d.is_file():
        return d
    gac = _gac_path()
    if gac:
        p = Path(gac).expanduser()
        if not p.is_file():
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is set but the file was not found: "
                f"{p}. Fix the path in your environment or .env."
            )
        return p
    return None


def _vertex_client_from_service_account_json(cred_path: Path) -> genai.Client:
    from google.oauth2 import service_account

    scopes = ("https://www.googleapis.com/auth/cloud-platform",)
    creds = service_account.Credentials.from_service_account_file(
        str(cred_path), scopes=scopes
    )
    project_id = ""
    try:
        with cred_path.open(encoding="utf-8") as f:
            info = json.load(f)
        if isinstance(info, dict):
            project_id = str(info.get("project_id") or "").strip()
    except (OSError, json.JSONDecodeError):
        project_id = ""
    project = (
        (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
        or (os.environ.get("GCLOUD_PROJECT") or "").strip()
        or project_id
    ) or None
    if not project:
        raise RuntimeError(
            "Vertex credentials file did not contain project_id and GOOGLE_CLOUD_PROJECT is unset."
        )
    loc = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip() or None
    kwargs: Dict[str, Any] = {
        "vertexai": True,
        "credentials": creds,
        "project": project,
    }
    if loc:
        kwargs["location"] = loc
    return genai.Client(**kwargs)


def init_gemini_state(state: Dict[str, Any]) -> None:
    _hydrate_gemini_env_from_dotenv()
    model_id = (os.environ.get("VLM_GEMINI_MODEL") or DEFAULT_GEMINI_MODEL_ID).strip()
    state["backend"] = "gemini"
    state["model_id"] = model_id
    state["device"] = None

    api_key = resolve_gemini_api_key()
    force_dev = _env_true("VLM_GEMINI_FORCE_DEVELOPER_API", "0")
    cred_path = _resolve_vertex_service_account_json()

    # Prefer Vertex when a service-account JSON is available (typical when org disables consumer API keys / ADC).
    if cred_path is not None and not (force_dev and bool(api_key)):
        state["gemini_client"] = _vertex_client_from_service_account_json(cred_path)
        return
    if api_key:
        state["gemini_client"] = genai.Client(api_key=api_key)
        return

    raise RuntimeError(
        "VLM_BACKEND=gemini needs a Vertex service-account JSON (preferred): "
        "VLM_GEMINI_SERVICE_ACCOUNT_JSON=/path/to.json, or simtoolreal-93aa22063ba0.json at the "
        "Generative_STR repo root, or GOOGLE_APPLICATION_CREDENTIALS to that file; "
        "optional GOOGLE_CLOUD_LOCATION. Or use GEMINI_API_KEY / GOOGLE_API_KEY for the "
        "Developer API when no JSON is present. Set VLM_GEMINI_FORCE_DEVELOPER_API=1 with an API "
        "key to override Vertex when both are configured."
    )


def split_system_instruction(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    if not messages or messages[0].get("role") != "system":
        return "", messages
    parts = messages[0].get("content") or []
    chunks = [
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    return "\n".join(chunks).strip(), list(messages[1:])


def sidecar_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> List[types.Content]:
    contents: List[types.Content] = []
    for msg in messages:
        role = msg.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        parts: List[types.Part] = []
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(types.Part.from_text(text=str(block.get("text") or "")))
            elif btype == "image":
                im = block.get("image")
                if im is None:
                    continue
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                parts.append(
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
                )
        if parts:
            contents.append(types.Content(role=gemini_role, parts=parts))
    return contents


def _gemini_thinking_disabled() -> bool:
    v = os.environ.get("VLM_GEMINI_THINKING_BUDGET", "-1").strip().lower()
    return v in ("0", "false", "no", "off", "disabled", "none")


def _optional_thinking_config() -> Optional[types.ThinkingConfig]:
    """Extra internal reasoning for supported Gemini / Vertex models (JSON answer stays structured)."""
    if _gemini_thinking_disabled():
        return None
    raw = (os.environ.get("VLM_GEMINI_THINKING_BUDGET") or "-1").strip()
    try:
        budget = int(raw)
    except ValueError:
        budget = -1
    if budget == 0:
        return None
    kw: Dict[str, Any] = {
        "thinking_budget": budget,
        # When true, thought parts are included in the API response (see ``_gemini_infer_log_and_parse``).
        "include_thoughts": _env_true("VLM_GEMINI_INCLUDE_THOUGHTS", "1"),
    }
    # Many Vertex / Gemini builds reject ``thinking_level`` (400). Only send when explicitly set.
    lvl = (os.environ.get("VLM_GEMINI_THINKING_LEVEL") or "").strip()
    if lvl:
        try:
            kw["thinking_level"] = types.ThinkingLevel(lvl)
        except ValueError:
            pass
    return types.ThinkingConfig(**kw)


def _gemini_max_output_tokens(requested: int) -> int:
    """Raise the completion cap when thinking is on (thinking consumes part of the budget)."""
    req = int(requested)
    try:
        floor = int(os.environ.get("VLM_GEMINI_MIN_OUTPUT_TOKENS", "8192").strip())
    except ValueError:
        floor = 8192
    if _optional_thinking_config() is not None:
        return max(req, floor)
    return req


def _generation_config(
    *, system_instruction: str, max_new_tokens: int
) -> types.GenerateContentConfig:
    kw: Dict[str, Any] = {
        "system_instruction": system_instruction,
        "max_output_tokens": _gemini_max_output_tokens(max_new_tokens),
        # Enforce JSON-style output shape to match sidecar parsing contract.
        "response_mime_type": "application/json",
    }
    if _env_true("VLM_GREEDY", "0"):
        kw["temperature"] = 0.0
        kw["top_p"] = 1.0
    else:
        kw["temperature"] = float(os.environ.get("VLM_TEMPERATURE", "0.28"))
        kw["top_p"] = float(os.environ.get("VLM_TOP_P", "0.88"))
    tc = _optional_thinking_config()
    if tc is not None:
        kw["thinking_config"] = tc
    return types.GenerateContentConfig(**kw)


_GEMINI_THOUGHTS_HEADER = "--- gemini_thoughts ---"
_GEMINI_ANSWER_HEADER = "--- gemini_answer ---"


def _gemini_infer_log_and_parse(resp: Any) -> Tuple[str, str]:
    """Split response into (full text for infer logs, answer-only text for JSON parsing).

    When ``include_thoughts`` is on, thought ``Part`` objects are written under ``_GEMINI_THOUGHTS_HEADER``;
    non-thought text parts (the model answer, usually JSON) follow ``_GEMINI_ANSWER_HEADER``.
    If no thought parts are present, both strings equal the concatenated answer text.
    """
    thought_chunks: List[str] = []
    answer_chunks: List[str] = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        for p in parts or []:
            pt = getattr(p, "text", None)
            if not isinstance(pt, str) or not pt.strip():
                continue
            if bool(getattr(p, "thought", False)):
                thought_chunks.append(pt.strip())
            else:
                answer_chunks.append(pt.strip())
    parse_text = "\n".join(answer_chunks).strip()
    if not parse_text:
        parse_text = (getattr(resp, "text", None) or "").strip()
    if not parse_text:
        fb = getattr(resp, "prompt_feedback", None)
        raise RuntimeError(f"Gemini returned empty text (prompt_feedback={fb!r})")

    if not thought_chunks:
        return parse_text, parse_text

    thoughts_body = "\n\n".join(thought_chunks)
    infer_log = (
        f"{_GEMINI_THOUGHTS_HEADER}\n{thoughts_body}\n\n"
        f"{_GEMINI_ANSWER_HEADER}\n{parse_text}"
    )
    return infer_log, parse_text


def generate_from_sidecar_messages(
    *,
    client: genai.Client,
    model_id: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[str, str]:
    """Returns ``(infer_log_text, parse_text)`` — infer log may prefix Gemini thoughts; parse JSON from ``parse_text``."""
    system_instruction, rest = split_system_instruction(messages)
    contents = sidecar_messages_to_gemini_contents(rest)
    if not contents:
        raise ValueError("no Gemini contents after conversion (empty user/model turns)")
    cfg = _generation_config(
        system_instruction=system_instruction, max_new_tokens=max_new_tokens
    )
    resp = client.models.generate_content(
        model=model_id, contents=contents, config=cfg
    )
    return _gemini_infer_log_and_parse(resp)
