"""HTTP sidecar: local Hugging Face VLM **or** Google Gemini API → same delta JSON (Python 3.9+ venv).

Run (from this directory)::

    pip install -r requirements.txt
    python app.py --host 127.0.0.1 --port 8765
    python app.py --vlm-backend gemini --host 127.0.0.1 --port 8765

**Backend:** default is local HF (``VLM_BACKEND`` unset or ``hf``). For Gemini use
``--vlm-backend gemini`` or set ``VLM_BACKEND=gemini``. **Vertex (service account JSON) is
preferred** when a JSON path resolves: ``VLM_GEMINI_SERVICE_ACCOUNT_JSON``, then repo-root
``simtoolreal-93aa22063ba0.json``, then ``GOOGLE_APPLICATION_CREDENTIALS`` (credentials are read
from the file, not via Application Default Credentials). Optional ``GOOGLE_CLOUD_LOCATION``.
**Developer API** (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``) is used only when no JSON is found, or
when ``VLM_GEMINI_FORCE_DEVELOPER_API=1`` with an API key set. Same variables can live in
``Generative_STR/.env`` or ``vlm_sidecar/.env``.
Model id: ``VLM_GEMINI_MODEL`` (default ``gemini-2.5-pro``). **Gemini extra reasoning:** set
``VLM_GEMINI_THINKING_BUDGET`` to ``-1`` (automatic, default) for internal thinking on supported models;
``0`` disables. Optional: ``VLM_GEMINI_THINKING_LEVEL`` (e.g. ``high``) **only if** your model/region
supports it (omit by default—Vertex often returns 400 otherwise). ``VLM_GEMINI_MIN_OUTPUT_TOKENS``
(default ``8192``, applied when thinking is enabled so the JSON reply is not truncated).
``VLM_GEMINI_INCLUDE_THOUGHTS`` defaults to ``1``: thought text is included in infer log files (under
``--- gemini_thoughts ---``) while the HTTP ``raw`` field stays the answer JSON only for parsers.

Filesystem infer logs are **off** unless you set form field ``run_log_dir`` (absolute directory on
the sidecar host) or set env ``VLM_INFER_LOG_DIR`` to a directory. There is no default under
``vlm_sidecar/infer_logs/`` (SimToolReal logs under ``simtoolreal/runs/`` from the HTTP JSON instead).
Set ``VLM_INFER_LOG_DISABLE=1`` to force no writes even when those are set. ``VLM_PRINT_FULL_RAW=1``
dumps the full decode to stdout.

SimToolReal ``llm_goal_env`` POSTs image + pose JSON + optional ``infer_history_json`` (prior turns).
Optional ``task_description`` is not put in the VLM text.

**Five-image mode:** multipart field ``image`` plus **all four** of ``image_aux_0`` … ``image_aux_3`` (PNG/JPEG).
The model sees, in order: main scene, then four zoomed tool-only probes (current pose and +45° body +X/+Y/+Z).
``llm_goal_env`` sends these by default; set ``VLM_FIVE_VIEW=0`` in the client process to send only the main image.

Local HF startup **requires a CUDA GPU** unless ``VLM_DEVICE=cpu`` or ``VLM_ALLOW_CPU=1`` (CPU is very slow).
Gemini mode does not load torch models. Qwen ``enable_thinking`` in the chat template is **off** by default;
set ``VLM_ENABLE_THINKING=1`` to turn on (HF only).

**Qwen3.5 speed:** ``requirements.txt`` includes ``flash-linear-attention`` and ``causal-conv1d`` so the
model’s hybrid blocks can use the CUDA fast path. On GPU load we also enable TF32 and high matmul precision.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from delta_parse import (
    extract_motion_delta_snippet_for_feedback,
    parse_delta_json,
    parse_phase_label,
    parse_plan_steps,
    parse_reasoning_summary,
    parse_rotation_satisfied,
)
import pose_prompt_ghost
import pose_prompt_sweep


def _load_vlm_dotenv_files() -> None:
    """Load ``.env`` from repo root then ``vlm_sidecar/.env`` (``override=False``; existing os.environ wins)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for p in (here.parent / ".env", here / ".env"):
        if p.is_file():
            load_dotenv(p, override=False)


_load_vlm_dotenv_files()


def _prompt_profile() -> str:
    raw = str(_state.get("prompt_profile") or os.environ.get("VLM_PROMPT_PROFILE", "sweep")).strip().lower()
    if raw in ("ghost", "legacy"):
        return "ghost"
    return "sweep"


def _prompt_mod():
    return pose_prompt_ghost if _prompt_profile() == "ghost" else pose_prompt_sweep


def _system_prompt_for_profile() -> str:
    return SYSTEM_PROMPT_GHOST if _prompt_profile() == "ghost" else SYSTEM_PROMPT_SWEEP


def _sanitize_sweep_text(text: str) -> str:
    """Remove legacy ghost phrasing from sweep history to prevent prompt contamination."""
    if _prompt_profile() == "ghost":
        return text
    s = text or ""
    repls = (
        ("green ghost", "sweep objective"),
        ("ghost goal", "task objective"),
        ("ghost", "target"),
        ("overlap", "align"),
        ("pre-grasp", "staging"),
        ("gripper", "brush"),
    )
    out = s
    for a, b in repls:
        out = out.replace(a, b)
        out = out.replace(a.title(), b.title())
        out = out.replace(a.upper(), b.upper())
    return out


def _ensure_cuda_toolchain_env() -> None:
    """Prefer CUDA 13 nvcc over distro /usr/bin/nvcc when running without ``source .venv/bin/activate``.

    Torch cu13 wheels expect a matching toolchain; Ubuntu often ships CUDA 12 ``nvcc`` on PATH first.
    """
    cuda_bin = Path("/usr/local/cuda/bin")
    nvcc = cuda_bin / "nvcc"
    if not nvcc.is_file():
        return
    sep = os.pathsep
    parts = os.environ.get("PATH", "").split(sep)
    bin_str = str(cuda_bin)
    if parts and parts[0] == bin_str:
        return
    os.environ["PATH"] = bin_str + sep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")


_ensure_cuda_toolchain_env()

SYSTEM_PROMPT_GHOST = """You are a visuomotor policy assistant for a robot manipulation scene.

**World coordinate axes (same frame as the printed tool pose and ``dx,dy,dz``):**
The simulator uses **Z-up**: **+Z** is vertical **up** (opposite gravity); the table top lies mostly in the **XY** plane.
**+X** and **+Y** are horizontal (meters). Translation deltas ``dx,dy,dz`` are in this **world** frame—not in pixels.

**Default RGB camera (Viser):** The browser view is set from roughly **negative world Y** toward **positive Y**, with **+Z** upward on screen. So **moving the tool “up” on the image** usually corresponds to **+dz**; **depth into the scene / toward the robot table** often aligns with **+dy** (approximate—use the image and the green ghost together with the numbers).

**Green ghost vs. solid tool:** The semi-transparent **green** layer is a **pose hint**: a ghost copy of the **same** tool mesh at the **goal** pose—not a different object or a generic “green thing” in the scene. The **solid** mesh (blue grips, grey metal, orange arm links, etc.) is the **actual** tool you move; drive it until it **overlaps** the green **tool-shaped** silhouette. The optional **Tool:** line in the user message only names which rigid body is tracked.

**Manipulator:** The arm moves the tool through contact that may **not** be a stable grasp yet (or only partial support)—ignore grasp / finger-close planning. Your job is **only** to choose ``motion_delta`` so the **tool** moves and rotates toward the green ghost.

**Multi-turn chat layout:** Older turns are **above** the latest user message. Each stored turn is **two**
messages in order: (1) **User** — one RGB screenshot plus text: tool pose **after** your previous motion was
applied, and a line restating the ``motion_delta`` we applied from your JSON; (2) **Assistant** — **only**
your prior reply’s JSON (``reasoning_summary`` + ``motion_delta``), replayed verbatim—**already executed**,
not a new command. The **bottom** user block is **this** turn’s fresh RGB + pose: reply with **one** new JSON
for the **next** motion only (never re-paste a prior answer as if it were new). Compare screenshots across
turns to see whether the solid tool is moving toward the green ghost; refine with **moderate** deltas—no
endless re-analysis.

**Past attempts → rotation:** Whenever history exists, **study your earlier choices** (especially past
``drx,dry,drz`` and the RGB after each was applied). Use them to **steer this turn’s rotation**: if a prior
body-axis push under-rotated, over-rotated, or turned the wrong way relative to the green ghost, **adjust**
direction or magnitude now instead of repeating the same mistake. Cross-check each history image: how did
the solid tool’s orientation change vs. what you intended, and what rotation still separates it from the ghost?

**Primary goal:** When that green **tool** ghost is visible, output ``motion_delta`` so the **solid** tool moves and rotates **toward** coinciding with it. If no such ghost is visible, use small cautious deltas.

**Think deeply before you answer:** Mentally work through the scene—RGB (and any rotation references),
printed pose, history if present, and how ``dx,dy,dz`` vs. ``drx,dry,drz`` will move the **solid** tool
relative to the **green** ghost—**before** you commit to numbers. Sanity-check sign/direction (world vs.
body frame) and whether your deltas are consistent with what you see. Only then emit the JSON; use
``reasoning_summary`` for a **short** takeaway of that deliberation (especially rotation), still respecting
the character limit below.

**Output contract (strict):** your entire reply is **one JSON object** — first character `{`, last `}`.
No markdown, no bullet lists, and **no characters before or after** that JSON. Put rationale
**only** inside ``reasoning_summary`` (<= 320 chars).

Use this phased schema:
{"phase":"plan|rotation|direction","reasoning_summary":"","plan_steps":[],"rotation_satisfied":false,"motion_delta":{"dx":0.00,"dy":0.00,"dz":0.00,"drx":0.00,"dry":0.00,"drz":0.00}}

Field rules:
- ``phase`` is required and must be one of ``plan``, ``rotation``, ``direction``.
- ``plan_steps`` is optional, but should be a short string list during ``plan``.
- ``rotation_satisfied`` is optional; set true when orientation is acceptable.
- ``motion_delta`` always includes all six numeric keys (use zeros when not moving).

**Hard rules:** do **not** wrap the JSON in markdown (no `` ``` `` or `` ```json ``). Do **not** emit ``"motion_delta": {}`` — every key ``dx,dy,dz,drx,dry,drz`` must be a numeric literal (use ``0.00`` for no motion).
These are **deltas**, not target orientation values: do not copy the current absolute pose numbers into ``drx,dry,drz``.

Meaning of ``motion_delta``:
- dx,dy,dz: translation delta in **world** frame (meters).
- drx,dry,drz: axis-angle **delta** in **tool body frame** before the motion; angle in **degrees** = vector length.
- Every number: **at most two decimal places**, round normally. No scientific notation.
- Use modest deltas; when the solid tool already matches the green ghost well enough, use zeros."""

SYSTEM_PROMPT_SWEEP = """You are a visuomotor policy assistant for a brush sweeping task.

World frame for dx,dy,dz is Z-up: +Z vertical; +X/+Y horizontal on table.
For this sweep profile, do NOT reason about any ghost/goal overlay. Use only real scene objects:
brush, balls, bin, table.

Task objective: sweep clustered red balls into the green bin with staged control:
- plan: produce qualitative stage plan
- rotation: choose brush orientation for current stage
- direction: choose translation for current stage

Output contract (strict): reply with one JSON object only, no markdown.
Use schema:
{"phase":"plan|rotation|direction","reasoning_summary":"","plan_steps":[],"rotation_satisfied":false,"motion_delta":{"dx":0.00,"dy":0.00,"dz":0.00,"drx":0.00,"dry":0.00,"drz":0.00}}
All six motion_delta keys must be numeric; use 0.00 where unused.
dx,dy,dz are world-frame meters; drx,dry,drz are body-frame axis-angle delta components in degrees.
"""

_state: Dict[str, Any] = {}

DEFAULT_VLM_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def _configure_torch_cuda_for_inference() -> None:
    """Best-effort CUDA matmul/conv tuning (safe no-ops on CPU or older torch)."""
    import torch

    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _preload_qwen_fast_kernels_optional() -> None:
    """Import FLA / causal_conv1d **before** transformers so Qwen3.5 can enable the fused fast path."""
    import importlib.util

    for mod in ("fla", "causal_conv1d"):
        try:
            if importlib.util.find_spec(mod) is not None:
                __import__(mod)
        except Exception:
            pass


def _load_model(model_id: str, device_pref: Optional[str] = None) -> None:
    import torch

    _configure_torch_cuda_for_inference()
    _preload_qwen_fast_kernels_optional()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    raw = (device_pref or "").strip().lower()
    has_cuda = torch.cuda.is_available()
    allow_cpu = raw == "cpu" or _env_true("VLM_ALLOW_CPU", "0")

    if not has_cuda and not allow_cpu:
        raise RuntimeError(
            "vlm_sidecar requires a CUDA GPU (torch.cuda.is_available() is False). "
            "Use a machine with an NVIDIA GPU and working PyTorch CUDA, or set VLM_DEVICE=cpu "
            "or VLM_ALLOW_CPU=1 for CPU-only debugging (very slow)."
        )
    if raw == "cuda" and not has_cuda:
        raise RuntimeError(
            "VLM_DEVICE=cuda was set but torch.cuda.is_available() is False "
            "(check drivers and torch CUDA build)."
        )
    if raw == "cpu":
        device = "cpu"
    else:
        device = "cuda" if has_cuda else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = dtype
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to(device)

    _state["backend"] = "hf"
    _state["processor"] = processor
    _state["model"] = model
    _state["device"] = device
    _state["model_id"] = model_id


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _generate_decode_kwargs(max_new_tokens: int) -> Dict[str, Any]:
    """Decoding for structured JSON (override with env vars).

    Default is **sampling** (``VLM_GREEDY=0``): helps break repetitive chain-of-thought loops; use
    ``VLM_GREEDY=1`` for deterministic greedy runs when debugging.
    """
    kw: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if _env_true("VLM_GREEDY", "0"):
        kw["do_sample"] = False
    else:
        kw["do_sample"] = True
        # Mildly low temp + nucleus + repetition penalty: steer toward JSON without wild token noise.
        kw["temperature"] = float(os.environ.get("VLM_TEMPERATURE", "0.28"))
        kw["top_p"] = float(os.environ.get("VLM_TOP_P", "0.88"))
        kw["repetition_penalty"] = float(os.environ.get("VLM_REPETITION_PENALTY", "1.22"))
    return kw


def _raw_debug_for_parse_error(raw: str, head: int = 2000, tail: int = 2000) -> Dict[str, Any]:
    """422 helper: long CoT is usually at the start; valid JSON is often at the end — show both."""
    n = len(raw)
    detail: Dict[str, Any] = {"raw_total_chars": n}
    if n <= head + tail + 64:
        detail["raw_preview"] = raw
        return detail
    detail["raw_preview_head"] = raw[:head]
    detail["raw_preview_tail"] = raw[-tail:]
    detail["raw_preview"] = (
        raw[:head]
        + f"\n\n...[middle omitted; total_chars={n}]...\n\n"
        + raw[-tail:]
    )
    return detail


_infer_log_lock = threading.Lock()
_infer_log_seq = 0


def _parse_optional_auxiliary_images(
    image_aux_0: Optional[UploadFile],
    image_aux_1: Optional[UploadFile],
    image_aux_2: Optional[UploadFile],
    image_aux_3: Optional[UploadFile],
) -> List[Image.Image]:
    """Return zero or four PIL RGB images; raises ``HTTPException`` if the set is partially filled."""
    ufs = [image_aux_0, image_aux_1, image_aux_2, image_aux_3]
    any_set = any(u is not None for u in ufs)
    all_set = all(u is not None for u in ufs)
    if any_set and not all_set:
        raise HTTPException(
            status_code=400,
            detail="image_aux_0, image_aux_1, image_aux_2, image_aux_3 must all be sent together "
            "(or omit all four for single-image infer).",
        )
    if not all_set:
        return []
    out: List[Image.Image] = []
    for i, uf in enumerate(ufs):
        assert uf is not None
        try:
            raw = uf.file.read()
            if not raw:
                raise ValueError("empty upload")
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image_aux_{i}: {e}"
            ) from e
        out.append(pil)
    return out


def _max_history_turns() -> int:
    try:
        return max(0, min(12, int(os.environ.get("VLM_MAX_HISTORY_TURNS", "3"))))
    except ValueError:
        return 3


def _parse_infer_history_json(
    raw: str, max_turns: int
) -> List[Dict[str, Any]]:
    """Decode ``infer_history_json`` from client.

    Required keys per item: ``pose``, ``delta6``, ``image_png_base64``.
    Optional keys: ``user_text`` and ``assistant_text`` (exact prior turn text),
    plus ``image_name`` (client-side filename label).
    """
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("infer_history_json must be a JSON array")
    if max_turns <= 0:
        return []
    if len(data) > max_turns:
        data = data[-max_turns:]
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"infer_history_json[{i}] must be an object")
        pose = item.get("pose")
        d6 = item.get("delta6")
        b64 = item.get("image_png_base64")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"infer_history_json[{i}].pose must be a length-7 array")
        if not isinstance(d6, list) or len(d6) != 6:
            raise ValueError(f"infer_history_json[{i}].delta6 must be a length-6 array")
        if not isinstance(b64, str) or not b64.strip():
            raise ValueError(
                f"infer_history_json[{i}].image_png_base64 must be a non-empty base64 string"
            )
        try:
            img_bytes = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ValueError(
                f"infer_history_json[{i}].image_png_base64: invalid base64 ({e})"
            ) from e
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        user_text = item.get("user_text")
        if not isinstance(user_text, str):
            user_text = ""
        assistant_text = item.get("assistant_text")
        if not isinstance(assistant_text, str):
            assistant_text = ""
        image_name = item.get("image_name")
        if not isinstance(image_name, str):
            image_name = ""
        out.append(
            {
                "image": pil,
                "pose": np.asarray(pose, dtype=np.float64).reshape(7),
                "delta6": np.asarray(d6, dtype=np.float64).reshape(6),
                "user_text": user_text,
                "assistant_text": assistant_text,
                "image_name": image_name,
            }
        )
    return out


def _serialize_vlm_messages_for_log(
    messages: List[Dict[str, Any]],
    *,
    image_labels_by_id: Optional[Dict[int, str]] = None,
) -> str:
    """Human-readable dump of the exact chat passed to the model (images as placeholders)."""
    lines: List[str] = []
    for mi, msg in enumerate(messages):
        role = msg.get("role", "?")
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"MESSAGE[{mi}] role={role}")
        lines.append("=" * 72)
        content = msg.get("content", [])
        if isinstance(content, str):
            lines.append(content)
            continue
        if not isinstance(content, list):
            lines.append(repr(content))
            continue
        for pi, part in enumerate(content):
            if not isinstance(part, dict):
                lines.append(f"part[{pi}]: {repr(part)}")
                continue
            ptype = part.get("type", "?")
            if ptype == "text":
                lines.append(part.get("text", "") or "")
            elif ptype == "image":
                img = part.get("image")
                if isinstance(img, Image.Image):
                    w, h = img.size
                    label = ""
                    if image_labels_by_id:
                        label = (image_labels_by_id.get(id(img), "") or "").strip()
                    suffix = f" saved_as={label}" if label else ""
                    lines.append(
                        f"[IMAGE part={pi}] PIL RGB {w}x{h}{suffix} (pixel data not inlined in this log)"
                    )
                else:
                    lines.append(f"[IMAGE part={pi}] type={type(img).__name__}")
            else:
                lines.append(f"[{ptype} part={pi}] {repr(part)[:800]}")
    return "\n".join(lines).strip() + "\n"


def _persist_infer_decode(
    raw: str,
    *,
    max_new_tokens: int,
    tool_name: str,
    task_description: str,
    prompt_pose_2dp: Union[str, Dict[str, Any]],
    parse_ok: bool,
    parse_error: Optional[str],
    history_turns_in_prompt: int = 0,
    run_log_dir: Optional[str] = None,
    full_prompt_log: str = "",
) -> Optional[Path]:
    """Write full model decode + metadata to a UTF-8 text file (one per /v1/infer)."""
    if _env_true("VLM_INFER_LOG_DISABLE", "0"):
        return None
    if run_log_dir and str(run_log_dir).strip():
        base = Path(str(run_log_dir).strip()).expanduser().resolve()
    else:
        raw = os.environ.get("VLM_INFER_LOG_DIR", "").strip()
        if not raw:
            return None
        base = Path(raw).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    global _infer_log_seq
    with _infer_log_lock:
        _infer_log_seq += 1
        seq = _infer_log_seq
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
    path = base / f"infer_{ts}_{seq:06d}.txt"
    model_id = _state.get("model_id", "")
    header_lines = [
        f"utc_iso={datetime.now(timezone.utc).isoformat()}",
        f"model_id={model_id}",
        f"max_new_tokens={max_new_tokens}",
        f"parse_ok={parse_ok}",
        f"parse_error={parse_error!r}",
        f"tool_name={tool_name!r}",
        f"task_description={task_description!r}",
        f"input_pose_prompt_2dp={prompt_pose_2dp!r}",
        f"history_turns_in_prompt={history_turns_in_prompt}",
        f"raw_chars={len(raw)}",
        "---",
        "",
    ]
    chunks: List[str] = ["\n".join(header_lines)]
    fp = (full_prompt_log or "").strip()
    if fp:
        chunks.append("--- full_prompt (system + history + current user) ---\n")
        chunks.append(fp)
        if not fp.endswith("\n"):
            chunks.append("\n")
        chunks.append("\n")
    chunks.append("--- model_output ---\n")
    chunks.append(raw if isinstance(raw, str) else str(raw))
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def _emit_infer_console(raw: str, path: Optional[Path]) -> None:
    if not _env_true("VLM_INFER_LOG_QUIET", "0"):
        if path is not None:
            print(
                f"[vlm_sidecar] infer decode {len(raw)} chars -> {path}",
                flush=True,
            )
        elif _env_true("VLM_INFER_LOG_DISABLE", "0"):
            print(
                "[vlm_sidecar] infer log disabled (VLM_INFER_LOG_DISABLE=1)",
                flush=True,
            )
    if _env_true("VLM_PRINT_FULL_RAW", "0"):
        print("----- VLM_PRINT_FULL_RAW begin -----", flush=True)
        print(raw, flush=True)
        print("----- VLM_PRINT_FULL_RAW end -----", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.environ.get("VLM_BACKEND", "hf").strip().lower().replace("-", "_")
    if backend in ("gemini", "google", "google_genai"):
        try:
            from gemini_backend import init_gemini_state
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "google" or (
                e.msg and "google" in str(e.msg).lower()
            ):
                raise RuntimeError(
                    "Gemini backend needs the PyPI package google-genai "
                    "(``pip install google-genai`` in this venv, or full "
                    "``pip install -r requirements.txt``)."
                ) from e
            raise
        init_gemini_state(_state)
    else:
        model_id = os.environ.get("VLM_MODEL_ID", DEFAULT_VLM_MODEL_ID)
        device_override = os.environ.get("VLM_DEVICE")
        _load_model(model_id, device_pref=device_override or None)
    yield
    _state.clear()


app = FastAPI(title="SimToolReal VLM sidecar", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "backend": _state.get("backend", "hf"),
        "model_id": _state.get("model_id"),
        "device": _state.get("device"),
    }


@app.post("/v1/infer")
def v1_infer(
    image: UploadFile = File(..., description="RGB image (PNG or JPEG)"),
    task_description: str = Form(
        "",
        description="Optional SimToolReal task label; **not** included in the VLM user text (logging / API only).",
    ),
    tool_name: str = Form(
        "",
        description="Human-readable tool name only (e.g. Blue Brush); **not** the task name.",
    ),
    object_pose_xyzw: str = Form(
        ...,
        description="JSON array of 7 floats: object pose [x,y,z,qx,qy,qz,qw] (xyzw quaternion)",
    ),
    infer_history_json: str = Form(
        "",
        description="Optional JSON array of prior turns (newest tail; server keeps at most "
        "``VLM_MAX_HISTORY_TURNS``, default 3): "
        '[{"pose":[7 floats],"delta6":[6 floats],"image_png_base64":"<base64>"}, ...]',
    ),
    render_feedback_note: str = Form(
        "",
        description="Optional framing hint from probe renderer (e.g., clipped near x_lower/y_upper).",
    ),
    phase: str = Form(
        "direction",
        description="Requested phase: plan | rotation | rotation_review | direction.",
    ),
    history_summary: str = Form(
        "",
        description="Optional compressed summary of older turns.",
    ),
    rotation_context_json: str = Form(
        "",
        description="Optional JSON context describing rotation proposal/original frame.",
    ),
    image_aux_0: Optional[UploadFile] = File(
        None, description="Optional PNG/JPEG; send with image_aux_1..3 for five-image infer."
    ),
    image_aux_1: Optional[UploadFile] = File(None),
    image_aux_2: Optional[UploadFile] = File(None),
    image_aux_3: Optional[UploadFile] = File(None),
    image_name_main: str = Form(
        "",
        description="Optional label for main image (e.g. frame_00012/01_direction_primary_input_main.png).",
    ),
    image_name_aux_json: str = Form(
        "",
        description="Optional JSON array of 4 labels for auxiliary images (in image_aux_0..3 order).",
    ),
    max_new_tokens: int = Form(4096),
    run_log_dir: str = Form(
        "",
        description="Optional absolute directory on this host for infer decode logs (same layout as when "
        "``VLM_INFER_LOG_DIR`` is set). When empty and ``VLM_INFER_LOG_DIR`` is unset, no infer log file is written.",
    ),
) -> JSONResponse:
    """Run VLM: build user text from pose only, generate, parse delta. Returns raw + delta6 + model_id."""
    backend = _state.get("backend", "hf")
    if backend == "gemini":
        if "gemini_client" not in _state:
            raise HTTPException(status_code=503, detail="Gemini client not initialized")
    elif "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        pose_list: List[float] = json.loads(object_pose_xyzw)
        if len(pose_list) != 7:
            raise ValueError(f"expected 7 floats, got {len(pose_list)}")
        pose7 = np.array(pose_list, dtype=np.float64)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid object_pose_xyzw JSON: {e}"
        ) from e

    try:
        raw_bytes = image.file.read()
        pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    aux_pils = _parse_optional_auxiliary_images(
        image_aux_0, image_aux_1, image_aux_2, image_aux_3
    )
    use_five_image_stack = len(aux_pils) == 4

    history_entries: List[Dict[str, Any]] = []
    if infer_history_json and infer_history_json.strip():
        try:
            history_entries = _parse_infer_history_json(
                infer_history_json.strip(), _max_history_turns()
            )
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid infer_history_json: {e}"
            ) from e

    prior_turns = len(history_entries)
    turn_index = prior_turns + 1
    requested_phase = (phase or "direction").strip().lower()
    if requested_phase not in ("plan", "rotation", "rotation_review", "direction"):
        requested_phase = "direction"
    orientation_reference_quat_xyzw = pose7[3:7].copy()
    if history_entries:
        try:
            orientation_reference_quat_xyzw = np.asarray(
                history_entries[0]["pose"], dtype=np.float64
            ).reshape(7)[3:7].copy()
        except Exception:
            orientation_reference_quat_xyzw = pose7[3:7].copy()
    prompt_mod = _prompt_mod()
    if _prompt_profile() == "sweep":
        user_text = prompt_mod.build_llm_user_text(
            object_pose_xyzw=pose7,
            tool_name=tool_name.strip(),
            task_description=task_description,
            turn_index_1based=turn_index,
            prior_turns_in_context=prior_turns,
            five_image_stack=use_five_image_stack,
            render_feedback_note=render_feedback_note.strip(),
            phase=phase.strip(),
            history_summary=history_summary.strip(),
            rotation_context_json=rotation_context_json.strip(),
            orientation_reference_quat_xyzw=orientation_reference_quat_xyzw,
        )
        prompt_pose_2dp = prompt_mod.prompt_pose_values_2dp(
            pose7, orientation_reference_quat_xyzw=orientation_reference_quat_xyzw
        )
    else:
        user_text = prompt_mod.build_llm_user_text(
            object_pose_xyzw=pose7,
            tool_name=tool_name.strip(),
            task_description=task_description,
            turn_index_1based=turn_index,
            prior_turns_in_context=prior_turns,
            five_image_stack=use_five_image_stack,
            render_feedback_note=render_feedback_note.strip(),
            phase=phase.strip(),
            history_summary=history_summary.strip(),
            rotation_context_json=rotation_context_json.strip(),
        )
        prompt_pose_2dp = prompt_mod.prompt_pose_values_2dp(pose7)

    system_text = _system_prompt_for_profile()
    if use_five_image_stack:
        system_text = system_text + "\n\n" + prompt_mod.SYSTEM_PROMPT_FIVE_IMAGE_APPEND

    aux_image_names: List[str] = []
    if image_name_aux_json and image_name_aux_json.strip():
        try:
            parsed_aux_names = json.loads(image_name_aux_json.strip())
            if not isinstance(parsed_aux_names, list):
                raise ValueError("must be a JSON array")
            aux_image_names = [str(x) for x in parsed_aux_names]
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image_name_aux_json: {e}"
            ) from e
    if aux_image_names and len(aux_image_names) != len(aux_pils):
        raise HTTPException(
            status_code=400,
            detail=(
                f"image_name_aux_json length {len(aux_image_names)} does not match "
                f"aux image count {len(aux_pils)}"
            ),
        )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
    ]
    image_labels_by_id: Dict[int, str] = {}
    for hi, h in enumerate(history_entries):
        pil_h = h["image"]
        pose_h = np.asarray(h["pose"], dtype=np.float64).reshape(7)
        delta_h = np.asarray(h["delta6"], dtype=np.float64).reshape(6)
        hist_image_name = str(h.get("image_name") or "").strip()
        if hist_image_name:
            image_labels_by_id[id(pil_h)] = hist_image_name
        hist_user = _sanitize_sweep_text(str(h.get("user_text") or "").strip())
        if not hist_user:
            hist_user = prompt_mod.build_history_turn_user_text(
                step_index_1based=hi + 1,
                object_pose_xyzw=pose_h,
                delta6_applied=delta_h,
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_h},
                    {"type": "text", "text": hist_user},
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                # Must be a list of parts (like user/system); a bare str makes Qwen3-VL-style
                # templates iterate characters and fail with: string indices must be integers, not 'str'.
                "content": [
                    {
                        "type": "text",
                        "text": _sanitize_sweep_text(str(h.get("assistant_text") or "").strip())
                        or prompt_mod.applied_delta6_to_assistant_json(delta_h),
                    },
                ],
            }
        )
    main_name = (image_name_main or "").strip()
    if main_name:
        image_labels_by_id[id(pil)] = main_name
    last_user_content: List[Dict[str, Any]] = [{"type": "image", "image": pil}]
    for i, ap in enumerate(aux_pils):
        if i < len(aux_image_names) and aux_image_names[i].strip():
            image_labels_by_id[id(ap)] = aux_image_names[i].strip()
        last_user_content.append({"type": "image", "image": ap})
    last_user_content.append({"type": "text", "text": _sanitize_sweep_text(user_text)})
    messages.append({"role": "user", "content": last_user_content})

    full_prompt_log = _serialize_vlm_messages_for_log(
        messages, image_labels_by_id=image_labels_by_id
    )

    img_count = 1 + len(aux_pils)
    print(
        "[vlm_sidecar] sending frame for inference "
        f"profile={_prompt_profile()!r} phase={requested_phase!r} turn={turn_index} history_turns={prior_turns} "
        f"images={img_count} max_new_tokens={max_new_tokens}",
        flush=True,
    )

    try:
        if backend == "gemini":
            from gemini_backend import generate_from_sidecar_messages

            infer_log_raw, raw = generate_from_sidecar_messages(
                client=_state["gemini_client"],
                model_id=str(_state["model_id"]),
                messages=messages,
                max_new_tokens=max_new_tokens,
            )
        else:
            import torch

            processor = _state["processor"]
            model = _state["model"]
            chat_kwargs: Dict[str, Any] = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if _env_true("VLM_ENABLE_THINKING", "0"):
                # Qwen-style processors; set VLM_ENABLE_THINKING=1 to enable chain-of-thought in the template.
                chat_kwargs["enable_thinking"] = True
            else:
                # Jinja defaults enable_thinking to true when undefined — pass False explicitly.
                chat_kwargs["enable_thinking"] = False
            try:
                inputs = processor.apply_chat_template(messages, **chat_kwargs)
            except TypeError:
                chat_kwargs.pop("enable_thinking", None)
                inputs = processor.apply_chat_template(messages, **chat_kwargs)
            dev = next(model.parameters()).device
            tensor_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    tensor_inputs[k] = v.to(dev)
                else:
                    tensor_inputs[k] = v
            gen_kw = _generate_decode_kwargs(max_new_tokens)
            with torch.inference_mode():
                out = model.generate(**tensor_inputs, **gen_kw)
            in_len = tensor_inputs["input_ids"].shape[1]
            gen_ids = out[0, in_len:]
            raw = processor.decode(gen_ids, skip_special_tokens=True)
            infer_log_raw = raw
    except Exception as e:
        label = "Gemini" if backend == "gemini" else "Generation"
        raise HTTPException(status_code=500, detail=f"{label} failed: {e}") from e

    parse_err: Optional[str] = None
    parse_ve: Optional[ValueError] = None
    try:
        if requested_phase == "plan":
            # Plan turn should not depend on motion deltas; ignore if model emits them.
            delta6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            delta = parse_delta_json(raw)
            delta6 = [round(float(x), 2) for x in delta.tolist()]
            if requested_phase in ("rotation", "rotation_review"):
                # Rotation stage is orientation-only.
                delta6[0] = 0.0
                delta6[1] = 0.0
                delta6[2] = 0.0
            elif requested_phase == "direction":
                # Direction stage is translation-only.
                delta6[3] = 0.0
                delta6[4] = 0.0
                delta6[5] = 0.0
        reasoning_summary = parse_reasoning_summary(raw)
        response_phase = parse_phase_label(raw)
        plan_steps = parse_plan_steps(raw)
        rotation_satisfied = parse_rotation_satisfied(raw)
    except ValueError as e:
        parse_err = str(e)
        parse_ve = e
        reasoning_summary = ""
        response_phase = "direction"
        plan_steps = []
        rotation_satisfied = False

    log_path = _persist_infer_decode(
        infer_log_raw,
        max_new_tokens=max_new_tokens,
        tool_name=tool_name.strip(),
        task_description=task_description,
        prompt_pose_2dp=prompt_pose_2dp,
        parse_ok=parse_err is None,
        parse_error=parse_err,
        history_turns_in_prompt=prior_turns,
        run_log_dir=run_log_dir.strip() or None,
        full_prompt_log=full_prompt_log,
    )
    _emit_infer_console(infer_log_raw, log_path)

    if parse_err is not None:
        dbg = _raw_debug_for_parse_error(raw)
        dbg["error"] = parse_err
        dbg["motion_delta_snippet"] = extract_motion_delta_snippet_for_feedback(raw)
        if log_path is not None:
            dbg["infer_log_path"] = str(log_path)
        raise HTTPException(status_code=422, detail=dbg) from parse_ve

    return JSONResponse(
        {
            "raw": raw,
            "delta6": delta6,
            "reasoning_summary": reasoning_summary,
            "phase": response_phase,
            "plan_steps": plan_steps,
            "rotation_satisfied": rotation_satisfied,
            "input_pose_prompt_2dp": prompt_pose_2dp,
            "model_id": _state.get("model_id"),
            "infer_log_path": str(log_path) if log_path is not None else None,
            "history_turns_used": prior_turns,
            "turn_index_1based": turn_index,
            "five_image_stack": use_five_image_stack,
            "full_prompt_log": full_prompt_log,
            "current_user_text": user_text,
            "prompt_profile": _prompt_profile(),
        }
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="VLM HTTP sidecar for SimToolReal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--vlm-backend",
        default=None,
        choices=("hf", "gemini", "google", "google_genai"),
        help="Backend: hf (local) or gemini (google/google_genai are aliases). Sets VLM_BACKEND when passed.",
    )
    parser.add_argument(
        "--prompt-profile",
        default="sweep",
        choices=("sweep", "ghost"),
        help=(
            "Prompt profile: sweep (default, no ghost language) or ghost (legacy alignment prompts). "
            "Overrides VLM_PROMPT_PROFILE for this process."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("VLM_MODEL_ID", DEFAULT_VLM_MODEL_ID),
        help="HF: Hugging Face model id. Ignored when VLM_BACKEND=gemini (use VLM_GEMINI_MODEL).",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Force torch device: cuda | cpu (empty = auto, or set VLM_DEVICE)",
    )
    args = parser.parse_args()
    if args.vlm_backend is not None:
        os.environ["VLM_BACKEND"] = args.vlm_backend
    os.environ["VLM_PROMPT_PROFILE"] = args.prompt_profile
    _state["prompt_profile"] = args.prompt_profile
    os.environ["VLM_MODEL_ID"] = args.model_id
    if args.device:
        os.environ["VLM_DEVICE"] = args.device
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        factory=False,
        reload=False,
    )
