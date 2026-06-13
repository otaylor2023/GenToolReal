"""Load Hugging Face and optional W&B credentials from the repo `.env` before training."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_hf_token_from_dotenv() -> None:
    """Merge `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and `WANDB_API_KEY` from `<repo>/.env` if unset."""
    env_path = repo_root() / ".env"
    if not env_path.is_file():
        return
    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY") and val:
            parsed[key] = val
    if not os.environ.get("HF_TOKEN") and "HF_TOKEN" in parsed:
        os.environ["HF_TOKEN"] = parsed["HF_TOKEN"]
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN") and "HUGGING_FACE_HUB_TOKEN" in parsed:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed["HUGGING_FACE_HUB_TOKEN"]
    # Hub accepts either; mirror so both are populated when only one is in `.env`.
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    if not os.environ.get("WANDB_API_KEY") and "WANDB_API_KEY" in parsed:
        os.environ["WANDB_API_KEY"] = parsed["WANDB_API_KEY"]


def apply_hf_env() -> None:
    """Merge `.env` HF token and optional W&B API key into the process environment (idempotent)."""
    load_hf_token_from_dotenv()


def apply_hf_cache(hf_cache_dir: str) -> Path:
    """Point Hugging Face Hub + Transformers at `hf_cache_dir` so weights reuse disk cache.

    Sets `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` for this process before any
    `from_pretrained` calls. Matches the layout used by default `~/.cache/huggingface`.
    """
    root = Path(hf_cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    transformers_cache = root / "transformers"
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    return root
