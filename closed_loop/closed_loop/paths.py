"""Package asset paths (checkpoint, norm stats, control frames, CLIP cache)."""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def assets_dir() -> Path:
    return package_root() / "assets"


def default_checkpoint_path() -> Path:
    return assets_dir() / "checkpoint_best.pt"


def default_normalization_stats_path() -> Path:
    return assets_dir() / "normalization_stats.json"


def control_frames_dir() -> Path:
    return assets_dir() / "control_frames"


def resolve_control_frame(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    candidate = control_frames_dir() / f"{name_or_path}.json"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Control frame not found: {name_or_path!r} (tried {candidate})"
    )


def list_control_frames() -> dict[str, Path]:
    """Return ``{stem: path}`` for every JSON under ``assets/control_frames/``."""
    out: dict[str, Path] = {}
    for path in sorted(control_frames_dir().glob("*.json")):
        out[path.stem] = path.resolve()
    return out


def default_clip_cache_dir() -> Path:
    return assets_dir() / "clip_cache"


def apply_clip_cache_env(cache_dir: Path | None = None) -> Path:
    """Point Hugging Face caches at packaged clip_cache (or given dir)."""
    root = (cache_dir or default_clip_cache_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    transformers_cache = root / "transformers"
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    return root
