"""Model registry: resolve named checkpoints + normalization stats in ``assets/``.

The package historically shipped a single default checkpoint
(``assets/checkpoint_best.pt`` + ``assets/normalization_stats.json``). To support
multiple deployable VLAs side by side (e.g. the single-task brush model and the
joint all-tasks pretrain), every packaged checkpoint is described in
``assets/model_registry.json``. Architecture hyperparameters are still read from
each checkpoint's own ``config`` dict at load time; the registry only locates the
files and records provenance/metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from closed_loop.paths import assets_dir


def registry_path() -> Path:
    return assets_dir() / "model_registry.json"


def _load_registry() -> Dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RegisteredModel:
    key: str
    description: str
    checkpoint_path: Path
    normalization_stats_path: Path
    config_yaml_path: Path | None
    default_control_frame: str
    tasks: List[str]
    metadata: Dict[str, Any]


def list_models() -> List[str]:
    return sorted(_load_registry().get("models", {}).keys())


def default_model_key() -> str:
    reg = _load_registry()
    return str(reg.get("default") or next(iter(reg.get("models", {}))))


def resolve_model(key: str | None = None) -> RegisteredModel:
    """Resolve a registry key to absolute asset paths (no model is loaded)."""
    reg = _load_registry()
    models = reg.get("models", {})
    if key is None:
        key = default_model_key()
    if key not in models:
        raise KeyError(f"Unknown model key {key!r}. Available: {sorted(models)}")
    entry = models[key]
    a = assets_dir()

    ckpt = a / entry["checkpoint"]
    stats = a / entry["normalization_stats"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint for {key!r} not found: {ckpt}")
    if not stats.is_file():
        raise FileNotFoundError(f"Normalization stats for {key!r} not found: {stats}")
    cfg_yaml = entry.get("config_yaml")
    cfg_path = (a / cfg_yaml) if cfg_yaml else None

    return RegisteredModel(
        key=key,
        description=str(entry.get("description", "")),
        checkpoint_path=ckpt.resolve(),
        normalization_stats_path=stats.resolve(),
        config_yaml_path=cfg_path.resolve() if cfg_path else None,
        default_control_frame=str(entry.get("default_control_frame", "blue_brush")),
        tasks=list(entry.get("tasks", [])),
        metadata={k: v for k, v in entry.items()},
    )


def load_policy(key: str | None = None, **brush_policy_kwargs):
    """Build a :class:`closed_loop.inference.BrushPolicy` from a registry key.

    ``brush_policy_kwargs`` are forwarded to ``BrushPolicy`` (e.g. ``device``,
    ``control_frame``, ``instruction``). ``checkpoint_path`` /
    ``normalization_stats_path`` are filled from the registry unless overridden.
    """
    from closed_loop.inference import BrushPolicy

    model = resolve_model(key)
    brush_policy_kwargs.setdefault("checkpoint_path", model.checkpoint_path)
    brush_policy_kwargs.setdefault(
        "normalization_stats_path", model.normalization_stats_path
    )
    brush_policy_kwargs.setdefault("control_frame", model.default_control_frame)
    return BrushPolicy(**brush_policy_kwargs)


def load_closed_loop_policy(key: str | None = None, **closed_loop_kwargs):
    """Build a :class:`closed_loop.policy.ClosedLoopBrushPolicy` from a registry key."""
    from closed_loop.policy import ClosedLoopBrushPolicy

    model = resolve_model(key)
    closed_loop_kwargs.setdefault("checkpoint_path", model.checkpoint_path)
    closed_loop_kwargs.setdefault(
        "normalization_stats_path", model.normalization_stats_path
    )
    closed_loop_kwargs.setdefault("control_frame", model.default_control_frame)
    return ClosedLoopBrushPolicy(**closed_loop_kwargs)
