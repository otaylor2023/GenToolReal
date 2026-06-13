from __future__ import annotations

from pathlib import Path


def _next_counter_dir(base_dir: Path, prefix: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    max_seen = -1
    for p in base_dir.glob(f"{prefix}[0-9][0-9][0-9][0-9]"):
        if not p.is_dir():
            continue
        suffix = p.name.removeprefix(prefix)
        if suffix.isdigit():
            max_seen = max(max_seen, int(suffix))
    next_idx = max_seen + 1
    out = base_dir / f"{prefix}{next_idx:04d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def ensure_run_dir(base_runs_dir: Path, run_id: str | None = None) -> Path:
    if run_id:
        run_dir = base_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return _next_counter_dir(base_runs_dir, "run_")


def inputs_dir(run_dir: Path) -> Path:
    d = run_dir / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(run_dir: Path, stage_name: str) -> Path:
    d = run_dir / stage_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def reason_dir(run_dir: Path) -> Path:
    return stage_dir(run_dir, "reason")


def next_predict_dir(run_dir: Path) -> Path:
    return _next_counter_dir(run_dir, "predict_")


def latest_predict_dir(run_dir: Path) -> Path | None:
    preds = sorted([p for p in run_dir.glob("predict_[0-9][0-9][0-9][0-9]") if p.is_dir()])
    return preds[-1] if preds else None

