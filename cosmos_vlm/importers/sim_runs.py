from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cosmos_vlm.artifacts.io import copy_file, sha256_file


def parse_log_rows(log_jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log_jsonl_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def find_latest_sim_session(sim_runs_root: Path) -> Path:
    root = sim_runs_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sim runs root not found: {root}")
    candidates = sorted([p for p in root.glob("vlm_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no sim sessions found under: {root}")
    return candidates[-1]


def resolve_sim_image(sim_session_dir: Path, step: int | None = None) -> tuple[Path, dict[str, Any]]:
    session = sim_session_dir.resolve()
    log_path = session / "log.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"missing sim log file: {log_path}")

    rows = parse_log_rows(log_path)
    if not rows:
        raise ValueError(f"log file has no rows: {log_path}")

    chosen = rows[-1] if step is None else next((r for r in rows if int(r.get("step", -1)) == step), None)
    if chosen is None:
        raise ValueError(f"step {step} not found in {log_path}")

    rel = chosen.get("rgb_image")
    if not isinstance(rel, str) or not rel:
        raise ValueError("selected log row missing 'rgb_image'")

    image_path = (session / rel).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"sim rgb image missing: {image_path}")

    meta = {
        "source_type": "sim_runs_log",
        "sim_session_dir": str(session),
        "log_jsonl": str(log_path),
        "selected_step": int(chosen.get("step", -1)),
        "rgb_image_relative": rel,
        "frame_dir": chosen.get("frame_dir"),
    }
    return image_path, meta


def import_sim_image(sim_session_dir: Path, destination_path: Path, step: int | None = None) -> dict[str, Any]:
    image_path, meta = resolve_sim_image(sim_session_dir, step=step)
    copy_file(image_path, destination_path)
    meta.update(
        {
            "source_path": str(image_path),
            "copied_to": str(destination_path.resolve()),
            "sha256": sha256_file(destination_path),
        }
    )
    return meta


def import_latest_sim_image(sim_runs_root: Path, destination_path: Path, step: int | None = None) -> dict[str, Any]:
    session = find_latest_sim_session(sim_runs_root)
    meta = import_sim_image(session, destination_path, step=step)
    meta["sim_runs_root"] = str(sim_runs_root.resolve())
    return meta

