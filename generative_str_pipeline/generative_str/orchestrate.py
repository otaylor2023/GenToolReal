"""Expand sweep matrix YAML into run directories (pilot helper).

Full orchestration of external model CLIs is intentionally left to your
``models.yaml`` commands; this module only materializes the run grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class SweepMatrix:
    capture_ids: List[str]
    video_models: List[str]
    mde_models: List[str]
    seeds: List[int]


def load_sweep(path: Path) -> SweepMatrix:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SweepMatrix(
        capture_ids=list(raw["capture_ids"]),
        video_models=list(raw["video_models"]),
        mde_models=list(raw["mde_models"]),
        seeds=list(raw.get("seeds", [0])),
    )


def expand_runs(matrix: SweepMatrix, base: Path) -> List[Path]:
    """Create run roots: ``base/<capture>/<video>/<mde>/seed_<s>/`` with meta only."""
    created: List[Path] = []
    for c in matrix.capture_ids:
        for v in matrix.video_models:
            for m in matrix.mde_models:
                for s in matrix.seeds:
                    root = base / c / v / m / f"seed_{s}"
                    meta = root / "meta"
                    meta.mkdir(parents=True, exist_ok=True)
                    (meta / "sweep.json").write_text(
                        json.dumps(
                            {
                                "capture_id": c,
                                "video_model": v,
                                "mde_model": m,
                                "seed": s,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    created.append(root)
    return created


def materialize_from_yaml(sweep_yaml: Path, output_base: Path) -> List[Path]:
    return expand_runs(load_sweep(sweep_yaml), output_base)
