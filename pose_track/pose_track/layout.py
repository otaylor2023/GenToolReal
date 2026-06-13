from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """``Generative_STR`` root (parent of the ``pose_track`` project directory)."""
    return Path(__file__).resolve().parents[2]


def project_dir() -> Path:
    """Directory containing ``pyproject.toml`` for this package."""
    return Path(__file__).resolve().parents[1]


def default_run_dir(name: str = "pilot_r4p8") -> Path:
    return project_dir() / "runs" / name


def ensure_stages(run_dir: Path) -> None:
    """Create STAGES-style folders under ``run_dir``."""
    for rel in (
        "capture/rgb",
        "capture/depth",
        "capture/masks",
        "video_gen/rgb",
        "mde_raw",
        "aligned_rgbd/rgb",
        "aligned_rgbd/depth",
        "mesh",
        "foundationpose",
        "meta",
        "trajectories",
    ):
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
