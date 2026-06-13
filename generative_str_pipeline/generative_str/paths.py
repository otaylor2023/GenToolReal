"""Standard directory layout for a single pipeline run."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for one run root (e.g. runs/capture/video/mde/fp)."""

    root: Path

    @property
    def capture(self) -> Path:
        return self.root / "capture"

    @property
    def video_gen(self) -> Path:
        return self.root / "video_gen"

    @property
    def mde_raw(self) -> Path:
        return self.root / "mde_raw"

    @property
    def aligned_rgbd(self) -> Path:
        return self.root / "aligned_rgbd"

    @property
    def mesh(self) -> Path:
        return self.root / "mesh"

    @property
    def foundationpose(self) -> Path:
        return self.root / "foundationpose"

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    def ensure_meta(self) -> None:
        self.meta.mkdir(parents=True, exist_ok=True)
