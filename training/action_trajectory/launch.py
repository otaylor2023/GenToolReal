"""Launch helper for waypoint-trajectory action training."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class LaunchArgs:
    config: Path = Path("training/cfg/action_trajectory_smoke.yaml")
    python_bin: str = "python"


def main() -> None:
    args = tyro.cli(LaunchArgs)
    cmd = f"{args.python_bin} -m training.action_trajectory.train --config {args.config}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
