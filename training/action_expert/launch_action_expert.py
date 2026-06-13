"""Launch helper for action expert training."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class LaunchArgs:
    config: Path = Path("training/cfg/tool_position_action_expert.yaml")
    python_bin: str = "python"


def main() -> None:
    args = tyro.cli(LaunchArgs)
    cmd = f"{args.python_bin} -m training.action_expert.train_action_expert --config {args.config}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

