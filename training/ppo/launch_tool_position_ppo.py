"""Launch helper for tool-position PPO training."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class LaunchArgs:
    config: Path = Path("training/cfg/tool_position_ppo.yaml")
    python_bin: str = "python"


def main() -> None:
    args = tyro.cli(LaunchArgs)
    cmd = f"{args.python_bin} -m training.ppo.train_tool_position_ppo --config {args.config}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

