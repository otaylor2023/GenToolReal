"""Multi-backend I2V benchmark (optional); not part of the default ``gstr`` CLI."""

from __future__ import annotations

import tyro

from generative_str.run_video_gen import BenchmarkVideoGen, run_benchmark_video_gen


def main() -> None:
    args = tyro.cli(BenchmarkVideoGen, prog="gstr-benchmark-video-gen")
    p = run_benchmark_video_gen(args)
    print(p)


if __name__ == "__main__":
    main()
