"""Single-scene sim orientation check for dataset_0010 (GT waypoints, 6 goals only).

Builds trajectory from shard datapoint, renders pyrender viz, rolls out in IsaacGym,
and writes a side-by-side PNG for review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description="Verify dataset_0010 sim alignment.")
    p.add_argument(
        "--shard_path",
        type=str,
        default="training/datasets/dataset_0010_brush_sweep_sim/shards/brush_sweep_sim_0000_shard.json",
    )
    p.add_argument("--datapoint_index", type=int, default=0)
    p.add_argument(
        "--output",
        type=str,
        default="training/verification/dataset_0010_brush_sweep_sim_viz/verify_sim_orientation.png",
    )
    p.add_argument("--max_steps", type=int, default=400)
    args = p.parse_args()

    shard_path = Path(args.shard_path)
    if not shard_path.is_absolute():
        shard_path = REPO_ROOT / shard_path
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    dp = None
    for d in shard["datapoints"]:
        if int(d["datapoint_index"]) == int(args.datapoint_index):
            dp = d
            break
    if dp is None:
        raise ValueError(f"datapoint {args.datapoint_index} not found")

    out_dir = REPO_ROOT / "training/verification/dataset_0010_brush_sweep_sim_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "verify_traj.json"
    traj_png = out_dir / "verify_traj_render.png"
    sim_mp4 = out_dir / "verify_sim_rollout.mp4"
    final_png = Path(args.output)
    if not final_png.is_absolute():
        final_png = REPO_ROOT / final_png

    # Build 6-waypoint-only trajectory JSON.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "generative_str_pipeline.sim_rollout.build_brush_trajectory",
            "--shard_path",
            str(shard_path),
            "--datapoint_index",
            str(args.datapoint_index),
            "--steps_per_segment",
            "1",
            "--marker_surface_z",
            str(float(dp["table_xyz_world"][2])),
            "--output",
            str(traj_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    n_goals = int(traj["_meta"]["num_goals"])
    print(f"Trajectory: start + {n_goals} goals (expect 6)")

    # Trajectory render (pyrender, sim table + camera).
    spec = {
        "items": [
            {
                "out_path": str(traj_png),
                "movement_token": "stroke_sweep",
                "scene": dp,
                "waypoints": dp["waypoints"],
            }
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "generative_str_pipeline.sim_rollout.render_rollout_trajectory",
            "--input",
            spec_path,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    # IsaacGym rollout (policy_exec env).
    policy_exec = REPO_ROOT / "policy_exec"
    mat = dp["material_xyz_world"]
    dest = dp["destination_xyz_world"]
    py = Path("/home/ubuntu/miniconda3/envs/policy_exec/bin/python")
    if not py.is_file():
        py = Path(sys.executable)
    cmd = [
        str(py),
        "-m",
        "dextoolbench.rollout_vla_trajectory",
        "--trajectory-path",
        str(traj_path),
        "--output-dir",
        str(out_dir),
        "--isaac-video",
        "--z-offset",
        "0.0",
        "--lift-z",
        "0.0",
        "--start-offset",
        "0.0",
        "0.0",
        "0.0",
    ]
    run_env = os.environ.copy()
    conda_lib = Path("/home/ubuntu/miniconda3/envs/policy_exec/lib")
    if conda_lib.is_dir():
        run_env["LD_LIBRARY_PATH"] = f"{conda_lib}:{run_env.get('LD_LIBRARY_PATH', '')}"
    subprocess.run(cmd, cwd=str(policy_exec), check=True, env=run_env)
    # Rollout saves under output_dir/<timestamp>/1.mp4
    sim_mp4 = None
    for sub in sorted(out_dir.glob("*/1.mp4"), key=lambda p: p.stat().st_mtime):
        sim_mp4 = sub
    if sim_mp4 is None and (out_dir / "0.mp4").exists():
        sim_mp4 = out_dir / "0.mp4"

    # Side-by-side: trajectory render | first sim frame (or stitched if mp4 exists).
    traj_img = Image.open(traj_png).convert("RGB")
    if sim_mp4.exists():
        import imageio.v2 as iio

        reader = iio.get_reader(str(sim_mp4))
        sim_frame = reader.get_data(0)
        reader.close()
        sim_img = Image.fromarray(sim_frame)
    else:
        sim_img = Image.new("RGB", (640, 480), (40, 40, 40))

    fh = traj_img.height
    tw = int(round(traj_img.width * fh / traj_img.height))
    traj_img = traj_img.resize((tw, fh), Image.BILINEAR)
    sw = int(round(sim_img.width * fh / sim_img.height))
    sim_img = sim_img.resize((sw, fh), Image.BILINEAR)
    combo = Image.new("RGB", (tw + sw, fh))
    combo.paste(traj_img, (0, 0))
    combo.paste(sim_img, (tw, 0))
    final_png.parent.mkdir(parents=True, exist_ok=True)
    combo.save(final_png)
    print(f"Wrote {final_png}")
    print(f"  material={mat} destination={dest} table_z={dp['table_xyz_world'][2]}")


if __name__ == "__main__":
    main()
