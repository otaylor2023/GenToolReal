"""Tyro CLI: ``gstr`` or ``python -m generative_str``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import tyro

from generative_str.capture_from_data import CaptureFromDataArgs, materialize_capture
from generative_str.eval.mde_metrics import depth_rmse
from generative_str.eval.pose_trajectory_diff import compare_trajectories
from generative_str.eval.video_metrics import frame_consistency_report
from generative_str.foundationpose_io import flatten_foundationpose_export
from generative_str.mde_align import align_run_directory
from generative_str.run_mesh_sam3d import (
    RunMeshFromFrame,
    RunMeshSam3d,
    run_mesh_from_frame,
    run_mesh_sam3d,
)
from generative_str.run_mde import RunMde, run_mde
from generative_str.run_video_gen import RunVideoGen, run_video_gen


@dataclass
class AlignDepth:
    """Scale MDE to GT at t=0 and write aligned_rgbd/."""

    capture_dir: Path
    mde_raw_dir: Path
    aligned_dir: Path
    mask: Optional[Path] = None
    robust: bool = False


@dataclass
class FlattenPoses:
    """Convert FoundationPose JSON to flat poses_robot list."""

    src: Path
    dst: Path


@dataclass
class EvalVideo:
    """Static-region MAE vs frame 0 for generated frames."""

    frame0: Path
    frames_dir: Path
    mask: Optional[Path] = None


@dataclass
class EvalMde:
    """RMSE vs dense GT depth directory (fixture)."""

    pred_dir: Path
    gt_dir: Path
    mask: Optional[Path] = None


@dataclass
class CompareTraj:
    """L2 position error between DexToolBench trajectory JSONs."""

    path_a: Path
    path_b: Path


@dataclass
class PrepareCaptureFromData:
    """Pick first DexToolBench task under ``data_root`` and write ``capture_out`` (rgb+depth+cam_K)."""

    data_root: Path
    capture_out: Path
    t_rc_source: Optional[Path] = None


def main() -> None:
    args = tyro.cli(
        Union[
            AlignDepth,
            FlattenPoses,
            EvalVideo,
            EvalMde,
            CompareTraj,
            PrepareCaptureFromData,
            RunVideoGen,
            RunMde,
            RunMeshSam3d,
            RunMeshFromFrame,
        ],
        prog="gstr",
    )
    if isinstance(args, AlignDepth):
        meta = align_run_directory(
            args.capture_dir,
            args.mde_raw_dir,
            args.aligned_dir,
            mask_path=args.mask,
            robust=args.robust,
        )
        print(meta)
    elif isinstance(args, FlattenPoses):
        flatten_foundationpose_export(args.src, args.dst)
        print(f"Wrote {args.dst}")
    elif isinstance(args, EvalVideo):
        frames = sorted(args.frames_dir.glob("frame_*.png"))
        print(frame_consistency_report(args.frame0, frames, mask=args.mask))
    elif isinstance(args, EvalMde):
        print(depth_rmse(args.pred_dir, args.gt_dir, mask=args.mask))
    elif isinstance(args, CompareTraj):
        print(compare_trajectories(args.path_a, args.path_b))
    elif isinstance(args, PrepareCaptureFromData):
        cap = materialize_capture(
            CaptureFromDataArgs(
                data_root=args.data_root,
                capture_out=args.capture_out,
                t_rc_source=args.t_rc_source,
            )
        )
        print(cap)
    elif isinstance(args, RunVideoGen):
        run_video_gen(args)
        print(args.video_gen_dir)
    elif isinstance(args, RunMde):
        run_mde(args)
        print(args.out_dir)
    elif isinstance(args, RunMeshSam3d):
        p = run_mesh_sam3d(args)
        print(p)
    elif isinstance(args, RunMeshFromFrame):
        p = run_mesh_from_frame(args)
        print(p)
    else:
        raise RuntimeError(f"unhandled {args!r}")
