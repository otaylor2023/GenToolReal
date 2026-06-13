from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pose_track.bootstrap import bootstrap_run
from pose_track.fp_run import run_foundation_pose
from pose_track.layout import default_run_dir, project_dir, repo_root
from pose_track.smooth_poses import smooth_poses_json
from pose_track.vendor.mde_align import align_run_directory
from pose_track.video_frames import extract_video_to_rgb_frames


def _run_da2(run_dir: Path) -> None:
    cmd = [sys.executable, "-m", "pose_track.da2_worker", str(run_dir)]
    subprocess.run(cmd, check=True)


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    rd = Path(args.run_dir).resolve() if args.run_dir else default_run_dir()
    gt = Path(args.gt_session_dir).resolve() if args.gt_session_dir else None
    bootstrap_run(rd, gt_session_dir=gt)
    print(str(rd))
    return 0


def _cmd_extract_video(args: argparse.Namespace) -> int:
    from PIL import Image

    rd = Path(args.run_dir).resolve()
    vid = rd / "meta" / "source_video.mp4"
    if not vid.is_file():
        alt = project_dir() / "assets" / "videos" / "r4p8" / "clip.mp4"
        if alt.is_file():
            vid = alt
    if not vid.is_file():
        raise FileNotFoundError(f"No video at {rd / 'meta' / 'source_video.mp4'} or assets clip")
    rgb0 = rd / "capture" / "rgb" / "frame_0000.png"
    target_hw = None
    if rgb0.is_file():
        im = Image.open(rgb0)
        target_hw = (im.height, im.width)
    n = extract_video_to_rgb_frames(
        vid, rd / "video_gen" / "rgb", start_index=1, target_hw=target_hw
    )
    print(f"extracted_frames={n}")
    return 0


def _cmd_run_da2(args: argparse.Namespace) -> int:
    _run_da2(Path(args.run_dir).resolve())
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    rd = Path(args.run_dir).resolve()
    meta = align_run_directory(
        rd / "capture",
        rd / "mde_raw",
        rd / "aligned_rgbd",
        mask_path=rd / "capture" / "masks" / "frame_0000.png",
        robust=args.robust,
    )
    print(f"scale={meta.scale} shift={meta.shift} rmse_after={meta.rmse_after}")
    return 0


def _cmd_run_fp(args: argparse.Namespace) -> int:
    run_foundation_pose(Path(args.run_dir).resolve(), fp_python=args.fp_python)
    return 0


def _cmd_smooth(args: argparse.Namespace) -> int:
    rd = Path(args.run_dir).resolve()
    src = rd / "foundationpose" / "poses.json"
    dst = rd / "foundationpose" / "poses_smoothed.json"
    smooth_poses_json(src, dst, window=args.window, polyorder=args.polyorder)
    print(str(dst))
    return 0


def _cmd_process_poses(args: argparse.Namespace) -> int:
    rd = Path(args.run_dir).resolve()
    root = repo_root()
    proc = root / "simtoolreal" / "dextoolbench" / "process_poses.py"
    if not proc.is_file():
        raise FileNotFoundError(proc)
    poses = rd / "foundationpose" / "poses_smoothed.json"
    if not poses.is_file():
        poses = rd / "foundationpose" / "poses.json"
    out = rd / "trajectories" / "task_from_video.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(proc),
        "--poses-in",
        str(poses),
        "--trajectory-out",
        str(out),
        "--min-z",
        str(args.min_z),
        "--downsample-factor",
        str(args.downsample_factor),
    ]
    subprocess.run(cmd, check=True)
    print(str(out))
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    from PIL import Image

    rd = Path(args.run_dir).resolve()
    vid = rd / "meta" / "source_video.mp4"
    if not vid.is_file():
        vid = project_dir() / "assets" / "videos" / "r4p8" / "clip.mp4"
    if not vid.is_file():
        print("[run-all] no video found; run bootstrap + place clip.mp4 first", file=sys.stderr)
        return 1
    rgb0 = rd / "capture" / "rgb" / "frame_0000.png"
    target_hw = None
    if rgb0.is_file():
        im = Image.open(rgb0)
        target_hw = (im.height, im.width)

    def extract() -> None:
        extract_video_to_rgb_frames(
            vid, rd / "video_gen" / "rgb", start_index=1, target_hw=target_hw
        )

    seq = [
        ("extract-video", extract),
        ("run-da2", lambda: _run_da2(rd)),
        (
            "align",
            lambda: align_run_directory(
                rd / "capture",
                rd / "mde_raw",
                rd / "aligned_rgbd",
                mask_path=rd / "capture" / "masks" / "frame_0000.png",
                robust=False,
            ),
        ),
        ("run-fp", lambda: run_foundation_pose(rd, fp_python=args.fp_python)),
        (
            "smooth",
            lambda: smooth_poses_json(
                rd / "foundationpose" / "poses.json",
                rd / "foundationpose" / "poses_smoothed.json",
                window=args.window,
                polyorder=2,
            ),
        ),
    ]
    for name, fn in seq:
        try:
            fn()
        except Exception as exc:
            print(f"[run-all] stopped at {name}: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pose-track")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bootstrap", help="Create run dirs, copy mesh/video, optional GT depth")
    pb.add_argument("--run-dir", default=None)
    pb.add_argument("--gt-session-dir", default=None, help="Sim frame dir with 00_depth_uint16mm.png")
    pb.set_defaults(func=_cmd_bootstrap)

    pe = sub.add_parser("extract-video", help="Decode pilot MP4 to video_gen/rgb/frame_0001+")
    pe.add_argument("--run-dir", required=True)
    pe.set_defaults(func=_cmd_extract_video)

    pd = sub.add_parser("run-da2", help="Depth Anything v2 → mde_raw (or POSE_TRACK_MOCK_MDE=1)")
    pd.add_argument("--run-dir", required=True)
    pd.set_defaults(func=_cmd_run_da2)

    pa = sub.add_parser("align", help="Scale/shift MDE to GT @0 → aligned_rgbd")
    pa.add_argument("--run-dir", required=True)
    pa.add_argument("--robust", action="store_true")
    pa.set_defaults(func=_cmd_align)

    pf = sub.add_parser("run-fp", help="Subprocess FoundationPose extract_poses.py")
    pf.add_argument("--run-dir", required=True)
    pf.add_argument("--fp-python", default=None, help="Python in foundationpose conda env")
    pf.set_defaults(func=_cmd_run_fp)

    ps = sub.add_parser("smooth", help="Savitzky–Golay smooth poses_cam")
    ps.add_argument("--run-dir", required=True)
    ps.add_argument("--window", type=int, default=11)
    ps.add_argument("--polyorder", type=int, default=2)
    ps.set_defaults(func=_cmd_smooth)

    pp = sub.add_parser("process-poses", help="Subprocess simtoolreal dextoolbench/process_poses.py")
    pp.add_argument("--run-dir", required=True)
    pp.add_argument("--min-z", type=float, default=0.65)
    pp.add_argument("--downsample-factor", type=int, default=10)
    pp.set_defaults(func=_cmd_process_poses)

    pa2 = sub.add_parser("run-all", help="extract-video → da2 → align → fp → smooth")
    pa2.add_argument("--run-dir", required=True)
    pa2.add_argument("--fp-python", default=None)
    pa2.add_argument("--window", type=int, default=11)
    pa2.set_defaults(func=_cmd_run_all)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
