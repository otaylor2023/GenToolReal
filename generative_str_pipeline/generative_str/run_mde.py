"""Run monocular depth estimation on ``rgb/frame_*.png`` → ``mde_raw/frame_*.png``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image

from generative_str.cam_k_io import load_cam_k_txt
from generative_str.depth_io import save_mde_depth_png, sorted_frame_stems

MdeModel = Literal["da3", "marigold", "unidepth_v2", "moge2"]


def _pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class RunMde:
    """MDE on every ``frame_*.png`` under ``rgb_dir``; writes parallel ``mde_raw`` PNGs."""

    model: MdeModel
    rgb_dir: Path
    out_dir: Path
    """Directory for ``frame_XXXX.png`` depth maps (``mde_raw``)."""

    hf_model_id: Optional[str] = None
    """Override default Hugging Face repo / checkpoint id for this backend."""

    cam_k: Optional[Path] = None
    """Optional ``cam_K.txt`` (3×3). Used by UniDepth V2 and optionally DA3."""

    device: str = "cuda"


def _run_da3(
    paths: list[Path],
    out_dir: Path,
    hf_id: str,
    device: str,
    cam_k: Optional[np.ndarray],
) -> None:
    from depth_anything_3.api import DepthAnything3

    m = DepthAnything3.from_pretrained(hf_id)
    m = m.to(device)
    intrinsics_np = None
    if cam_k is not None:
        intrinsics_np = cam_k.astype(np.float32)[np.newaxis, ...]

    for p in paths:
        pred = m.inference(
            [str(p)],
            intrinsics=intrinsics_np,
            export_dir=None,
            infer_gs=False,
        )
        dep = np.asarray(pred.depth[0], dtype=np.float32)
        save_mde_depth_png(out_dir / f"{p.stem}.png", dep)


def _run_marigold(paths: list[Path], out_dir: Path, hf_id: str, device: str) -> None:
    from diffusers import MarigoldDepthPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    load_kw = {"torch_dtype": dtype}
    if dtype == torch.float16:
        load_kw["variant"] = "fp16"
    pipe = MarigoldDepthPipeline.from_pretrained(hf_id, **load_kw)
    pipe.to(device)
    dev_type = "cuda" if device == "cuda" else "cpu"
    for p in paths:
        im = _pil_rgb(p)
        with torch.autocast(device_type=dev_type, dtype=dtype, enabled=(device == "cuda")):
            out = pipe(im, match_input_resolution=True)
        dep = np.asarray(out.prediction, dtype=np.float32).squeeze()
        save_mde_depth_png(out_dir / f"{p.stem}.png", dep)


def _run_unidepth(
    paths: list[Path],
    out_dir: Path,
    hf_id: str,
    device: str,
    cam_k: np.ndarray,
) -> None:
    from unidepth.models import UniDepthV2

    m = UniDepthV2.from_pretrained(hf_id)
    m = m.to(device)
    m.eval()
    K = torch.from_numpy(cam_k.astype(np.float32)).unsqueeze(0).to(device)

    for p in paths:
        rgb = np.array(_pil_rgb(p), dtype=np.uint8)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            o = m.infer(t, camera=K, normalize=True)
        dep = o["depth"].squeeze().float().cpu().numpy()
        save_mde_depth_png(out_dir / f"{p.stem}.png", dep)


def _run_moge2(paths: list[Path], out_dir: Path, hf_id: str, device: str) -> None:
    import cv2

    from moge.model import import_model_class_by_version

    m = import_model_class_by_version("v2").from_pretrained(hf_id).to(device).eval()
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            o = m.infer(
                t,
                resolution_level=9,
                use_fp16=device.startswith("cuda"),
            )
        dep = o["depth"].squeeze().float().cpu().numpy()
        if dep.shape != (h, w):
            dep = cv2.resize(dep, (w, h), interpolation=cv2.INTER_LINEAR)
        save_mde_depth_png(out_dir / f"{p.stem}.png", dep)


def run_mde(args: RunMde) -> None:
    if not args.rgb_dir.is_dir():
        raise FileNotFoundError(args.rgb_dir)

    stems = sorted_frame_stems(args.rgb_dir)
    if not stems:
        raise FileNotFoundError(f"No frame_*.png under {args.rgb_dir}")
    paths = [args.rgb_dir / f"{s}.png" for s in stems]

    cam = load_cam_k_txt(args.cam_k) if args.cam_k is not None else None

    defaults: dict[str, str] = {
        "da3": "depth-anything/DA3-LARGE-1.1",
        "marigold": "prs-eth/marigold-depth-lcm-v1-0",
        "unidepth_v2": "lpiccinelli/unidepth-v2-vitl14",
        "moge2": "Ruicheng/moge-2-vitl-normal",
    }
    hf = args.hf_model_id or defaults[args.model]

    if args.model == "unidepth_v2" and cam is None:
        raise ValueError("UniDepth V2 requires --cam-k (3×3 intrinsics).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "da3":
        _run_da3(paths, args.out_dir, hf, args.device, cam)
    elif args.model == "marigold":
        _run_marigold(paths, args.out_dir, hf, args.device)
    elif args.model == "unidepth_v2":
        assert cam is not None
        _run_unidepth(paths, args.out_dir, hf, args.device, cam)
    else:
        _run_moge2(paths, args.out_dir, hf, args.device)

    meta = args.out_dir.parent / "meta" / "mde_raw.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps(
            {"model": args.model, "hf_model_id": hf, "frames": stems},
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    run_mde(tyro.cli(RunMde))


if __name__ == "__main__":
    main()
