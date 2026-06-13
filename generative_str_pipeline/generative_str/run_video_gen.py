"""Image-to-video generation into ``video_gen/rgb/frame_0001.png`` …."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import tyro
from PIL import Image

from .pacific_time import (
    PACIFIC_TZ_NAME,
    format_path_stamp,
    isoformat_pacific,
    now_pacific,
)

VideoModel = Literal["cogvideox_i2v", "ltx_i2v", "wan_i2v"]


def _resolve_prompt(
    capture_dir: Path,
    video_gen_dir: Path,
    prompt: Optional[str],
    prompt_file: Optional[Path],
) -> str:
    if prompt is not None and prompt.strip():
        return prompt.strip()
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise FileNotFoundError(prompt_file)
        return prompt_file.read_text(encoding="utf-8").strip()
    for base in (video_gen_dir.parent, capture_dir.parent):
        cand = base / "meta" / "prompt.txt"
        if cand.is_file():
            return cand.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        "No text prompt: set --prompt, --prompt-file, or meta/prompt.txt next to the run."
    )


def _save_rgb_sequence(video_gen_dir: Path, frames: Sequence[Union[Image.Image, torch.Tensor]]) -> None:
    rgb = video_gen_dir / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)

    for i, fr in enumerate(frames, start=1):
        out = rgb / f"frame_{i:04d}.png"
        if isinstance(fr, torch.Tensor):
            t = fr.detach().cpu().float()
            if t.ndim == 4:
                t = t[0]
            if t.shape[0] in (1, 3):
                t = (t * 0.5 + 0.5).clamp(0, 1)
                t = t.permute(1, 2, 0).numpy()
            else:
                raise ValueError(f"Unexpected tensor shape {tuple(fr.shape)}")
            arr = (t * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out)
        else:
            fr.save(out)


def _maybe_downscale_image(im: Image.Image, max_side: Optional[int]) -> Image.Image:
    if max_side is None or max_side <= 0:
        return im
    w, h = im.size
    m = max(w, h)
    if m <= max_side:
        return im
    scale = max_side / float(m)
    nw = max(32, int(round(w * scale / 32)) * 32)
    nh = max(32, int(round(h * scale / 32)) * 32)
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _diffusers_memory_savings(pipe) -> None:
    """Best-effort VRAM reductions; pipelines differ by version."""
    for name in (
        "enable_vae_slicing",
        "enable_vae_tiling",
        "enable_attention_slicing",
    ):
        fn = getattr(pipe, name, None)
        if callable(fn):
            fn()


def _cogvideox_leading_padding_latent_frames(pipe, num_frames: int) -> int:
    """Latent temporal padding trimmed before VAE decode (matches Diffusers pipeline)."""
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is None or latent_frames % patch_size_t == 0:
        return 0
    return patch_size_t - latent_frames % patch_size_t


def _run_cogvideox(
    image: Image.Image,
    prompt: str,
    video_gen_dir: Path,
    hf_id: str,
    num_frames: int,
    seed: int,
    device: str,
    *,
    low_vram: bool = False,
    low_vram_max_side: int = 512,
    cpu_offload: Literal["none", "sequential", "model"] = "sequential",
) -> None:
    from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    # `sequential` / `model`: denoise with CPU offload, then VAE-decode on CPU (avoids ~4GiB GPU spike at end).
    # `none`: everything on GPU (needs enough VRAM, e.g. 80GB class; FP16).
    if device == "cuda":
        if cpu_offload == "none" or low_vram:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        dtype = torch.float32
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(hf_id, torch_dtype=dtype)
    if device == "cuda":
        _diffusers_memory_savings(pipe)
        if cpu_offload == "none":
            pipe.to("cuda")
        elif cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    im = load_image(image)
    if low_vram:
        im = _maybe_downscale_image(im, low_vram_max_side)
    gen_device = (
        "cuda" if device == "cuda" and cpu_offload == "none" else "cpu"
    )
    gen = torch.Generator(device=gen_device).manual_seed(seed)

    if device == "cuda" and cpu_offload != "none":
        out = pipe(
            image=im,
            prompt=prompt,
            num_frames=num_frames,
            generator=gen,
            output_type="latent",
        )
        latents = out.frames
        pad = _cogvideox_leading_padding_latent_frames(pipe, num_frames)
        if pad:
            latents = latents[:, pad:]
        scale = float(pipe.vae_scaling_factor_image)
        video_processor = pipe.video_processor
        lat_cpu = latents.detach().float().cpu()
        del out, latents, pipe
        gc.collect()
        torch.cuda.empty_cache()

        try:
            vae_cpu = AutoencoderKLCogVideoX.from_pretrained(
                hf_id, subfolder="vae", torch_dtype=torch.float16
            )
        except OSError:
            vae_cpu = AutoencoderKLCogVideoX.from_pretrained(
                hf_id, torch_dtype=torch.float16
            )
        vae_cpu = vae_cpu.to("cpu").eval()
        latents_perm = lat_cpu.permute(0, 2, 1, 3, 4)
        latents_perm = (1.0 / scale) * latents_perm.to(dtype=torch.float16)
        with torch.no_grad():
            video_tensor = vae_cpu.decode(latents_perm).sample.float()
        video = video_processor.postprocess_video(
            video=video_tensor, output_type="pil"
        )
        vid = video[0]
        del vae_cpu
    else:
        out = pipe(
            image=im,
            prompt=prompt,
            num_frames=num_frames,
            generator=gen,
            output_type="pil",
        )
        vid = out.frames[0]
    _save_rgb_sequence(video_gen_dir, vid)
    preview = video_gen_dir / "rgb.mp4"
    export_to_video(vid, str(preview), fps=8)


def _run_ltx(
    image: Image.Image,
    prompt: str,
    video_gen_dir: Path,
    hf_id: str,
    num_frames: int,
    seed: int,
    device: str,
    *,
    low_vram: bool = False,
    low_vram_max_side: int = 512,
    cpu_offload: Literal["none", "sequential", "model"] = "none",
) -> None:
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    if device == "cuda":
        dtype = torch.float16 if low_vram else torch.bfloat16
    else:
        dtype = torch.float32
    pipe = LTXImageToVideoPipeline.from_pretrained(hf_id, torch_dtype=dtype)
    if device == "cuda":
        _diffusers_memory_savings(pipe)
        if low_vram and cpu_offload != "none":
            if cpu_offload == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
    else:
        pipe.to(device)

    im = load_image(image)
    if low_vram:
        im = _maybe_downscale_image(im, low_vram_max_side)
    h = max(256, im.height - im.height % 32)
    w = max(256, im.width - im.width % 32)
    im = im.resize((w, h), Image.Resampling.LANCZOS)
    gen_dev = (
        "cpu"
        if (device == "cuda" and low_vram and cpu_offload != "none")
        else device
    )
    gen = torch.Generator(device=gen_dev).manual_seed(seed)
    out = pipe(
        image=im,
        prompt=prompt,
        num_frames=num_frames,
        height=h,
        width=w,
        generator=gen,
        output_type="pil",
    )
    vid = out.frames[0]
    _save_rgb_sequence(video_gen_dir, vid)
    export_to_video(vid, str(video_gen_dir / "rgb.mp4"), fps=24)


def _run_wan(
    image: Image.Image,
    prompt: str,
    video_gen_dir: Path,
    ckpt_dir: Path,
    num_frames: int,
    seed: int,
    device: str,
) -> None:
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V

    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Wan I2V needs downloaded checkpoints under --wan-ckpt-dir (missing {ckpt_dir}). "
            "See https://github.com/Wan-Video/Wan2.1 — use huggingface-cli download for the I2V bundle."
        )

    cfg = WAN_CONFIGS["i2v-14B"]
    dev_id = 0 if device == "cuda" else 0
    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=dev_id,
        init_on_cpu=True,
    )
    vid = wan_i2v.generate(
        input_prompt=prompt,
        img=image,
        frame_num=num_frames,
        seed=seed,
        offload_model=True,
    )
    # vid: (C, N, H, W) in [-1, 1]
    vid = vid.detach().cpu().float()
    c, n, h, w = vid.shape
    frames: List[Image.Image] = []
    for fi in range(n):
        fr = vid[:, fi, :, :].permute(1, 2, 0).numpy()
        fr = ((fr + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(fr))
    _save_rgb_sequence(video_gen_dir, frames)


@dataclass
class RunVideoGen:
    """Read ``capture/rgb/frame_0000.png`` and write generated frames under ``video_gen/rgb/``."""

    capture_dir: Path
    video_gen_dir: Path
    model: VideoModel = "cogvideox_i2v"

    hf_model_id: Optional[str] = None
    prompt: Optional[str] = None
    prompt_file: Optional[Path] = None

    num_frames: int = 14
    seed: int = 0
    device: str = "cuda"

    wan_ckpt_dir: Optional[Path] = None
    """Directory with Wan2.1 I2V weights (``models_t5_*.pth``, VAE, DiT, CLIP). Required for ``wan_i2v``."""

    low_vram: bool = False
    """Downscale input, float16 on CUDA, VAE/attention slicing, and CPU offload (Diffusers paths)."""

    low_vram_max_side: int = 512
    """When ``low_vram``, resize so max(width,height) is at most this (multiple of 32)."""

    cpu_offload: Literal["none", "sequential", "model"] = "sequential"
    """CogVideo: ``sequential``/``model`` offload denoise then CPU VAE decode (default, fits L4/L40S). ``none``: all GPU."""

    stamp_output_subdir: bool = False
    """If true, nest outputs under a new folder named with the run time next to ``video_gen_dir``."""

    run_id: Optional[str] = None
    """If set (e.g. from a parent benchmark run), appended to ``meta/video_gen.json`` for traceability."""


def _unique_timestamped_dir(base: Path, stamp: str) -> Path:
    d = base / stamp
    if not d.exists():
        return d
    n = 2
    while (base / f"{stamp}_{n}").exists():
        n += 1
    return base / f"{stamp}_{n}"


def run_video_gen(args: RunVideoGen) -> None:
    # Work around intermittent crashes in huggingface_hub Xet backend
    # (RuntimeError: Background writer channel closed / GIL crash).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    if args.stamp_output_subdir:
        ts = format_path_stamp()
        vg = args.video_gen_dir
        stamped = _unique_timestamped_dir(vg.parent, ts) / vg.name
        args = replace(args, video_gen_dir=stamped, stamp_output_subdir=False)

    cap_rgb = args.capture_dir / "rgb" / "frame_0000.png"
    if not cap_rgb.is_file():
        raise FileNotFoundError(cap_rgb)

    prompt = _resolve_prompt(args.capture_dir, args.video_gen_dir, args.prompt, args.prompt_file)

    defaults: dict[str, str] = {
        "cogvideox_i2v": "THUDM/CogVideoX1.5-5B-I2V",
        "ltx_i2v": "Lightricks/LTX-Video",
        "wan_i2v": "",
    }
    hf = args.hf_model_id or defaults[args.model]

    image = Image.open(cap_rgb).convert("RGB")

    if args.model == "cogvideox_i2v":
        _run_cogvideox(
            image,
            prompt,
            args.video_gen_dir,
            hf,
            args.num_frames,
            args.seed,
            args.device,
            low_vram=args.low_vram,
            low_vram_max_side=args.low_vram_max_side,
            cpu_offload=args.cpu_offload,
        )
    elif args.model == "ltx_i2v":
        _run_ltx(
            image,
            prompt,
            args.video_gen_dir,
            hf,
            args.num_frames,
            args.seed,
            args.device,
            low_vram=args.low_vram,
            low_vram_max_side=args.low_vram_max_side,
            cpu_offload=args.cpu_offload,
        )
    elif args.model == "wan_i2v":
        if args.wan_ckpt_dir is None:
            raise ValueError("wan_i2v requires --wan-ckpt-dir with downloaded Wan2.1 I2V checkpoints.")
        wan_image = (
            _maybe_downscale_image(image, args.low_vram_max_side)
            if args.low_vram
            else image
        )
        _run_wan(
            wan_image,
            prompt,
            args.video_gen_dir,
            args.wan_ckpt_dir,
            args.num_frames,
            args.seed,
            args.device,
        )
    else:
        raise ValueError(args.model)

    meta = args.video_gen_dir.parent / "meta" / "video_gen.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "model": args.model,
        "hf_model_id": hf or None,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "prompt": prompt,
        "low_vram": args.low_vram,
        "low_vram_max_side": args.low_vram_max_side,
        "cpu_offload": args.cpu_offload,
        "timezone": PACIFIC_TZ_NAME,
        "generated_at": isoformat_pacific(),
    }
    if args.run_id:
        meta_payload["run_id"] = args.run_id
    meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")


@dataclass
class BenchmarkVideoGen:
    """Run several I2V backends on the same ``capture/rgb/frame_0000.png`` and shared prompt.

    By default writes under ``output-dir/<YYYY-mm-dd_HH-MM-SS>/`` so repeated runs do not overwrite.
    Under that: ``<model>/video_gen/rgb/``, ``<model>/meta/video_gen.json``, and top-level ``meta/``
    Summary JSON is ``meta/benchmark_video_gen.json`` with Pacific ``started_at`` (see ``timezone``).

    Use the separate entry point ``gstr-benchmark-video-gen`` (not ``gstr``) so the default CLI stays
    single-model ``gstr run-video-gen``.
    """

    capture_dir: Path
    output_dir: Path
    """Root folder; each model is written under ``<output-dir>/<timestamp>/<model>/`` when stamping is on."""

    models: list[VideoModel] = field(
        default_factory=lambda: ["cogvideox_i2v", "ltx_i2v"]
    )
    """Backends to run. Repeat ``--models``: ``--models cogvideox_i2v --models ltx_i2v``."""

    prompt: Optional[str] = None
    prompt_file: Optional[Path] = None

    num_frames: int = 14
    seed: int = 0
    device: str = "cuda"

    wan_ckpt_dir: Optional[Path] = None

    cogvideox_hf_model_id: Optional[str] = None
    """If set, load this HF repo for ``cogvideox_i2v`` (e.g. ``THUDM/CogVideoX-5b-I2V``) instead of the default."""

    low_vram: bool = False
    low_vram_max_side: int = 512
    cpu_offload: Literal["none", "sequential", "model"] = "sequential"

    stop_on_error: bool = False
    """If true, abort when a backend raises; otherwise record the error and continue."""

    stamp_output_dir: bool = True
    """If true, create ``output-dir/<YYYY-mm-dd_HH-MM-SS>/`` (US Pacific wall time) and write there."""


def run_benchmark_video_gen(args: BenchmarkVideoGen) -> Path:
    cap_rgb = args.capture_dir / "rgb" / "frame_0000.png"
    if not cap_rgb.is_file():
        raise FileNotFoundError(cap_rgb)

    started = now_pacific()
    started_iso = isoformat_pacific(started)
    stamp_str = format_path_stamp(started)

    out = args.output_dir
    run_id: Optional[str] = None
    if args.stamp_output_dir:
        out = _unique_timestamped_dir(args.output_dir, stamp_str)
        run_id = out.name
        print(f"[benchmark] output -> {out}")

    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shared_prompt_path = meta_dir / "prompt.txt"

    probe_vg = out / "_probe" / "video_gen"
    prompt_text = _resolve_prompt(
        args.capture_dir, probe_vg, args.prompt, args.prompt_file
    )
    shared_prompt_path.write_text(prompt_text, encoding="utf-8")

    runs: list[dict] = []
    for m in args.models:
        if m == "wan_i2v" and args.wan_ckpt_dir is None:
            err = "wan_i2v requires --wan-ckpt-dir"
            runs.append({"model": m, "ok": False, "error": err})
            print(f"[benchmark] skip {m}: {err}")
            if args.stop_on_error:
                raise ValueError(err)
            continue

        vg = out / m / "video_gen"
        hf_model_id: Optional[str] = None
        if m == "cogvideox_i2v" and args.cogvideox_hf_model_id is not None:
            hf_model_id = args.cogvideox_hf_model_id
        sub = RunVideoGen(
            model=m,
            capture_dir=args.capture_dir,
            video_gen_dir=vg,
            prompt_file=shared_prompt_path,
            hf_model_id=hf_model_id,
            num_frames=args.num_frames,
            seed=args.seed,
            device=args.device,
            wan_ckpt_dir=args.wan_ckpt_dir,
            low_vram=args.low_vram,
            low_vram_max_side=args.low_vram_max_side,
            cpu_offload=args.cpu_offload,
            run_id=run_id,
        )
        try:
            run_video_gen(sub)
            runs.append(
                {
                    "model": m,
                    "ok": True,
                    "video_gen_dir": str(vg),
                    "meta_dir": str(vg.parent / "meta"),
                }
            )
            print(f"[benchmark] ok {m} -> {vg}")
        except Exception as e:  # noqa: BLE001 — surface model errors per backend
            runs.append({"model": m, "ok": False, "error": str(e)})
            print(f"[benchmark] fail {m}: {e}")
            if args.stop_on_error:
                raise

    summary_path = meta_dir / "benchmark_video_gen.json"
    summary_path.write_text(
        json.dumps(
            {
                "prompt": prompt_text,
                "capture_dir": str(args.capture_dir.resolve()),
                "output_dir": str(out.resolve()),
                "stamp_output_dir": args.stamp_output_dir,
                "run_id": run_id,
                "timezone": PACIFIC_TZ_NAME,
                "started_at": started_iso,
                "num_frames": args.num_frames,
                "seed": args.seed,
                "low_vram": args.low_vram,
                "low_vram_max_side": args.low_vram_max_side,
                "cpu_offload": args.cpu_offload,
                "cogvideox_hf_model_id": args.cogvideox_hf_model_id,
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    run_video_gen(tyro.cli(RunVideoGen))


if __name__ == "__main__":
    main()

