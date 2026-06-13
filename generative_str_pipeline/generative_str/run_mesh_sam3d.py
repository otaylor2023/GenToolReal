"""Delegate mesh extraction to SAM 3D Objects (separate Python env)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def _infer_generative_str_repo() -> Path:
    # generative_str/run_mesh_sam3d.py -> generative_str_pipeline -> Generative_STR
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class RunMeshSam3d:
    """Write ``mesh/object.obj`` from ``capture/`` using SAM 3D Objects."""

    run_dir: Path
    """Run root containing ``capture/rgb/frame_0000.png``."""

    sam3d_root: Optional[Path] = None
    """SAM 3D Objects repo root. Default: ``SAM3D_ROOT`` env or ``<repo>/sam-3d-objects``."""

    sam3d_python: Optional[str] = None
    """Interpreter with SAM 3D deps. Default: ``SAM3D_PYTHON`` env or ``python3``."""

    mask_path: Optional[Path] = None
    """Object mask PNG; default ``capture/masks/frame_0000.png`` if present."""

    config_relpath: str = "checkpoints/hf/pipeline.yaml"
    """Config path relative to ``sam3d_root`` (or absolute)."""

    seed: int = 42


@dataclass
class RunMeshFromFrame:
    """Generate ``object.obj`` from a single frame path."""

    frame_path: Path
    out_dir: Path
    """Output directory where ``object.obj`` is written."""

    mask_path: Optional[Path] = None
    """Optional binary mask image. If omitted, generate with SAM2 auto-mask."""

    sam2_model_id: str = "facebook/sam2-hiera-large"
    """SAM2 Hugging Face model id for automatic mask generation."""

    sam3d_root: Optional[Path] = None
    sam3d_python: Optional[str] = None
    config_relpath: str = "checkpoints/hf/pipeline.yaml"
    seed: int = 42


def _pick_largest_mask(mask_records: list[dict], shape_hw: tuple[int, int]) -> np.ndarray:
    if not mask_records:
        raise RuntimeError("SAM2 returned no masks")
    best = max(mask_records, key=lambda m: int(m.get("area", 0)))
    seg = best.get("segmentation")
    if seg is None:
        raise RuntimeError("SAM2 mask record missing 'segmentation'")
    mask = np.asarray(seg, dtype=bool)
    if mask.shape != shape_hw:
        raise RuntimeError(f"SAM2 mask shape mismatch: got {mask.shape}, expected {shape_hw}")
    return mask


def _generate_mask_with_sam2(frame_path: Path, out_mask_path: Path, model_id: str) -> Path:
    # Avoid unstable xet backend during first checkpoint download.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image = Image.open(frame_path).convert("RGB")
    arr = np.asarray(image)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(model_id)
    masks = generator.generate(arr)
    best = _pick_largest_mask(masks, (arr.shape[0], arr.shape[1]))
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((best.astype(np.uint8) * 255), mode="L").save(out_mask_path)
    return out_mask_path


def run_mesh_sam3d(args: RunMeshSam3d) -> Path:
    run_dir = args.run_dir.resolve()
    capture = run_dir / "capture"
    if not (capture / "rgb" / "frame_0000.png").is_file():
        raise FileNotFoundError(f"Expected {capture / 'rgb' / 'frame_0000.png'}")

    sam3d_root = args.sam3d_root
    if sam3d_root is None:
        env_root = os.environ.get("SAM3D_ROOT")
        sam3d_root = Path(env_root) if env_root else _infer_generative_str_repo() / "sam-3d-objects"
    sam3d_root = sam3d_root.resolve()

    export_script = sam3d_root / "scripts" / "export_obj_from_capture.py"
    if not export_script.is_file():
        raise FileNotFoundError(
            f"Missing {export_script}. Set sam3d_root= or SAM3D_ROOT to your SAM 3D clone."
        )

    py = args.sam3d_python or os.environ.get("SAM3D_PYTHON", "python3")

    mask = args.mask_path
    if mask is None:
        default_m = capture / "masks" / "frame_0000.png"
        if default_m.is_file():
            mask = default_m
    if mask is None or not mask.is_file():
        raise FileNotFoundError(
            "Object mask required: add capture/masks/frame_0000.png or pass mask_path="
            f"(tried {capture / 'masks' / 'frame_0000.png'})"
        )

    mesh_dir = run_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    out_obj = mesh_dir / "object.obj"

    cfg = Path(args.config_relpath)
    if not cfg.is_absolute():
        cfg_arg = args.config_relpath
    else:
        cfg_arg = str(cfg)

    cmd = [
        py,
        str(export_script),
        "--capture-dir",
        str(capture),
        "--mask-path",
        str(mask.resolve()),
        "--output-obj",
        str(out_obj.resolve()),
        "--config",
        cfg_arg,
        "--seed",
        str(args.seed),
    ]
    child_env = os.environ.copy()
    # Reduce allocator fragmentation in long SAM3D runs.
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(sam3d_root),
            env=child_env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        combined = f"{stdout}\n{stderr}".lower()
        if "outofmemoryerror" in combined or "cuda out of memory" in combined:
            raise RuntimeError(
                "SAM3D failed with CUDA OOM during mesh decode. "
                "This commonly happens on <=16GB GPUs (e.g., T4), while the "
                "upstream SAM3D Objects setup recommends >=32GB VRAM. "
                "Try a larger GPU or a lighter mesh extraction path."
            ) from exc
        raise RuntimeError(
            f"SAM3D export failed (exit={exc.returncode}). "
            f"Command: {' '.join(cmd)}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        ) from exc
    return out_obj


def run_mesh_from_frame(args: RunMeshFromFrame) -> Path:
    frame_path = args.frame_path.resolve()
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gstr_mesh_from_frame_") as tmp:
        tmp_root = Path(tmp)
        capture = tmp_root / "capture"
        rgb_dir = capture / "rgb"
        masks_dir = capture / "masks"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        rgb0 = rgb_dir / "frame_0000.png"
        Image.open(frame_path).convert("RGB").save(rgb0)

        if args.mask_path is not None:
            m = args.mask_path.resolve()
            if not m.is_file():
                raise FileNotFoundError(m)
            mask0 = masks_dir / "frame_0000.png"
            Image.open(m).convert("L").save(mask0)
        else:
            mask0 = _generate_mask_with_sam2(
                frame_path=rgb0,
                out_mask_path=masks_dir / "frame_0000.png",
                model_id=args.sam2_model_id,
            )

        # Reuse existing SAM3D wrapper by creating a minimal run root.
        mesh_run_dir = tmp_root / "run"
        (mesh_run_dir / "capture").mkdir(parents=True, exist_ok=True)
        shutil.copytree(capture, mesh_run_dir / "capture", dirs_exist_ok=True)
        out_obj = run_mesh_sam3d(
            RunMeshSam3d(
                run_dir=mesh_run_dir,
                sam3d_root=args.sam3d_root,
                sam3d_python=args.sam3d_python,
                mask_path=mask0,
                config_relpath=args.config_relpath,
                seed=args.seed,
            )
        )
        final_obj = out_dir / "object.obj"
        shutil.copy2(out_obj, final_obj)
        return final_obj
