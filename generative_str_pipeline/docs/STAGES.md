# Pipeline stage I/O contracts

All stages communicate through **run directories**. A typical layout:

```
runs/<capture_id>/<video_model_id>/<mde_model_id>/fp/
├── capture/           # stage: capture
├── video_gen/         # stage: video_gen
├── mde_raw/           # stage: mde (model-specific; optional intermediate)
├── aligned_rgbd/      # stage: mde_align → FoundationPose input
├── mesh/              # stage: mesh
├── foundationpose/    # stage: foundationpose
├── trajectories/      # after process_poses
└── meta/
    ├── prompt.txt
    ├── models_used.yaml
    └── mde_alignment.json
```

## `capture`

**Inputs:** (robot / simulator)  
**Outputs** (under `capture/` or run root):

| Artifact | Description |
|----------|-------------|
| `rgb/frame_0000.png` | First RGB frame |
| `depth/frame_0000.png` | Metric depth, same size as RGB (uint16 PNG common) |
| `cam_K.txt` | 3×3 camera intrinsics (space or newline separated) |
| `T_RC.txt` | 4×4 camera–robot transform (FoundationPose convention) |
| `masks/frame_0000.png` | *(Optional in capture; required for SAM 3D mesh)* Binary object mask, same size as RGB |

## `video_gen`

**Inputs:** `capture/rgb/frame_0000.png`, `meta/prompt.txt`, model config  
**Outputs:**

| Artifact | Description |
|----------|-------------|
| `rgb/frame_0001.png` … `frame_XXXX.png` | Generated frames (continuous indices) |
| `rgb.mp4` | Optional encoded preview |
| `meta/video_gen.json` | model id, seed, hyperparameters |

## `mde`

**Inputs:** `rgb/frame_*.png`  
**Outputs:** raw relative depth per frame (`mde_raw/` or numpy arrays); scale arbitrary until `mde_align`.

## `mde_align`

**Inputs:**

- GT: `capture/depth/frame_0000.png` (metric)
- MDE: one disparity/depth map per frame for `t ≥ 1` (same H×W as RGB)

**Outputs** (`aligned_rgbd/`):

| Artifact | Description |
|----------|-------------|
| `rgb/frame_*.png` | Copy or symlink of full RGB sequence |
| `depth/frame_0000.png` | **Identical copy of capture GT** |
| `depth/frame_0001.png` … | MDE maps after global **scale + shift** fit to GT at t=0 |
| `cam_K.txt` | Copied from capture |

Use `python -m generative_str.mde_align` (see package README).

## `mesh` (SAM 3D Objects)

Real captures rarely ship a CAD mesh. This stage **reconstructs** `mesh/object.obj` from **frame 0 RGB + an object mask** using [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (run in that project’s conda env, with `checkpoints/hf/` populated per upstream instructions).

**Inputs:**

| Artifact | Description |
|----------|-------------|
| `capture/rgb/frame_0000.png` | Same as `capture` stage |
| Object mask | Binary PNG, nonzero = foreground. Default path: `capture/masks/frame_0000.png`. You can also pass `--mask-path` on the CLI (segmentation from SAM 2, manual label, etc.). |

**Outputs:** `mesh/object.obj` (wavefront) for FoundationPose. Vertices are in SAM 3D’s reconstructed coordinates; if poses are metric from RGB-D, you may need a uniform scale or FP refinement—document any scale fix you apply in `meta/`.

**CLI (from `generative_str_pipeline`, lightweight env):**

```bash
export SAM3D_PYTHON=/path/to/sam3d-env/bin/python   # optional; default: python3
export SAM3D_ROOT=/path/to/sam-3d-objects         # optional if repo sits next to this package
gstr run-mesh-sam3d --run-dir /path/to/run_root [--mask-path /path/to/mask.png]
```

The wrapper calls `sam-3d-objects/scripts/export_obj_from_capture.py`, which runs the pipeline with `with_mesh_postprocess=True` (unlike the notebook `Inference` helper, which disables it for speed).

## `foundationpose`

**Inputs:** `aligned_rgbd/` layout + `mesh/object.obj` + `T_RC.txt`  
**Outputs:** `foundationpose/poses.json` (FoundationPose format with `poses_robot`).

## `process_poses` (SimToolReal)

**Inputs:** flattened robot-frame list or `poses.json` with `poses_robot`  
**Outputs:** DexToolBench trajectory JSON (`start_pose`, `goals` in world frame).

```bash
cd simtoolreal
python dextoolbench/process_poses.py \
  --poses-in /path/to/poses.json \
  --trajectory-out /path/to/task.json \
  --min-z 0.65 --downsample-factor 10
```

## `rgb_pose` (optional branch)

Same trajectory contract as `process_poses`; no depth after frame 0 at inference.

## `sim_eval`

Use SimToolReal `dextoolbench/visualize_task.py` and `dextoolbench/eval.py`; see `generative_str_pipeline/docs/CALIBRATION.md`.
