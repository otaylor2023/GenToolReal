# Generative STR pipeline (SimToolReal / DexToolBench)

Modular **on-disk** stages from a robot **RGB-D frame 0** → generated video → aligned depth → FoundationPose → DexToolBench trajectories → sim eval.

## Install

```bash
cd /home/ubuntu/Generative_STR/generative_str_pipeline
pip install -e ".[dev]"
```

CLI entry point: `gstr` (or `python -m generative_str`). For **single-model** video generation use `gstr run-video-gen` (default model: `cogvideox_i2v`). Multi-backend comparison is a separate command: `gstr-benchmark-video-gen` (after `pip install -e .`).

## Commands

```bash
# Scale raw MDE to metric GT at t=0 (requires mde_raw/frame_0000.png)
gstr align-depth --capture-dir runs/foo/capture --mde-raw-dir runs/foo/mde_raw --aligned-dir runs/foo/aligned_rgbd

# FoundationPose dict → flat list for DexToolBench
gstr flatten-poses --src fp/poses.json --dst poses_flat.json

# Video consistency vs frame 0 (optional static mask PNG)
gstr eval-video --frame0 capture/rgb/frame_0000.png --frames-dir video_gen/rgb

# MDE vs dense GT (fixture)
gstr eval-mde --pred-dir aligned/depth --gt-dir sim_gt/depth

# Trajectory JSON comparison (world-frame goals)
gstr compare-traj --path-a traj_a.json --path-b traj_b.json

# Mesh from RGB + mask via SAM 3D Objects (use that repo’s Python; see docs/STAGES.md)
export SAM3D_PYTHON=/path/to/sam3d/bin/python
gstr run-mesh-sam3d --run-dir runs/my_run
```

## SimToolReal integration

Single-file pose processing (robot frame → world trajectory):

```bash
cd /home/ubuntu/Generative_STR/simtoolreal
python dextoolbench/process_poses.py \
  --poses-in /path/to/poses.json \
  --trajectory-out dextoolbench/trajectories/brush/blue_brush/sweep_forward.json \
  --min-z 0.65 --downsample-factor 10
```

## Layout

- [`docs/STAGES.md`](docs/STAGES.md) — artifact contracts per stage  
- [`docs/CALIBRATION.md`](docs/CALIBRATION.md) — `cam_K`, `T_RC`, MDE alignment  
- [`config/models.yaml.example`](config/models.yaml.example) — model registry template  
- [`config/sweep_matrix.example.yaml`](config/sweep_matrix.example.yaml) — factorial grid  
- [`prompts/examples/sweep_brush_prompt.txt`](prompts/examples/sweep_brush_prompt.txt) — example I2V prompt  
- [`assets/pilot/frame_0000.png`](assets/pilot/frame_0000.png) — pilot RGB (if present)  

## Tests

```bash
pytest -q tests/
```

## Scripts

- `scripts/init_run_directory.sh <run_root>` — create folder skeleton  
- `scripts/run_dextoolbench_smoke.sh` — optional `visualize_task` smoke (from `simtoolreal/` root)  
