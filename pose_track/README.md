# pose_track

Self-contained pipeline: **sim GT depth @ frame 0** (from SimToolReal when `SAVE_VLM_GT_DEPTH=1`) → **generated video frames** → **Depth Anything V2** (optional `pip install -e ".[da2]"`) → **scale/shift align** (vendored `mde_align`) → **FoundationPose** (subprocess to repo `FoundationPose/`) → **smoothing**.

## Operator gate (only human step)

After the SimToolReal depth hook is merged, run sweep Viser **once** with:

```bash
export SAVE_VLM_GT_DEPTH=1
# then your usual run_llm_viser / sim session
```

Artifacts appear next to `00_main.png`: `00_depth_uint16mm.png`, `cam_K.txt`, `depth_meta.json`.

## Autonomous pilot bootstrap

From repo root:

```bash
pip install -e ./pose_track
pose-track bootstrap --run-dir pose_track/runs/pilot_r4p8
```

This copies **mesh** from [`simtoolreal/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj`](../simtoolreal/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj), seeds `capture/` layout, copies any `videos/**/r4p8*.mp4` or `cosmos_vlm/**/output_0000.mp4` if found, and writes a **mask from GT depth** when `POSE_TRACK_GT_SESSION_DIR` points at a frame folder containing `00_depth_uint16mm.png`.

**Inventoried default sources (repo-relative):**

| Artifact | Source |
|----------|--------|
| Tool mesh | `simtoolreal/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj` |
| Pilot MP4 | First match of `videos/**/*r4p8*.mp4` (case-insensitive) else first `cosmos_vlm/runs/**/output_*.mp4` |
| GT depth + `cam_K` | Operator: Sim session frame dir (`POSE_TRACK_GT_SESSION_DIR=.../frame_00000`) after `SAVE_VLM_GT_DEPTH=1` |

## Full chain

```bash
pose-track extract-video --run-dir pose_track/runs/pilot_r4p8
pose-track run-da2 --run-dir pose_track/runs/pilot_r4p8
pose-track align --run-dir pose_track/runs/pilot_r4p8
pose-track run-fp --run-dir pose_track/runs/pilot_r4p8 --fp-python /path/to/foundationpose/env/bin/python
pose-track smooth --run-dir pose_track/runs/pilot_r4p8
```

Or `pose-track run-all ...` (stops early if a stage is missing inputs).

## v2 (SAM2 + re-init)

Deferred; see [V2_SAM2.md](V2_SAM2.md).
