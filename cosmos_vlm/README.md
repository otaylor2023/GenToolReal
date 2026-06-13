# Cosmos VLM (Separate Module)

This module is intentionally isolated from existing `vlm_sidecar` and `simtoolreal` runtime logic.

It provides these manual CLI stages:

1. `import-image` - copy one starting workspace image into a Cosmos run.
2. `reason-plan` - run local `Cosmos-Reason2-8B` to generate a detailed trajectory plan artifact.
3. `predict` - create a new `predict_00XX/` folder and draft the video prompt from `reason/trajectory_steps`.
4. `submit-video` - submit an image-to-video task to Wavespeed Cosmos Predict 2.5.
5. `get-video-result` - poll a Wavespeed task result and save outputs.

## Environment

Place secrets/config in `.env` at workspace root. Values are loaded automatically.

Required:

- `WAVESPEED_API_KEY=<your_key>`

Optional:

- `COSMOS_REASON2_MODEL_ID=nvidia/Cosmos-Reason2-8B`
- `COSMOS_DEVICE=cuda`
- `COSMOS_DTYPE=bfloat16`
- `COSMOS_MAX_NEW_TOKENS=8192`
- `WAVESPEED_BASE_URL=https://api.wavespeed.ai/api/v3`
- `COSMOS_RUNS_DIR=cosmos_vlm/runs`

## Stage Commands

### Stage 1: Import

Default behavior (no flags): copy one `00_main.png` from latest `simtoolreal/runs/vlm_*` into the next run folder `run_00XX`.

```bash
python -m cosmos_vlm.cli import-image
```

Explicit source options:

```bash
python -m cosmos_vlm.cli import-image --source-file /abs/path/image.png --run-id my_run
python -m cosmos_vlm.cli import-image --sim-session-dir /abs/path/simtoolreal/runs/vlm_20260416_023104 --step 0 --run-id my_run
```

### Stage 2: Reason

```bash
python -m cosmos_vlm.cli reason-plan \
  --model-id /home/ubuntu/Generative_STR/models/Cosmos-Reason2-8B
```

If no run is provided, this creates the next `run_00XX` automatically and auto-imports a start image from latest `simtoolreal/runs`.

Artifacts:

- `reason/raw.txt`
- `reason/plan.json`
- `reason/meta.json`

### Stage 3: Predict Prompt Draft (No API Call)

```bash
python -m cosmos_vlm.cli predict --run-id run_0001
```

This creates the next `predict_00XX/` folder under that run and writes `prompt.txt`.

### Stage 4: Submit Video Task

`--image` must be a publicly accessible `http(s)` URL.

```bash
python -m cosmos_vlm.cli submit-video \
  --run-dir /home/ubuntu/Generative_STR/cosmos_vlm/runs/my_run \
  --image https://example.com/workspace_start.png \
  --prompt "The robot sweeps the red balls into the green bin with smooth, deliberate motion."
```

Artifacts (per attempt under `predict_00XX/`):

- `predict_00XX/prompt.txt` (if drafted)
- `predict_00XX/request.json`
- `predict_00XX/response.json`

### Stage 5: Poll Result

If `--task-id` is omitted, the command reads the task id from the selected `predict_00XX/response.json`.

```bash
python -m cosmos_vlm.cli get-video-result \
  --run-dir /home/ubuntu/Generative_STR/cosmos_vlm/runs/my_run
```

Artifacts (same `predict_00XX/`):

- `predict_00XX/result.json`
- `predict_00XX/summary.json`

