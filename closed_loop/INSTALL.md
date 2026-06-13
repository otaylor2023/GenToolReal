# Installing `closed_loop`

Standalone inference package for the reactive VLA (receding-horizon closed loop).
It has **no dependency** on `training/`, `generative_str_pipeline/`,
`policy_exec/`, or IsaacGym — so it installs cleanly on the robot.

## 1. What to copy onto the robot

From this repo, copy these two directories to the SimToolReal machine:

- `closed_loop/` — this package (includes packaged checkpoints under
  `closed_loop/assets/` and `model_registry.json`).
- `simtoolreal/` — the ROS deployment. Only the
  `deployment/goal_pose_node_closed_loop.py` node is needed for closed-loop runs.

## 2. Python version

- **Robot (ROS Noetic):** Python 3.8. `viz_interactive` ships a compatibility
  shim, and the package floor is `>=3.8`.
- **Workstation / dev:** Python 3.9+ is fine.

Use whatever interpreter your ROS nodes already run with, so `rospy` resolves.

## 3. Install

```bash
cd closed_loop
pip install -e .
```

### Extras

| Use case | Command | Adds |
| --- | --- | --- |
| Core inference only | `pip install -e .` | `numpy`, `torch`, `transformers` |
| Robot ROS node | `pip install -e ".[robot]"` | `tyro`, `termcolor` |
| Interactive viz / annotation | `pip install -e ".[viz]"` | `viser`, `trimesh`, `tyro` |
| Dev / tests | `pip install -e ".[dev]"` | `pytest` |

> `rospy` and `isaacgymenvs` are **not** pip-installed here — they come from the
> robot's existing ROS workspace. The `[robot]` extra only adds the small CLI /
> logging deps the node needs on top of that.

Plain `requirements.txt` is also provided for environments that prefer it:

```bash
pip install -r requirements.txt
```

## 4. CLIP text-encoder weights

Inference encodes the instruction/labels with the CLIP text tower
(`openai/clip-vit-base-patch32`). Provide the weights in one of these ways
(checked in this order):

1. **Packaged cache (offline / air-gapped robot):** place the HF snapshot under
   `closed_loop/closed_loop/assets/clip_cache/hub/models--openai--clip-vit-base-patch32/...`.
   When present, inference forces `local_files_only` and needs no network.
2. **HF cache env:** set `HF_HOME` (and/or `TRANSFORMERS_CACHE`) to a directory
   containing the snapshot.
3. **Download on first run:** if neither cache is present and the machine has
   network access, `transformers` downloads it automatically.

## 5. Verify the install (CPU-only, no robot needed)

```bash
# Registry resolves and lists every packaged model
python -c "from closed_loop import list_models; print(list_models())"

# Load + single prediction for a chosen model, CPU only
CUDA_VISIBLE_DEVICES="" python -m closed_loop.examples.verify_registry_load
```

You should see keys like `all_tasks_joint_pretrain_epoch10`,
`all_tasks_grpo_iter10`, `all_tasks_grpo_iter20`, and the per-task pretrains.

## 6. Run on the robot

```bash
# from the SimToolReal repo root, after `pip install -e ../closed_loop` (+[robot])
python simtoolreal/deployment/goal_pose_node_closed_loop.py \
  --model all_tasks_grpo_iter20 \
  --control-frame blue_brush \
  --fixed-destination -0.365 -0.056 0.517 \
  --device cuda
```

- `--model` selects a `model_registry.json` key (omit to use the package default).
- `--control-frame` picks the tool: `blue_brush`, `flat_spatula`, `spoon_spatula`,
  `mallet_hammer`, `claw_hammer`.
- `--instruction` is optional; when unset, the basic per-task default for the
  control frame is used.

Subscribes: `--tool-topic` (`/robot_frame/current_tool_pose`),
`--material-topic` (`/robot_frame/current_material_pose`).
Publishes: `/robot_frame/goal_object_pose`.

## 7. Troubleshooting

- **`closed_loop package not found`** — the ROS interpreter isn't the one you
  `pip install -e`'d into. Install into that exact env.
- **CLIP download / network errors on the robot** — use the packaged
  `assets/clip_cache/` (option 1 above) so `local_files_only` kicks in.
- **`Unknown model key ...`** — run `python -c "from closed_loop import list_models; print(list_models())"`
  to see valid keys; the checkpoint file must exist under `assets/`.
- **CUDA not available** — pass `--device cpu` (slower) or install a CUDA-enabled
  torch build matching the robot's drivers.
