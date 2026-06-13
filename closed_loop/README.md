# closed_loop

Standalone inference package for the brush VLA (receding-horizon closed loop). Train a checkpoint elsewhere in this repo, then drop it here and run on the robot via SimToolReal.

## Deployable unit

Copy to the SimToolReal machine:

- `closed_loop/` (this package)
- `simtoolreal/` (ROS deployment; only needs the new `goal_pose_node_closed_loop.py` added)

`closed_loop` does **not** import `training/`, `generative_str_pipeline/`, `policy_exec/`, or IsaacGym.

## Model drop-in

1. Train in the repo → get `checkpoint_best.pt` and `normalization_stats_*.json`.
2. Copy into `closed_loop/closed_loop/assets/`:
   - `checkpoint_best.pt`
   - `normalization_stats.json`
3. Run examples or the ROS node. Architecture hyperparams are read from the checkpoint `config` dict.

## Model registry (multiple checkpoints)

The package can hold more than one deployable VLA. Every packaged checkpoint is
listed in `closed_loop/closed_loop/assets/model_registry.json` with its
normalization stats, control-frame hints, and provenance. Architecture
hyperparameters are still read from each checkpoint's own `config` dict at load
time; the registry only locates files and records metadata.

Registered models (run `python -c "from closed_loop import list_models; print(list_models())"`):

- `brush_sweep_reactive_best` — legacy single-task brush sweep (package default;
  `checkpoint_best.pt` + `normalization_stats.json`).
- `all_tasks_joint_pretrain_epoch10` — **joint pretrain on all four reactive tasks
  (brush sweep, spatula flip, spoon pour, hammer nail)**, epoch-10 checkpoint of the
  joint pretrain run (`training/runs/action_trajectory/run_0026/checkpoint_epoch_0010.pt`).
- `all_tasks_grpo_iter10` / `all_tasks_grpo_iter20` — all-task GRPO post-train of the
  joint pretrain (`training/runs/action_trajectory_grpo/run_0051`), after 10 and 20
  GRPO iterations respectively.
- `spatula_flip_pretrain_epoch10`, `spoon_pour_pretrain_epoch10`,
  `hammer_nail_pretrain_epoch10` — single-task epoch-10 pretrains for those tasks.

All-tasks models share `normalization_stats_all_tasks.json` and
`config_all_tasks_10epoch.yaml`; each single-task model ships its own stats/config.

Load by key (paths resolved from the registry; architecture read from the ckpt):

```python
from closed_loop import load_closed_loop_policy

policy = load_closed_loop_policy(
    "all_tasks_grpo_iter20",
    device="cuda",
    control_frame="flat_spatula",  # pick the tool for the task
    instruction="Flip the cube with the spatula",
)
```

Or resolve just the asset paths (no model loaded):

```python
from closed_loop import resolve_model
m = resolve_model("all_tasks_grpo_iter20")
print(m.checkpoint_path, m.normalization_stats_path, m.tasks)
```

`BrushPolicy` / `ClosedLoopBrushPolicy` also accept explicit
`checkpoint_path=` and `normalization_stats_path=` if you prefer not to use the
registry. To make the all-tasks model the implicit default for the examples/ROS
node, either change `"default"` in `model_registry.json` or copy its two files
over `checkpoint_best.pt` / `normalization_stats.json`.

## Install

```bash
cd closed_loop
pip install -e .
```

For the viser control-frame selector:

```bash
pip install -e ".[viz]"
```

CLIP text tower: set `HF_HOME` / `TRANSFORMERS_CACHE` or place weights under `closed_loop/closed_loop/assets/clip_cache/`.

## Usage (Python API)

```python
from closed_loop import ClosedLoopBrushPolicy

policy = ClosedLoopBrushPolicy(
    device="cuda",
    control_frame="blue_brush",
    instruction="Sweep the cube to the goal with the brush",
    frame_shift=(0.0, 0.8, 0.0),
    chunk_size=5,
)
policy.set_destination([-0.365, -0.056, 0.517])  # robot frame
policy.reset(tool_root_xyz, tool_root_quat_xyzw, material_xyz)
while not policy.done:
    chunk = policy.plan_chunk()  # list of (xyz, quat_xyzw) robot frame
    # execute on robot, then:
    policy.observe(new_tool_root_xyz, new_tool_root_quat_xyzw, new_material_xyz)
```

## ROS node (SimToolReal)

```bash
# from simtoolreal repo root, after pip install -e ../closed_loop
python simtoolreal/deployment/goal_pose_node_closed_loop.py \
  --fixed-destination -0.365 -0.056 0.517 \
  --control-frame blue_brush
```

Subscribes: `--tool-topic`, `--material-topic` (defaults in node). Publishes `/robot_frame/goal_object_pose`.

## Retargeting

Pass `--control-frame <name>` or a path to any viser-annotated JSON under `assets/control_frames/`. The VLA outputs contact frames; `T_obj_from_contact` maps them to the selected object's root pose.

## Control-frame annotation

```bash
python -m closed_loop.tools.annotate_object_control_point --object-name blue_brush
```

Writes `closed_loop/closed_loop/assets/control_frames/<object>.json`.
