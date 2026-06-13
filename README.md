# GenToolReal: Generative Trajectory Prediction for Reactive Tool Manipulation via Language-Conditioned Flow Matching

A reactive, state-conditioned Language-Action model for dexterous tool
manipulation. Given a natural-language instruction and the current state of the
tool and the object it acts on, the model predicts a short-horizon trajectory of
**contact-frame waypoints**, executes the beginning of it, then reobserves and
replans. Because trajectories are expressed in a contact frame anchored to the
tool's functional surface, a single policy form is not keyed to a hard-coded
tool identity and works across tools whose functional surfaces differ
(demonstrated on brush sweeping and nail hammering).

The predicted poses are handed to the [SimToolReal](https://simtoolreal.github.io/)
controller, which executes them on a 22-DoF Sharpa five-fingered hand mounted on
a 7-DoF KUKA iiwa 14 arm.

- **Paper:** `paper/main.tex` (build with `pdflatex`; see `paper/README.md`)
- **Project website:** `docs/` (static site, GitHub Pages from `/docs`)

## Pipeline at a glance

```
FoundationPose  ->  symbolic 6D state  ->  LA flow model  ->  contact-frame waypoints
   (perception)     (tool, object,          (this repo)        |
        ^            destination, table)                       v
        |                                              SimToolReal controller
        +------------------ reobserve & replan ------------- (Sharpa hand + KUKA arm)
```

## Repository layout (core, on `main`)

| Path | What it is |
|------|------------|
| `training/action_trajectory/` | Flow-matching trajectory model + joint pretraining (pure PyTorch, no simulator). `model.py`, `train.py`, `dataset.py`, `launch.py`. |
| `training/action_trajectory_rl/` | Multi-task GRPO fine-tuning and simulation eval. `train_grpo.py`, `flow_grpo.py`, `reward.py`, `closed_loop_rollout.py`, `eval_all_tasks.py`. Requires `policy_exec/` (see below). |
| `training/cfg/` | Training/GRPO YAML configs (`action_trajectory_*`, `*_grpo_*`). |
| `generative_str_pipeline/` | Procedural reactive data generation (`build_dataset_00XX_*.py`) and the `sim_rollout/` + `sim_workspace.py` helpers GRPO uses. |
| `closed_loop/` | Standalone inference package for robot deployment. Light deps (numpy, torch, transformers); does **not** import `training/`, the pipeline, `policy_exec/`, or IsaacGym. |
| `closed_loop/deployment/` | ROS nodes that run the policy on the robot (copied into SimToolReal at deploy time — see `closed_loop/deployment/README.md`). |
| `paper/` | LaTeX paper. |
| `docs/` | Project website (static, GitHub Pages). |

## Quickstart

### 1. Pretrain the trajectory model (no simulator needed)

```bash
python -m training.action_trajectory.launch \
    --config training/cfg/action_trajectory_all_tasks_10epoch.yaml
```

### 2. GRPO fine-tuning + sim eval (needs `policy_exec/` + IsaacGym)

See [`POLICY_EXEC_SETUP.md`](POLICY_EXEC_SETUP.md) to set up the simulation
runtime, then:

```bash
export PYTHONPATH=$(pwd):$(pwd)/policy_exec:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/policy_exec/lib:$LD_LIBRARY_PATH
$HOME/miniconda3/envs/policy_exec/bin/python -m training.action_trajectory_rl.train_grpo \
    --config training/cfg/action_trajectory_all_tasks_grpo_from_10epoch.yaml
```

### 3. Closed-loop deployment

```bash
pip install -e closed_loop
# drop a trained checkpoint + normalization stats into closed_loop/closed_loop/assets/
# then run an example or the ROS node (see closed_loop/README.md + closed_loop/deployment/README.md)
```

## External dependencies (not committed)

These are large and/or maintained elsewhere, so they are **gitignored** and must
be obtained separately:

| Dependency | Needed for | How to get it |
|------------|-----------|---------------|
| `policy_exec/` | GRPO fine-tuning, sim eval, sim rollout rendering | [`POLICY_EXEC_SETUP.md`](POLICY_EXEC_SETUP.md) |
| `simtoolreal/` | Real-robot deployment (low-level controller + ROS) | https://simtoolreal.github.io/ — clone alongside this repo. Add our nodes from `closed_loop/deployment/`. |
| `FoundationPose/` | 6D pose estimation in the data pipeline / deployment | Clone from the FoundationPose project. |
| `sam-3d-objects/` | Object meshing in the data pipeline | Clone from the SAM-3D-Objects project. |

Large artifacts are also gitignored: datasets, training runs, model checkpoints
(`*.pt` / `*.pth`), `wandb/`, logs, and generated media. **Trained checkpoints**
(including the `closed_loop` deployable weights) are not in git; add them via a
release artifact, `scp`, or Git LFS.

## Branches

- **`main`** — the core GenToolReal code above.
- **`archive`** — adds exploratory / superseded code we did not end up using
  (earlier action-expert / BC / PPO trainers, the Gemini/Cosmos VLM
  experiments, the VLM sidecar, pose-track tooling, and two one-off scripts).
  These dirs are listed under "Off-main (archived) code" in `.gitignore` and are
  force-added (skipping venvs/runs) only on `archive`. The `isaaclab_simtoolreal/`
  and `isaacsim_envs/` sandboxes are *not* archived in git — they are several GB
  of vendored meshes/weights and are reconstructable from the vendored sim
  assets.

To build the branches (no remote is configured yet):

```bash
# identity, if not already set
git config user.name  "Your Name"
git config user.email "you@example.com"

# 1) main: core code only (.gitignore excludes data, vendored deps, archived code)
git checkout -b main
git add -A
git commit -m "GenToolReal core: training, data-gen, closed_loop, paper, docs"

# 2) archive: main + exploratory/superseded code (venvs/runs skipped)
git checkout -b archive
git add -f training/action_expert training/bc training/ppo \
    training/action_no_image training/gemini \
    cosmos_vlm ':!cosmos_vlm/runs' \
    vlm_sidecar ':!vlm_sidecar/.venv' \
    pose_track ':!pose_track/.venv' \
    gemini_robotics_er.py veo_frame_to_video.py
git commit -m "Archive: exploratory and superseded code"
git checkout main

# 3) when you have a remote:
# git remote add origin <git-url>
# git push -u origin main
# git push -u origin archive
```

## License / authorship

See repository settings; contributions and authorship are managed by the repo
owner.
