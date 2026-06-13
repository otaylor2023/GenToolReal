# Reconstructing `policy_exec/` (IsaacGym simulation runtime)

`policy_exec/` is **not** committed to this repo. It is a large (~17 GB),
self-contained copy of the SimToolReal / DexToolBench policy-execution stack
(IsaacGym environments, the pretrained low-level controller, and tool/object
assets). GenToolReal only *uses* it as a runtime to roll out predicted tool
trajectories in simulation.

You only need `policy_exec/` for:

- **GRPO fine-tuning** — `training/action_trajectory_rl/train_grpo.py`
- **Simulation evaluation** — `training/action_trajectory_rl/eval_all_tasks.py`
- **Sim rollout rendering / dataset sim checks** — `generative_str_pipeline/sim_rollout/*`

You do **not** need it for:

- Flow-matching pretraining (`training/action_trajectory/`) — pure PyTorch.
- Closed-loop inference / robot deployment (`closed_loop/`).
- Building the paper or website.

## What the GenToolReal code imports from it

Everything funnels through a single sim wrapper:

```python
# training/action_trajectory_rl/train_grpo.py (and render_hammer_videos.py)
sys.path.insert(0, str(REPO_ROOT / "policy_exec"))
from dextoolbench.vec_rollout import VectorizedSimRollout
```

`VectorizedSimRollout` in turn pulls in `from isaacgym import ...`,
`deployment.isaac.isaac_env.create_env`, and `dextoolbench.eval`, plus the tool
and object assets under `policy_exec/assets/` and `policy_exec/dextoolbench/`.

## Expected directory tree

```
policy_exec/
  dextoolbench/            # DexToolBench: sim rollout + eval (vec_rollout.py, eval.py, objects.py, ...)
    data/                 # ~14 GB generated object/task data  (regenerate or copy from DexToolBench)
    trajectories/         # reference tool trajectories per object/task
  deployment/             # isaac/mujoco/fake env wrappers (deployment/isaac/isaac_env.py)
  env/                    # create_env helpers
  eval/                   # run_eval entry point
  isaacgymenvs/           # IsaacGymEnvs tasks (tasks/simtoolreal/env.py, base/vec_task.py, ...)
  rl_games/               # rl_games (vendored RL library used by the low-level controller)
  runtime/                # player.py runtime helpers
  assets/                 # URDFs + calibration (assets/urdf/dextoolbench/<category>/<tool>/*.obj)
  meshes/                 # ~890 MB tool/object/robot meshes
  pretrained_policy/      # low-level SimToolReal controller
    config.yaml
    model.pth             # ~395 MB weights
  third_party/
    isaacgym/             # IsaacGym Preview 4 (see install below)
```

## How to reconstruct

1. **IsaacGym Preview 4.** Download from NVIDIA and either install into your
   conda env or place under `policy_exec/third_party/isaacgym`. Create the env:

   ```bash
   conda create -n policy_exec python=3.8
   conda activate policy_exec
   cd policy_exec/third_party/isaacgym/python && pip install -e .
   ```

2. **SimToolReal / DexToolBench stack + assets.** Obtain the
   `dextoolbench/`, `deployment/`, `env/`, `eval/`, `isaacgymenvs/`,
   `rl_games/`, `runtime/`, `assets/`, and `meshes/` trees from the SimToolReal
   release (see https://simtoolreal.github.io/). The bulky
   `dextoolbench/data/` and `meshes/` can be regenerated with the DexToolBench
   object/trajectory generation scripts or copied from an existing checkout.

3. **Pretrained low-level controller.** Place `config.yaml` and `model.pth`
   under `policy_exec/pretrained_policy/`.

4. **Verify** with the smoke test:

   ```bash
   cd policy_exec && python -m dextoolbench.grpo_sim_smoke
   ```

## Running GRPO / eval once it exists

GRPO and eval must run in the `policy_exec` conda env (IsaacGym + transformers in
the same env), with `policy_exec` on `PYTHONPATH` and its libs on
`LD_LIBRARY_PATH`:

```bash
export PYTHONPATH=$(pwd):$(pwd)/policy_exec:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/policy_exec/lib:$LD_LIBRARY_PATH
PY=$HOME/miniconda3/envs/policy_exec/bin/python

# GRPO fine-tuning
$PY -m training.action_trajectory_rl.train_grpo --config training/cfg/action_trajectory_all_tasks_grpo_from_10epoch.yaml

# Simulation evaluation
$PY -m training.action_trajectory_rl.eval_all_tasks \
    --checkpoint training/runs/action_trajectory/run_XXXX/checkpoint_epoch_0010.pt \
    --out training/runs/action_trajectory_eval/eval.json --scenes-per-task 8
```
