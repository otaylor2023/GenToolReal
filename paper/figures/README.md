# Paper Figures

Place or symlink rendered assets here before final submission. The paper compiles with placeholder paths; copy assets from the repo as needed.

## Suggested sources

| Figure | Source in repo | Notes |
|--------|----------------|-------|
| `system_overview.png` | Compose from `00_main.png` + closed-loop diagram | High-level pipeline: observe → VLA → waypoints → SimToolReal |
| `contact_frame.png` | Screenshot from `closed_loop/closed_loop/tools/viz_interactive.py` | Show normal + surface_dir arrows on tool |
| `reactive_rollout.png` | `generative_str_pipeline/render_reactive_rollout_viz.py` output | Multi-generation closed-loop data |
| `grpo_architecture.png` | Diagram in paper (TikZ) or export from wandb | Multi-process GRPO worker layout |
| `task_montage.png` | Viz renders for brush sweep and hammer nail | Demonstrated-task montage |
| `robot_deploy.png` | `goal_pose_node_closed_loop_viz.py` Viser screenshot | Full 15-pt trajectory on robot |

## Existing assets (repo root)

- `00_main.png` — early goal-prediction visualization (background context)
- `00_main_keypoints.png` — keypoint-based scene
- `00_main_target_dot.png` — goal target visualization

## Commands to regenerate

```bash
# Interactive closed-loop viz (requires [viz] extras)
python -m closed_loop.tools.viz_interactive --model brush_hammer_joint_pretrain_epoch10

# Reactive rollout visualization
python generative_str_pipeline/render_reactive_rollout_viz.py --shard <path> --scene 0

# ROS debug node (on robot machine)
python simtoolreal/deployment/goal_pose_node_closed_loop_viz.py --model all_tasks_grpo_iter20
```
