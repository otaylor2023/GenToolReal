# Training Modules

## PPO (existing)
- Launch with `python -m training.ppo.launch_tool_position_ppo --config training/cfg/tool_position_ppo.yaml`

## Behavior Cloning (new)
- Launch with `python -m training.bc.launch_tool_position_bc --config training/cfg/tool_position_bc.yaml`
- Primary trainer: `training/bc/train_tool_position_bc.py`
- Data split: shard-level `85/10/5` train/val/test
- Loss schedule:
  - Epochs `1..20`: canonical XYZ MSE
  - Epochs `21+`: region-aware constraint loss

## Quick stub smoke run
Use this to validate the BC entrypoint without loading real Qwen:

`python -m training.bc.train_tool_position_bc --config training/cfg/tool_position_bc.yaml`

Set `use_real_qwen: false` in the config for lightweight startup checks.

## Action Expert (VLM + flow matching)
- `google/paligemma-3b-pt-224` is a gated Hugging Face repo: accept the model terms on the Hub, then put `HF_TOKEN=...` in the repo root `.env` (or export it in your shell). The trainer loads `.env` before download (`training/action_expert/hf_env.py`).
- Weights cache: `hf_cache_dir` in the YAML sets `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` at process start so Hub + Transformers reuse the same tree (under `hf_cache_dir/hub/` for model snapshots). `PaliGemmaContextEncoder` still passes `cache_dir=` into `from_pretrained` for explicit alignment.
- Metrics: each run writes `progress.json` (updated during training and after eval), `train_metrics.jsonl` (one JSON object per `metrics_log_every_steps` train step plus epoch summaries and eval rows), and `metrics.json` (epoch-level eval history). Optional Weights & Biases: set `use_wandb: true` in the YAML (and `wandb_project` / `wandb_entity` as needed), install `wandb`, then put `WANDB_API_KEY=...` in the repo root `.env` (loaded at startup like `HF_TOKEN`) or run `wandb login`.
- XYZ: build `training/cfg/normalization_stats.json` once with `python -m training.action_expert.compute_xyz_normalization --dataset_dir <shards_dir> --output training/cfg/normalization_stats.json`. Training uses normalized keypoints + flow targets; eval denormalizes predictions for L2 and region checks. Keypoint `label_emb` uses `Linear` + `LayerNorm` after mean-pooled PaliGemma token embeddings.
- Launch with `python -m training.action_expert.launch_action_expert --config training/cfg/tool_position_action_expert.yaml`
- Primary trainer: `training/action_expert/train_action_expert.py`
- Evaluator: `python -m training.action_expert.eval_action_expert --config training/cfg/tool_position_action_expert.yaml --checkpoint training/runs/tool_position_action_expert/<run_id>/checkpoint_best.pt`
- Input structure:
  - VLM text path uses plain text (`system prompt + instruction + object labels`).
  - VLM image path uses PaliGemma image input.
  - Action expert consumes fused keypoint tokens (`label_emb + pos_emb`) with noisy position token last.
- PaliGemma adaptation:
  - SigLIP vision encoder is fully frozen.
  - LoRA is applied to language attention `W_q/W_v` modules.
- Training objective is flow-matching velocity MSE only.
- Region constraints are used for:
  - training sample quality filtering (downweight/discard invalid labels),
  - inference sample filtering before averaging multi-sample predictions.

