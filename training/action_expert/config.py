"""Configuration schema for action-expert training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ActionExpertConfig:
    dataset_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    lora_lr: float
    action_expert_lr: float
    label_proj_weight_decay: float
    eval_every_epochs: int
    checkpoint_every_epochs: int
    log_every_steps: int
    device: str

    # Append `train_metrics.jsonl` on this step interval (independent of console `log_every_steps`).
    metrics_log_every_steps: int = 10
    gradient_accumulation_steps: int = 1
    cosine_warmup_steps: int = 500
    resume_checkpoint_path: str = ""
    # When > 0 and training is resumed from a checkpoint, apply a separate
    # clamped cosine decay (over this many *optimizer steps*) to the
    # `label_proj` + `action_expert` param groups, down to `eta_min`.
    action_expert_cosine_T_max_steps: int = 0
    action_expert_cosine_eta_min: float = 1e-5
    # Optional resume-only cosine tail for the LoRA group.
    lora_cosine_T_max_steps: int = 0
    lora_cosine_eta_min: float = 0.0
    # Bad-batch guards. If either threshold is exceeded, apply retry/skip logic.
    bad_batch_loss_threshold: float = 50.0
    bad_batch_grad_norm_threshold: float = 100.0
    bad_batch_abort_after_step: int = 300
    bad_batch_max_retries_per_step: int = 5
    # If None, `label_proj` uses `action_expert_lr` (backward compatible).
    label_proj_lr: float | None = None

    seed: int = 7
    num_workers: int = 4
    use_amp: bool = True
    image_size: int = 224
    run_tag: str = ""
    train_fraction: float = 0.85
    val_fraction: float = 0.10

    paligemma_model_id: str = "google/paligemma-3b-pt-224"
    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    local_files_only: bool = False
    enable_gradient_checkpointing: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    d_model: int = 1024
    num_heads: int = 8
    num_action_expert_layers: int = 6
    ffn_multiplier: int = 4
    action_dropout: float = 0.1
    pos_norm_denom: float = 1.0

    integration_steps: int = 20
    inference_samples: int = 16
    max_keypoints: int = 30

    # If true, expand each shard datapoint that contains 4 instruction strings into
    # 4 training samples (one per instruction variant). If the datapoint has only 1
    # instruction, behavior is unchanged.
    explode_instruction_variants: bool = False

    # If true, each __getitem__ replaces the dataset goal with a deterministic rejection
    # sample inside `satisfies_constraint` (seeded by scene / shard / datapoint / instruction).
    sample_goal_in_constraint_region: bool = False
    goal_rejection_sample_max_attempts: int = 512

    filter_invalid_train_examples: bool = True
    invalid_sample_weight: float = 0.25

    # BC region semantics for quality filtering/inference filtering.
    z_min_offset_m: float = -0.02
    z_max_offset_m: float = 0.20
    above_xy_min_m: float = 0.025
    above_xy_max_m: float = 0.05
    near_xy_min_m: float = 0.075
    near_xy_max_m: float = 0.10
    directional_axis_min_m: float = 0.075
    directional_axis_max_m: float = 0.10
    directional_lateral_tol_m: float = 0.03
    between_midpoint_fraction_of_span: float = 0.40
    exact_above_xy_radius_m: float = 0.01

    system_prompt: str = (
        "You are a robot control assistant that grounds instructions in the image and object labels."
    )

    # Global XYZ normalization (from `compute_xyz_normalization.py`).
    normalization_stats_path: str = "training/cfg/normalization_stats.json"
    normalization_eps: float = 1e-8

    # Weights & Biases (`pip install wandb`, then `wandb login` or `WANDB_API_KEY`).
    use_wandb: bool = False
    wandb_project: str = "generative-str-action-expert"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str = ""
    wandb_prediction_log_every_steps: int = 1000
    wandb_prediction_num_examples: int = 8


def load_config(path: Path) -> ActionExpertConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ActionExpertConfig(**raw)

