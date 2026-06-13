"""Configuration for image-free action-expert training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ActionNoImageConfig:
    dataset_dir: str
    shard_path: str
    output_dir: str
    normalization_stats_path: str

    max_shards: int = 0
    """If >0, load the first N alphabetical *_shard.json files under dataset_dir (ignores shard_path)."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 3.0e-4
    action_expert_lr: float = 3.0e-4
    label_proj_lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 5
    log_every_steps: int = 10
    metrics_log_every_steps: int = 10
    device: str = "cuda"
    seed: int = 7
    num_workers: int = 2
    use_amp: bool = False
    run_tag: str = ""  # ignored for run directory naming (always run_NNNN); kept for YAML compatibility
    train_fraction: float = 0.85
    val_fraction: float = 0.10

    clip_model_id: str = "openai/clip-vit-base-patch32"
    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    local_files_only: bool = False

    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    ffn_multiplier: int = 4
    action_dropout: float = 0.0
    pos_norm_denom: float = 1.0

    integration_steps: int = 50
    inference_samples: int = 16

    table_label: str = "table surface center"
    table_xyz_world: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.53])

    explode_instruction_variants: bool = False
    sample_goal_in_constraint_region: bool = True
    goal_rejection_sample_max_attempts: int = 512

    normalization_eps: float = 1e-8

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

    use_wandb: bool = False
    wandb_project: str = "generative-str-action-expert"
    wandb_entity: str = ""
    wandb_group: str = "action_no_image"
    wandb_run_name: str = ""
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str = ""
    wandb_prediction_num_examples: int = 8
    depth_occlusion_eps_m: float = 0.025

    cosine_warmup_steps: int = 200


def load_config(path: Path) -> ActionNoImageConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ActionNoImageConfig(**raw)
