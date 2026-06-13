"""Configuration for waypoint-trajectory action training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ActionTrajectoryConfig:
    dataset_dir: str
    shard_path: str
    output_dir: str
    normalization_stats_path: str

    max_shards: int = 0
    """If >0, load the first N alphabetical shard files under dataset_dir (ignores shard_path)."""

    dataset_dirs: list[str] = field(default_factory=list)
    """If non-empty, load shards from multiple dataset directories (ignores dataset_dir)."""

    max_shards_per_dir: list[int] = field(default_factory=list)
    """Per-directory shard caps aligned with dataset_dirs; 0 means all shards in that dir."""

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
    run_tag: str = ""
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

    normalization_eps: float = 1e-8

    use_wandb: bool = False
    wandb_project: str = "generative-str-action-expert"
    wandb_entity: str = ""
    wandb_group: str = "action_trajectory"
    wandb_run_name: str = ""
    # When wandb_run_name is empty, the run is named "{wandb_run_prefix}_{idx:04d}"
    # using the local run dir index (falls back to the run dir name if empty).
    wandb_run_prefix: str = ""
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str = ""
    wandb_prediction_num_examples: int = 8

    cosine_warmup_steps: int = 200


def load_config(path: Path) -> ActionTrajectoryConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ActionTrajectoryConfig(**raw)
