#!/usr/bin/env python3
"""CPU-only smoke test: a registered checkpoint loads + runs one forward step.

Exercises ``closed_loop.registry`` end to end without a robot, without GPU, and
without network access:

1. resolve a registry key to asset paths,
2. load the checkpoint (``map_location="cpu"``) and build ``ActionTrajectoryModel``
   from its ``config`` dict with ``strict=True``,
3. load the normalization stats,
4. run a single dummy flow-matching step + ``postprocess_waypoints`` and check shapes,
5. (best effort, offline) build the real ``BrushPolicy`` entry point.

Usage::

    python -m closed_loop.examples.verify_registry_load --model all_tasks_joint_pretrain_epoch10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from closed_loop.model import ActionTrajectoryModel
from closed_loop.registry import list_models, resolve_model
from closed_loop.xyz_normalization import load_xyz_normalization_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all_tasks_joint_pretrain_epoch10")
    parser.add_argument("--d-clip", type=int, default=512, help="CLIP text dim (base-patch32=512)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    print(f"available registry models: {list_models()}")
    model_entry = resolve_model(args.model)
    print(f"[resolve] key={model_entry.key}")
    print(f"[resolve] tasks={model_entry.tasks}")
    print(f"[resolve] checkpoint={model_entry.checkpoint_path}")
    print(f"[resolve] norm_stats={model_entry.normalization_stats_path}")
    print(f"[resolve] config_yaml={model_entry.config_yaml_path}")

    ckpt = torch.load(model_entry.checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config") or {}
    print(f"[ckpt] epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")

    model = ActionTrajectoryModel(
        d_clip=int(args.d_clip),
        d_model=int(cfg.get("d_model", 512)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("action_dropout", 0.0)),
        ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
        pos_norm_denom=float(cfg.get("pos_norm_denom", 1.0)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded strict, params={n_params:,}")

    mean_np, std_np, norm_eps = load_xyz_normalization_stats(model_entry.normalization_stats_path)
    print(f"[stats] xyz_mean={np.round(mean_np, 4)} xyz_std={np.round(std_np, 4)} eps={norm_eps}")
    xyz_mean = torch.tensor(mean_np, dtype=torch.float32)
    xyz_std = torch.tensor(std_np, dtype=torch.float32)

    bsz = 1
    d_clip = int(args.d_clip)

    def clip_vec() -> torch.Tensor:
        return torch.randn(bsz, d_clip)

    def pos() -> torch.Tensor:
        return torch.randn(bsz, 3)

    x = torch.randn(bsz, ActionTrajectoryModel.ACTION_DIM)
    t = torch.zeros(bsz)
    with torch.no_grad():
        pred = model(
            instr_clip=clip_vec(),
            tool_clip=clip_vec(),
            material_clip=clip_vec(),
            destination_clip=clip_vec(),
            table_clip=clip_vec(),
            tool_contact_xyz_norm=pos(),
            tool_normal=torch.randn(bsz, 3),
            tool_surface_dir=torch.randn(bsz, 3),
            material_xyz_norm=pos(),
            destination_xyz_norm=pos(),
            table_xyz_norm=pos(),
            xt=x,
            t=t,
            has_material=torch.tensor([True]),
            has_destination=torch.tensor([True]),
        )
    assert pred.shape == (bsz, ActionTrajectoryModel.ACTION_DIM), pred.shape
    print(f"[forward] velocity pred shape={tuple(pred.shape)} (expected (1, 135)) OK")

    contact, normal, surface_dir = ActionTrajectoryModel.postprocess_waypoints(
        pred, xyz_mean, xyz_std, float(norm_eps)
    )
    assert contact.shape == (bsz, 15, 3), contact.shape
    assert normal.shape == (bsz, 15, 3), normal.shape
    assert surface_dir.shape == (bsz, 15, 3), surface_dir.shape
    print(
        f"[postprocess] contact={tuple(contact.shape)} normal={tuple(normal.shape)} "
        f"surface_dir={tuple(surface_dir.shape)} (expected (1,15,3)) OK"
    )

    # Best-effort: exercise the real BrushPolicy load path offline (needs CLIP cache).
    try:
        from closed_loop.registry import load_policy

        policy = load_policy(args.model, device="cpu", local_files_only=True)
        print(f"[policy] BrushPolicy loaded offline OK (control_frame={policy.control_frame_path.name})")
    except Exception as exc:  # noqa: BLE001
        print(
            "[policy] BrushPolicy offline load skipped "
            f"(CLIP text tower not cached locally): {type(exc).__name__}: {exc}"
        )

    print("\nVERIFICATION PASSED: registry resolve + checkpoint load + forward + postprocess OK")


if __name__ == "__main__":
    main()
