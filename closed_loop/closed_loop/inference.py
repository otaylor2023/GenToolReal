"""VLA inference: load checkpoint, predict 15 contact-frame waypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from closed_loop.model import ActionTrajectoryModel
from closed_loop.paths import (
    apply_clip_cache_env,
    default_checkpoint_path,
    default_clip_cache_dir,
    default_normalization_stats_path,
    resolve_control_frame,
)
from closed_loop.scene import SceneState
from closed_loop.text_encoder import ClipTextEncoder
from closed_loop.viz import default_instruction_for_control_frame
from closed_loop.waypoint_to_pose import load_control_frame, waypoint_to_object_pose
from closed_loop.xyz_normalization import load_xyz_normalization_stats, normalize_xyz_np


def _rollout(
    *,
    model: ActionTrajectoryModel,
    batch_tensors: Dict[str, Any],
    steps: int,
    n_samples: int,
) -> torch.Tensor:
    device = batch_tensors["has_material"].device
    bsz = 1
    dt = 1.0 / max(1, int(steps))
    finals: List[torch.Tensor] = []
    for _ in range(int(n_samples)):
        x = torch.randn(bsz, ActionTrajectoryModel.ACTION_DIM, device=device)
        for i in range(int(steps)):
            t = torch.full((bsz,), float(i) / float(max(1, steps)), device=device)
            pred = model(
                instr_clip=batch_tensors["instr_clip"],
                tool_clip=batch_tensors["tool_clip"],
                material_clip=batch_tensors["material_clip"],
                destination_clip=batch_tensors["destination_clip"],
                table_clip=batch_tensors["table_clip"],
                tool_contact_xyz_norm=batch_tensors["tool_contact_xyz_norm"],
                tool_normal=batch_tensors["tool_normal"],
                tool_surface_dir=batch_tensors["tool_surface_dir"],
                material_xyz_norm=batch_tensors["material_xyz_norm"],
                destination_xyz_norm=batch_tensors["destination_xyz_norm"],
                table_xyz_norm=batch_tensors["table_xyz_norm"],
                xt=x,
                t=t,
                has_material=batch_tensors["has_material"],
                has_destination=batch_tensors["has_destination"],
            )
            x = x + pred * dt
        finals.append(x)
    return torch.stack(finals, dim=1)


def _scene_to_batch_tensors(
    scene: SceneState,
    *,
    xyz_mean: np.ndarray,
    xyz_std: np.ndarray,
    norm_eps: float,
    clip: ClipTextEncoder,
    device: torch.device,
) -> Dict[str, Any]:
    def _n_pos(x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64).reshape(1, 3)
        return normalize_xyz_np(a, xyz_mean, xyz_std, norm_eps)[0].astype(np.float32)

    instr_clip = clip.encode([scene.instruction])
    tool_clip = clip.encode([scene.tool_label])
    material_clip = clip.encode(["the cube"])
    destination_clip = clip.encode(["the goal"])
    table_clip = clip.encode(["table surface center"])

    return {
        "instr_clip": instr_clip,
        "tool_clip": tool_clip,
        "material_clip": material_clip,
        "destination_clip": destination_clip,
        "table_clip": table_clip,
        "tool_contact_xyz_norm": torch.from_numpy(_n_pos(scene.tool_contact_xyz_world)).to(device),
        "tool_normal": torch.from_numpy(
            np.asarray(scene.tool_current_normal, dtype=np.float32).reshape(3)
        ).to(device),
        "tool_surface_dir": torch.from_numpy(
            np.asarray(scene.tool_current_surface_dir, dtype=np.float32).reshape(3)
        ).to(device),
        "material_xyz_norm": torch.from_numpy(_n_pos(scene.material_xyz_world)).to(device),
        "destination_xyz_norm": torch.from_numpy(_n_pos(scene.destination_xyz_world)).to(device),
        "table_xyz_norm": torch.from_numpy(_n_pos(scene.table_xyz_world)).to(device),
        "has_material": torch.tensor([True], device=device),
        "has_destination": torch.tensor([True], device=device),
    }


class BrushPolicy:
    """Load trained VLA and run single-shot inference."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        checkpoint_path: Path | None = None,
        normalization_stats_path: Path | None = None,
        control_frame: str = "blue_brush",
        clip_model_id: str | None = None,
        clip_cache_dir: Path | None = None,
        local_files_only: bool = False,
        integration_steps: int | None = None,
        inference_samples: int | None = None,
        instruction: str | None = None,
        tool_label: str = "the brush",
        table_z: float = 0.53,
        seed: int = 7,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        ckpt_path = Path(checkpoint_path or default_checkpoint_path())
        stats_path = Path(normalization_stats_path or default_normalization_stats_path())
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not stats_path.is_file():
            raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt.get("config") or {}
        self._cfg = cfg
        self.instruction = (
            str(instruction)
            if instruction is not None
            else default_instruction_for_control_frame(control_frame)
        )
        self.tool_label = str(tool_label)
        self.table_z = float(table_z)
        self.integration_steps = int(integration_steps or cfg.get("integration_steps", 30))
        self.inference_samples = int(inference_samples or cfg.get("inference_samples", 4))
        self.norm_eps = float(cfg.get("normalization_eps", 1e-8))

        mean_np, std_np, norm_eps_file = load_xyz_normalization_stats(stats_path)
        self.norm_eps = float(norm_eps_file)
        self._xyz_mean_np = mean_np
        self._xyz_std_np = std_np
        self._xyz_mean = torch.tensor(mean_np, dtype=torch.float32, device=self.device)
        self._xyz_std = torch.tensor(std_np, dtype=torch.float32, device=self.device)

        cache = clip_cache_dir or default_clip_cache_dir()
        clip_id = str(clip_model_id or cfg.get("clip_model_id", "openai/clip-vit-base-patch32"))
        model_cache_name = f"models--{clip_id.replace('/', '--')}"
        clip_cache_dir_arg: str | None = None

        def _first_snapshot(snap_root: Path) -> Path | None:
            if not snap_root.is_dir():
                return None
            snaps = sorted(snap_root.iterdir())
            return snaps[0] if snaps else None

        packaged_snap_root = cache / "hub" / model_cache_name / "snapshots"
        snap = _first_snapshot(packaged_snap_root)
        if snap is not None:
            apply_clip_cache_env(cache)
            clip_id = str(snap)
            local_files_only = True
            clip_cache_dir_arg = str(cache / "hub")
        elif not local_files_only:
            import os
            from pathlib import Path as _Path

            hf_home = _Path(os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface"))
            snap = _first_snapshot(hf_home / model_cache_name / "snapshots")
            if snap is not None:
                os.environ.setdefault("HF_HOME", str(hf_home))
                os.environ.setdefault("HF_HUB_CACHE", str(hf_home))
                clip_id = str(snap)
                local_files_only = True

        self.clip = ClipTextEncoder(
            model_id=clip_id,
            device=self.device,
            cache_dir=clip_cache_dir_arg,
            local_files_only=local_files_only,
        )
        self.model = ActionTrajectoryModel(
            d_clip=int(self.clip.d_clip),
            d_model=int(cfg.get("d_model", 512)),
            num_heads=int(cfg.get("num_heads", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            dropout=float(cfg.get("action_dropout", 0.0)),
            ffn_multiplier=int(cfg.get("ffn_multiplier", 4)),
            pos_norm_denom=float(cfg.get("pos_norm_denom", 1.0)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.clip.eval()

        cf_path = resolve_control_frame(control_frame)
        self.T_oc = load_control_frame(cf_path)
        self.control_frame_path = cf_path

    def set_control_frame(self, name_or_path: str) -> Path:
        """Reload ``T_obj_from_contact`` from a packaged or absolute JSON path."""
        cf_path = resolve_control_frame(name_or_path)
        self.T_oc = load_control_frame(cf_path)
        self.control_frame_path = cf_path
        return cf_path

    @torch.no_grad()
    def predict_waypoints(self, scene: SceneState) -> np.ndarray:
        """Return [15, 9] contact-frame waypoints in model/world frame."""
        batch = _scene_to_batch_tensors(
            scene,
            xyz_mean=self._xyz_mean_np,
            xyz_std=self._xyz_std_np,
            norm_eps=self.norm_eps,
            clip=self.clip,
            device=self.device,
        )
        samples_out = _rollout(
            model=self.model,
            batch_tensors=batch,
            steps=self.integration_steps,
            n_samples=self.inference_samples,
        )
        pred_norm = samples_out[0].mean(dim=0)
        c, nrm, sd = ActionTrajectoryModel.postprocess_waypoints(
            pred_norm.unsqueeze(0), self._xyz_mean, self._xyz_std, self.norm_eps
        )
        waypoints = torch.cat([c, nrm, sd], dim=-1).reshape(-1, 9).cpu().numpy().astype(np.float32)
        return waypoints

    def waypoints_to_object_poses_robot(
        self,
        waypoints: np.ndarray,
        frame_shift: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Convert model-frame waypoints to robot-frame object root poses (xyz, quat_xyzw)."""
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(waypoints.shape[0]):
            xyz_m, quat = waypoint_to_object_pose(
                waypoints[i, 0:3], waypoints[i, 3:6], waypoints[i, 6:9], self.T_oc
            )
            xyz_r = xyz_m.copy()
            xyz_r = xyz_r - np.asarray(frame_shift, dtype=np.float64).reshape(3)
            out.append((xyz_r.astype(np.float64), quat.astype(np.float64)))
        return out
