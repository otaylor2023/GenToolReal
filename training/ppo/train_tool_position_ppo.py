"""Standalone PPO trainer for tool-position policy (absolute world XYZ).

This trainer is intentionally separate from SimToolReal control-policy training.
It uses dataset shards built by gemini_position_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from training.ppo.tool_position_ppo_env import ToolPositionPPOEnv


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class QwenBackboneStub(nn.Module):
    """Placeholder VL backbone adaptor.

    In v1 planning we use Qwen2.5-VL. This stub keeps the module runnable even
    without loading large checkpoint weights; replace with actual Qwen encoder
    wiring when preparing full training runs.
    """

    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.output_dim = output_dim
        self.img_proj = nn.Sequential(
            nn.Linear(224 * 224 * 3, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )

    def forward(
        self,
        image_uint8: torch.Tensor,
        text_context: Sequence[str] | None = None,
        *,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        # image_uint8: [B, H, W, 3]
        x = image_uint8.float() / 255.0
        x = x.reshape(x.shape[0], -1)
        return self.img_proj(x)


class FrozenQwenVLBackbone(nn.Module):
    """Frozen Qwen2.5-VL encoder that returns one embedding per sample."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "FrozenQwenVLBackbone requires transformers with Qwen2.5-VL support"
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
        )
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine hidden_size for Qwen VL model config")
        self.output_dim = int(hidden_size)
        self.device = device

    def _encode_batch(self, image_uint8: torch.Tensor, text_context: Sequence[str]) -> torch.Tensor:
        if text_context is None or len(text_context) != image_uint8.shape[0]:
            raise ValueError("text_context batch must be provided for FrozenQwenVLBackbone")
        imgs = [img.detach().cpu().numpy().astype(np.uint8) for img in image_uint8]
        text_payloads: List[str] = []
        for txt in text_context:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": str(txt)},
                    ],
                }
            ]
            text_payloads.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        inputs = self.processor(
            text=text_payloads,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            last = out.hidden_states[-1]  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1).to(last.dtype)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float()

    def forward(
        self,
        image_uint8: torch.Tensor,
        text_context: Sequence[str],
        *,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        bsz = int(image_uint8.shape[0])
        csz = int(chunk_size)
        if csz <= 0 or csz >= bsz:
            return self._encode_batch(image_uint8, text_context)
        chunks: List[torch.Tensor] = []
        for start in range(0, bsz, csz):
            end = min(bsz, start + csz)
            chunks.append(
                self._encode_batch(image_uint8[start:end], list(text_context[start:end]))
            )
        return torch.cat(chunks, dim=0)


class ToolPositionActorCritic(nn.Module):
    """Qwen2.5-VL + MLP action head: fused_dim -> 1024 -> 512 -> 3."""

    def __init__(
        self,
        *,
        use_real_qwen: bool,
        vl_model_id: str,
        device: torch.device,
        hf_cache_dir: str,
        qwen_local_files_only: bool,
        qwen_forward_chunk_size: int,
    ):
        super().__init__()
        if use_real_qwen:
            self.vl = FrozenQwenVLBackbone(
                model_id=vl_model_id,
                device=device,
                cache_dir=hf_cache_dir,
                local_files_only=qwen_local_files_only,
            )
        else:
            self.vl = QwenBackboneStub(output_dim=768)
        self.qwen_forward_chunk_size = int(qwen_forward_chunk_size)
        fused_dim = int(self.vl.output_dim)
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 3),
        )
        self.log_std = nn.Parameter(torch.zeros(3))
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def _fused(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vl(
            obs["image"],
            obs.get("text_context"),
            chunk_size=self.qwen_forward_chunk_size,
        )

    def act(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused(obs)
        mean = self.action_head(fused)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.value_head(fused).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self._fused(obs)
        mean = self.action_head(fused)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_head(fused).squeeze(-1)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    dataset_dir: str
    output_dir: str
    total_updates: int
    rollout_steps: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    lr: float
    ppo_epochs: int
    minibatch_size: int
    eval_every_updates: int
    device: str
    checkpoint_every_updates: int = 25
    log_every_updates: int = 5
    num_envs: int = 1
    async_envs: bool = False
    eval_episodes: int = 32
    seed: int = 7
    run_tag: str = ""
    use_amp: bool = True
    token_success_bonus: float = 1.0
    exact_above_xy_margin_m: float = 0.01
    exact_above_z_margin_m: float = 0.01
    exact_above_w_xy: float = 1.5
    exact_above_w_z: float = 1.0
    task_prompt_template: str = (
        "You are a robot tool-positioning policy. Your goal is to output the absolute "
        "3D target position (x, y, z) in meters where the tool should move, based on "
        "the instruction. Movement is defined using the specified tool keypoint. You "
        "are given the scene image and a set of scene keypoints with coordinates to "
        "help you understand where objects are located. Return only the target position."
    )
    world_scale_prompt_template: str = (
        "Coordinate system: canonical tabletop frame in meters. The tabletop surface is z = 0. "
        "Axis directions are fixed: +X points to the right relative to the camera, +Y points away "
        "from the camera, and +Z points upward. All scene keypoint coordinates and the tool "
        "keypoint position are provided in this same frame."
    )
    use_real_qwen: bool = True
    vl_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    qwen_local_files_only: bool = False
    qwen_forward_chunk_size: int = 0


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _format_text_context(
    *,
    task_prompt: str,
    world_scale_prompt: str,
    instruction: str,
    tool_keypoint_label: str,
    tool_keypoint_xyz_world: Sequence[float],
    all_keypoints_label_position: Sequence[Dict[str, Any]],
) -> str:
    tool_xyz = [float(tool_keypoint_xyz_world[0]), float(tool_keypoint_xyz_world[1]), float(tool_keypoint_xyz_world[2])]
    tool_name = tool_keypoint_label.split(" of ", 1)[1] if " of " in tool_keypoint_label else "unknown"
    tool_kp_name = tool_keypoint_label.split(" of ", 1)[0]
    kp_items = []
    for kp in all_keypoints_label_position:
        lbl = str(kp.get("label", "")).strip()
        obj = str(kp.get("object_name", "")).strip()
        pos = kp.get("position_xyz_world")
        if not lbl or not isinstance(pos, (list, tuple)) or len(pos) != 3:
            continue
        full_label = f"{lbl} of {obj}" if obj else lbl
        kp_items.append(
            (
                obj.lower(),
                lbl.lower(),
                f"- {full_label}: [{float(pos[0]):.5f}, {float(pos[1]):.5f}, {float(pos[2]):.5f}]",
            )
        )
    kp_items.sort(key=lambda x: (x[0], x[1]))
    kp_lines = [row[2] for row in kp_items]
    all_kp_block = "\n".join(kp_lines)
    return (
        f"Task: {task_prompt}\n"
        f"WorldScale: {world_scale_prompt}\n\n"
        f"AllKeypoints:\n{all_kp_block}\n\n"
        f"Tool:\n"
        f"- object: {tool_name}\n"
        f"- keypoint: {tool_kp_name}\n"
        f"- position_xyz: [{tool_xyz[0]:.5f}, {tool_xyz[1]:.5f}, {tool_xyz[2]:.5f}]\n\n"
        f"Instruction:\n{instruction}"
    )


def _extract_text_context_batch(info: Dict[str, Any], batch_size: int) -> List[str]:
    out: List[str] = []
    for i in range(batch_size):
        out.append(
            _format_text_context(
                task_prompt=str(info["task_prompt"][i]),
                world_scale_prompt=str(info["world_scale_prompt"][i]),
                instruction=str(info["instruction"][i]),
                tool_keypoint_label=str(info["tool_keypoint_label"][i]),
                tool_keypoint_xyz_world=info["tool_keypoint_xyz_world"][i],
                all_keypoints_label_position=info["all_keypoints_label_position"][i],
            )
        )
    return out


def _extract_text_context_single(info: Dict[str, Any]) -> str:
    return _format_text_context(
        task_prompt=str(info["task_prompt"]),
        world_scale_prompt=str(info["world_scale_prompt"]),
        instruction=str(info["instruction"]),
        tool_keypoint_label=str(info["tool_keypoint_label"]),
        tool_keypoint_xyz_world=info["tool_keypoint_xyz_world"],
        all_keypoints_label_position=info["all_keypoints_label_position"],
    )


def _obs_to_tensor(
    obs: Dict[str, np.ndarray], device: torch.device, *, text_context: str
) -> Dict[str, Any]:
    return {
        "image": _to_tensor(obs["image"][None, ...], device),
        "text_context": [text_context],
    }


def _obs_batch_to_tensor(
    obs: Dict[str, np.ndarray], device: torch.device, *, text_context: Sequence[str]
) -> Dict[str, Any]:
    return {
        "image": _to_tensor(obs["image"], device),
        "text_context": list(text_context),
    }


def _to_float_stat(x: Any) -> float:
    if isinstance(x, np.ndarray):
        return float(np.mean(x))
    return float(x)


def _prepare_run_dir(output_root: Path, run_tag: str = "") -> Tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for p in output_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("run_"):
            continue
        suffix = name.split("_", 1)[1]
        numeric = suffix.split("_", 1)[0]
        if numeric.isdigit():
            max_idx = max(max_idx, int(numeric))
    next_idx = max_idx + 1
    run_id = f"run_{next_idx:04d}"
    clean_tag = str(run_tag).strip().replace(" ", "_")
    if clean_tag:
        run_id = f"{run_id}_{clean_tag}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def evaluate_policy(
    env: "ToolPositionPPOEnv",
    model: ToolPositionActorCritic,
    device: torch.device,
    episodes: int = 32,
) -> Dict[str, float]:
    errs: List[float] = []
    successes = 0
    lengths: List[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs, info = env.reset()
            text_context = _extract_text_context_single(info)
            done = False
            trunc = False
            ep_len = 0
            final_err = 1e9
            while not (done or trunc):
                t_obs = _obs_to_tensor(obs, device, text_context=text_context)
                action, _, _ = model.act(t_obs)
                obs, _, done, trunc, info = env.step(action.cpu().numpy()[0])
                text_context = _extract_text_context_single(info)
                ep_len += 1
                final_err = float(info["position_error_m"])
                if info.get("success", False):
                    successes += 1
            errs.append(final_err)
            lengths.append(ep_len)
    model.train()
    return {
        "eval_mean_final_error_m": float(np.mean(errs)) if errs else 0.0,
        "eval_success_rate": float(successes / max(1, episodes)),
        "eval_mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
    }


def train(cfg: PPOConfig) -> None:
    from training.ppo.tool_position_ppo_env import ToolPositionPPOEnv, load_episode_samples

    device = torch.device(cfg.device)
    output_root = Path(cfg.output_dir)
    out_dir, run_id = _prepare_run_dir(output_root, cfg.run_tag)
    print(f"[run] output_dir={out_dir}")

    train_samples = load_episode_samples(
        Path(cfg.dataset_dir),
        max_keypoints=128,
        split="train",
        task_prompt=cfg.task_prompt_template,
        world_scale_prompt=cfg.world_scale_prompt_template,
    )
    eval_samples = load_episode_samples(
        Path(cfg.dataset_dir),
        max_keypoints=128,
        split="eval",
        task_prompt=cfg.task_prompt_template,
        world_scale_prompt=cfg.world_scale_prompt_template,
    )
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(cfg.dataset_dir),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "config": asdict(cfg),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    progress_path = out_dir / "progress.json"
    num_envs = max(1, int(cfg.num_envs))

    def _make_train_env(rank: int):
        def _thunk():
            env = ToolPositionPPOEnv(
                train_samples,
                token_success_bonus=cfg.token_success_bonus,
                exact_above_xy_margin_m=cfg.exact_above_xy_margin_m,
                exact_above_z_margin_m=cfg.exact_above_z_margin_m,
                exact_above_w_xy=cfg.exact_above_w_xy,
                exact_above_w_z=cfg.exact_above_w_z,
            )
            env.seed(int(cfg.seed) + int(rank))
            return env

        return _thunk

    env_fns = [_make_train_env(i) for i in range(num_envs)]
    if bool(cfg.async_envs) and num_envs > 1:
        train_env = AsyncVectorEnv(env_fns)
    else:
        train_env = SyncVectorEnv(env_fns)

    eval_env = ToolPositionPPOEnv(
        eval_samples,
        token_success_bonus=cfg.token_success_bonus,
        exact_above_xy_margin_m=cfg.exact_above_xy_margin_m,
        exact_above_z_margin_m=cfg.exact_above_z_margin_m,
        exact_above_w_xy=cfg.exact_above_w_xy,
        exact_above_w_z=cfg.exact_above_w_z,
    )

    model = ToolPositionActorCritic(
        use_real_qwen=bool(cfg.use_real_qwen),
        vl_model_id=str(cfg.vl_model_id),
        device=device,
        hf_cache_dir=str(cfg.hf_cache_dir),
        qwen_local_files_only=bool(cfg.qwen_local_files_only),
        qwen_forward_chunk_size=int(cfg.qwen_forward_chunk_size),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics: List[Dict[str, Any]] = []
    obs_np, info_np = train_env.reset(seed=int(cfg.seed))
    text_context = _extract_text_context_batch(info_np, num_envs)

    def _checkpoint_payload(update: int) -> Dict[str, Any]:
        # Save only trainable policy components; Qwen VL stays frozen and reloaded from model_id/cache.
        return {
            "update": int(update),
            "run_id": run_id,
            "config": asdict(cfg),
            "model_trainable_state": {
                "action_head": model.action_head.state_dict(),
                "value_head": model.value_head.state_dict(),
                "log_std": model.log_std.detach().cpu(),
            },
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "frozen_backbone": {
                "use_real_qwen": bool(cfg.use_real_qwen),
                "vl_model_id": str(cfg.vl_model_id),
                "hf_cache_dir": str(cfg.hf_cache_dir),
            },
        }

    t_train0 = time.perf_counter()
    for update in range(1, cfg.total_updates + 1):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []
        rollout_xy_penalties: List[float] = []
        rollout_z_penalties: List[float] = []
        rollout_token_successes: List[float] = []
        rollout_token_rewards: List[float] = []

        for _ in range(cfg.rollout_steps):
            obs_t = _obs_batch_to_tensor(obs_np, device, text_context=text_context)
            with torch.no_grad():
                action, logp, value = model.act(obs_t)
            next_obs, reward, terminated, truncated, info = train_env.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf.append(obs_t)
            act_buf.append(action.detach())
            logp_buf.append(logp.detach())
            rew_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            val_buf.append(value.detach())
            done_buf.append(torch.tensor(done, dtype=torch.float32, device=device))

            xy = info.get("xy_penalty", 0.0)
            z = info.get("z_penalty", 0.0)
            ts = info.get("token_success", False)
            tr = info.get("token_reward", 0.0)
            rollout_xy_penalties.append(_to_float_stat(xy))
            rollout_z_penalties.append(_to_float_stat(z))
            rollout_token_successes.append(
                float(np.mean(ts.astype(np.float32))) if isinstance(ts, np.ndarray) else (1.0 if bool(ts) else 0.0)
            )
            rollout_token_rewards.append(_to_float_stat(tr))

            obs_np = next_obs
            text_context = _extract_text_context_batch(info, num_envs)

        with torch.no_grad():
            last_value = model.act(
                _obs_batch_to_tensor(obs_np, device, text_context=text_context)
            )[2].detach()

        rewards = torch.stack(rew_buf, dim=0)  # [T, N]
        values = torch.stack(val_buf, dim=0)  # [T, N]
        dones = torch.stack(done_buf, dim=0)  # [T, N]
        advantages, returns = _compute_gae(
            rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        obs_img = torch.cat([o["image"] for o in obs_buf], dim=0)
        obs_text: List[str] = []
        for o in obs_buf:
            obs_text.extend(o["text_context"])
        actions = torch.cat(act_buf, dim=0)
        old_logp = torch.cat(logp_buf, dim=0).detach()

        N = actions.shape[0]
        inds = np.arange(N)
        last_policy_loss_t = torch.zeros((), device=device)
        last_value_loss_t = torch.zeros((), device=device)
        last_entropy_loss_t = torch.zeros((), device=device)
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, N, cfg.minibatch_size):
                mb = inds[start : start + cfg.minibatch_size]
                mb_obs = {
                    "image": obs_img[mb],
                    "text_context": [obs_text[int(i)] for i in mb.tolist()],
                }
                mb_actions = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    new_logp, entropy, new_value = model.evaluate_actions(mb_obs, mb_actions)
                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss
                last_policy_loss_t = policy_loss.detach()
                last_value_loss_t = value_loss.detach()
                last_entropy_loss_t = entropy_loss.detach()

                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        elapsed_s = time.perf_counter() - t_train0
        updates_per_min = (update / elapsed_s) * 60.0 if elapsed_s > 0 else 0.0
        heartbeat = {
            "update": int(update),
            "elapsed_seconds": float(elapsed_s),
            "updates_per_min": float(updates_per_min),
            "policy_loss": float(last_policy_loss_t.cpu().item()),
            "value_loss": float(last_value_loss_t.cpu().item()),
            "entropy_loss": float(last_entropy_loss_t.cpu().item()),
            "rollout_mean_xy_penalty": float(np.mean(rollout_xy_penalties)) if rollout_xy_penalties else 0.0,
            "rollout_mean_z_penalty": float(np.mean(rollout_z_penalties)) if rollout_z_penalties else 0.0,
            "rollout_token_success_rate": float(np.mean(rollout_token_successes)) if rollout_token_successes else 0.0,
            "rollout_mean_token_reward": float(np.mean(rollout_token_rewards)) if rollout_token_rewards else 0.0,
        }
        progress_path.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")

        if update % max(1, int(cfg.log_every_updates)) == 0 or update == 1:
            print(
                f"[train] update={update} upm={heartbeat['updates_per_min']:.2f} "
                f"pi={heartbeat['policy_loss']:.4f} v={heartbeat['value_loss']:.4f} "
                f"ent={heartbeat['entropy_loss']:.4f} "
                f"tok_succ={heartbeat['rollout_token_success_rate']:.3f}"
            )

        if update % max(1, int(cfg.checkpoint_every_updates)) == 0 or update == 1:
            torch.save(_checkpoint_payload(update), out_dir / f"checkpoint_{update:06d}.pt")

        if update % cfg.eval_every_updates == 0 or update == 1:
            eval_metrics = evaluate_policy(eval_env, model, device, episodes=int(cfg.eval_episodes))
            row = {
                "update": update,
                **eval_metrics,
                "rollout_mean_xy_penalty": float(np.mean(rollout_xy_penalties)) if rollout_xy_penalties else 0.0,
                "rollout_mean_z_penalty": float(np.mean(rollout_z_penalties)) if rollout_z_penalties else 0.0,
                "rollout_token_success_rate": float(np.mean(rollout_token_successes)) if rollout_token_successes else 0.0,
                "rollout_mean_token_reward": float(np.mean(rollout_token_rewards)) if rollout_token_rewards else 0.0,
            }
            metrics.append(row)
            print(
                f"[eval] update={update} "
                f"err={row['eval_mean_final_error_m']:.4f} "
                f"succ={row['eval_success_rate']:.3f} "
                f"len={row['eval_mean_episode_len']:.2f} "
                f"envs={num_envs} "
                f"xy_pen={row['rollout_mean_xy_penalty']:.3f} "
                f"z_pen={row['rollout_mean_z_penalty']:.3f} "
                f"tok_succ={row['rollout_token_success_rate']:.3f}"
            )
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_env.close()


def _load_config(path: Path) -> PPOConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PPOConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tool-position PPO module.")
    parser.add_argument("--config", type=Path, required=True, help="Path to tool_position_ppo.yaml")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

