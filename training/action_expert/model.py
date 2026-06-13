"""Action expert transformer model."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] in [0, 1], returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.eps = 1e-6
        self.to_scale_shift = nn.Linear(d_model, d_model * 2)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # h: [B, S, D], t_emb: [B, D]
        scale_shift = self.to_scale_shift(t_emb).unsqueeze(1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (h * rms) * (1.0 + scale) + shift


class ActionExpertBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_multiplier: int):
        super().__init__()
        self.self_ln = nn.LayerNorm(d_model)
        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_out_ln = nn.LayerNorm(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.adaptive_norm = AdaptiveRMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
        )

    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        t_emb: torch.Tensor,
        c_key_padding_mask: torch.Tensor | None = None,
        return_cross_attn_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_self = self.self_ln(h)
        x, _ = self.self_attn(h_self, h_self, h_self, need_weights=False)
        h = h + x
        h_cross = self.cross_ln(h)
        x, _ = self.cross_attn(
            h_cross,
            c,
            c,
            key_padding_mask=c_key_padding_mask,
            need_weights=False,
        )
        x = self.cross_out_ln(x)
        cross_norm = x.norm(dim=-1).mean() if return_cross_attn_norm else None
        h = h + x
        h = self.adaptive_norm(h, t_emb)
        h_ffn = self.ffn_ln(h)
        h = h + self.ffn(h_ffn)
        return h, cross_norm


class ActionExpertModel(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
        pos_norm_denom: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pos_norm_denom = float(pos_norm_denom)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_in_proj = nn.Linear(3, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.context_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(d_model, num_heads, dropout, ffn_multiplier)
                for _ in range(int(num_layers))
            ]
        )
        self.action_out_proj = nn.Linear(d_model, 3)
        self.last_debug_stats: dict[str, float] = {}

    def build_action_tokens(
        self,
        *,
        label_embeddings: Sequence[torch.Tensor],
        keypoint_positions: Sequence[torch.Tensor],
        xt: torch.Tensor,
    ) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for i, pos in enumerate(keypoint_positions):
            pos = pos.to(xt.device).float()
            if pos.numel() == 0:
                kp_tokens = torch.zeros(0, self.d_model, device=xt.device)
            else:
                pos_emb = self.pos_mlp(pos / self.pos_norm_denom)
                kp_tokens = label_embeddings[i].to(xt.device).float() + pos_emb
            noisy = self.action_in_proj(xt[i].unsqueeze(0))
            rows.append(torch.cat([kp_tokens, noisy], dim=0))
        max_len = max((r.shape[0] for r in rows), default=1)
        padded: List[torch.Tensor] = []
        for r in rows:
            if r.shape[0] < max_len:
                pad = torch.zeros(max_len - r.shape[0], self.d_model, device=r.device)
                r = torch.cat([r, pad], dim=0)
            padded.append(r)
        return torch.stack(padded, dim=0)  # [B, S, D]

    def forward(
        self,
        *,
        label_embeddings: Sequence[torch.Tensor],
        keypoint_positions: Sequence[torch.Tensor],
        xt: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
        collect_debug_stats: bool = False,
    ) -> torch.Tensor:
        h = self.build_action_tokens(
            label_embeddings=label_embeddings,
            keypoint_positions=keypoint_positions,
            xt=xt,
        )
        t_emb = self.time_mlp(sinusoidal_time_embedding(t.float(), self.d_model))
        context_normed = self.context_norm(context)
        c_mask = None
        if context_attention_mask is not None:
            c_mask = ~context_attention_mask.bool()
        cross_norms: List[torch.Tensor] = []
        for block in self.blocks:
            h, cross_norm = block(
                h,
                context_normed,
                t_emb,
                c_mask,
                return_cross_attn_norm=bool(collect_debug_stats),
            )
            if cross_norm is not None:
                cross_norms.append(cross_norm)
        if collect_debug_stats and cross_norms:
            cross_norm_t = torch.stack(cross_norms).mean()
            self.last_debug_stats = {
                "cross_attn_out_norm_mean": float(cross_norm_t.detach().cpu().item())
            }
        # Keep prior debug stats when not collecting (e.g. eval/no-grad path),
        # so train telemetry is not spuriously reset to zero around eval steps.
        # Noisy token is always final real token per sample.
        noisy_h = []
        for i, pos in enumerate(keypoint_positions):
            noisy_index = int(pos.shape[0])  # after all keypoint tokens
            noisy_h.append(h[i, noisy_index])
        noisy_h = torch.stack(noisy_h, dim=0)
        return self.action_out_proj(noisy_h)

