"""Self-attention waypoint trajectory model with timestep-conditioned RMSNorm."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.action_expert.xyz_normalization import denormalize_xyz_torch


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
        scale_shift = self.to_scale_shift(t_emb).unsqueeze(1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (h * rms) * (1.0 + scale) + shift


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, ffn_multiplier: int):
        super().__init__()
        self.self_ln = nn.LayerNorm(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
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
        t_emb: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_self = self.self_ln(h)
        x, _ = self.self_attn(
            h_self,
            h_self,
            h_self,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = h + x
        h = self.adaptive_norm(h, t_emb)
        h_ffn = self.ffn_ln(h)
        h = h + self.ffn(h_ffn)
        return h


class ActionTrajectoryModel(nn.Module):
    """Token layout (S_max=6); noisy is always last non-padded position.

    has_material=True,  has_destination=True:  [instr, tool, material, destination, table, noisy]
    has_material=False, has_destination=True:  [instr, tool, destination, table, noisy, PAD]
    has_material=True,  has_destination=False: [instr, tool, material, table, noisy, PAD]
    has_material=False, has_destination=False: [instr, tool, table, noisy, PAD, PAD]
    """

    NUM_WAYPOINTS = 15
    WAYPOINT_DIM = 9
    ACTION_DIM = 135
    S_MAX = 6

    def __init__(
        self,
        *,
        d_clip: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        ffn_multiplier: int = 4,
        pos_norm_denom: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_clip = int(d_clip)
        self.pos_norm_denom = float(pos_norm_denom)

        self.label_proj = nn.Linear(self.d_clip, self.d_model)
        self.instr_proj = nn.Linear(self.d_clip, self.d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.normal_mlp = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.surface_dir_mlp = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.action_in_proj = nn.Linear(self.ACTION_DIM, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttnBlock(self.d_model, num_heads, dropout, ffn_multiplier)
                for _ in range(int(num_layers))
            ]
        )
        self.action_out_proj = nn.Linear(self.d_model, self.ACTION_DIM)

    def _ref_token(
        self,
        clip_emb: torch.Tensor,
        xyz_norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.label_proj(clip_emb.float()) + self.pos_mlp(
            xyz_norm.float() / self.pos_norm_denom
        )

    def build_sequence(
        self,
        *,
        instr_clip: torch.Tensor,
        tool_clip: torch.Tensor,
        material_clip: torch.Tensor,
        destination_clip: torch.Tensor,
        table_clip: torch.Tensor,
        tool_contact_xyz_norm: torch.Tensor,
        tool_normal: torch.Tensor,
        tool_surface_dir: torch.Tensor,
        material_xyz_norm: torch.Tensor,
        destination_xyz_norm: torch.Tensor,
        table_xyz_norm: torch.Tensor,
        xt: torch.Tensor,
        has_material: torch.Tensor,
        has_destination: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns h [B, 6, D], key_padding_mask [B, 6] (True = pad, ignore)."""
        bsz = instr_clip.shape[0]
        device = instr_clip.device

        h_instr = self.instr_proj(instr_clip.float()).unsqueeze(1)
        h_tool = (
            self.label_proj(tool_clip.float())
            + self.pos_mlp(tool_contact_xyz_norm.float() / self.pos_norm_denom)
            + self.normal_mlp(tool_normal.float())
            + self.surface_dir_mlp(tool_surface_dir.float())
        ).unsqueeze(1)
        h_material = self._ref_token(material_clip, material_xyz_norm).unsqueeze(1)
        h_destination = self._ref_token(destination_clip, destination_xyz_norm).unsqueeze(1)
        h_table = self._ref_token(table_clip, table_xyz_norm).unsqueeze(1)
        h_noise = self.action_in_proj(xt.float()).unsqueeze(1)

        h_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []

        dt = h_instr.dtype
        pad = torch.zeros(1, self.d_model, device=device, dtype=dt)
        for i in range(bsz):
            hm = bool(has_material[i].item())
            hd = bool(has_destination[i].item())
            if hm and hd:
                hi = torch.cat(
                    [
                        h_instr[i],
                        h_tool[i],
                        h_material[i],
                        h_destination[i],
                        h_table[i],
                        h_noise[i],
                    ],
                    dim=0,
                )
                m = torch.zeros(self.S_MAX, device=device, dtype=torch.bool)
            elif hd and not hm:
                hi = torch.cat(
                    [h_instr[i], h_tool[i], h_destination[i], h_table[i], h_noise[i], pad], dim=0
                )
                m = torch.tensor([False, False, False, False, False, True], device=device)
            elif hm and not hd:
                hi = torch.cat(
                    [h_instr[i], h_tool[i], h_material[i], h_table[i], h_noise[i], pad], dim=0
                )
                m = torch.tensor([False, False, False, False, False, True], device=device)
            else:
                hi = torch.cat([h_instr[i], h_tool[i], h_table[i], h_noise[i], pad, pad], dim=0)
                m = torch.tensor([False, False, False, False, True, True], device=device)
            h_list.append(hi)
            mask_list.append(m)

        h = torch.stack(h_list, dim=0)
        key_padding_mask = torch.stack(mask_list, dim=0)
        return h, key_padding_mask

    def forward(
        self,
        *,
        instr_clip: torch.Tensor,
        tool_clip: torch.Tensor,
        material_clip: torch.Tensor,
        destination_clip: torch.Tensor,
        table_clip: torch.Tensor,
        tool_contact_xyz_norm: torch.Tensor,
        tool_normal: torch.Tensor,
        tool_surface_dir: torch.Tensor,
        material_xyz_norm: torch.Tensor,
        destination_xyz_norm: torch.Tensor,
        table_xyz_norm: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        has_material: torch.Tensor,
        has_destination: torch.Tensor,
    ) -> torch.Tensor:
        h, key_padding_mask = self.build_sequence(
            instr_clip=instr_clip,
            tool_clip=tool_clip,
            material_clip=material_clip,
            destination_clip=destination_clip,
            table_clip=table_clip,
            tool_contact_xyz_norm=tool_contact_xyz_norm,
            tool_normal=tool_normal,
            tool_surface_dir=tool_surface_dir,
            material_xyz_norm=material_xyz_norm,
            destination_xyz_norm=destination_xyz_norm,
            table_xyz_norm=table_xyz_norm,
            xt=xt,
            has_material=has_material,
            has_destination=has_destination,
        )
        t_emb = self.time_mlp(sinusoidal_time_embedding(t.float(), self.d_model))
        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask=key_padding_mask)

        bsz = h.shape[0]
        seq_lengths = (~key_padding_mask).sum(dim=1)
        last_idx = seq_lengths - 1
        h_action = h[torch.arange(bsz, device=h.device), last_idx]
        return self.action_out_proj(h_action)

    @staticmethod
    def postprocess_waypoints(
        pred_norm: torch.Tensor,
        xyz_mean: torch.Tensor,
        xyz_std: torch.Tensor,
        norm_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference postprocess for [B, ACTION_DIM] normalized predictions.

        Returns contact_world, normal, surface_dir each [B, 6, 3].
        """
        wp = pred_norm.view(-1, ActionTrajectoryModel.NUM_WAYPOINTS, ActionTrajectoryModel.WAYPOINT_DIM)
        contact_world = denormalize_xyz_torch(wp[..., 0:3], xyz_mean, xyz_std, float(norm_eps))
        normal = F.normalize(wp[..., 3:6], dim=-1, eps=1e-6)
        sd_raw = wp[..., 6:9]
        sd_perp = sd_raw - (sd_raw * normal).sum(dim=-1, keepdim=True) * normal
        surface_dir = F.normalize(sd_perp, dim=-1, eps=1e-6)
        return contact_world, normal, surface_dir
