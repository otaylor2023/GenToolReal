"""Frozen CLIP text encoder with string dedup cache."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ClipTextEncoder(nn.Module):
    """Frozen CLIP text tower; returns L2-normalized text features [B, d_clip]."""

    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError("ClipTextEncoder requires transformers with CLIP support.") from exc

        self.model_id = str(model_id)
        self._device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
        )
        self.text_model = CLIPTextModel.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
        ).to(device)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.d_clip = int(self.text_model.config.hidden_size)
        self._emb_cache: dict[str, torch.Tensor] = {}
        self._cache_max = 8192

    @property
    def device(self) -> torch.device:
        return self._device

    def _pool(self, outputs, input_ids: torch.Tensor) -> torch.Tensor:
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        eos = input_ids.argmax(dim=-1)
        last = outputs.last_hidden_state
        return last[torch.arange(last.shape[0], device=last.device), eos]

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.d_clip, device=self.device, dtype=torch.float32)

        unique: List[str] = []
        index_map: List[int] = []
        seen: dict[str, int] = {}
        for t in texts:
            s = str(t)
            if s not in seen:
                seen[s] = len(unique)
                unique.append(s)
            index_map.append(seen[s])

        emb_per: list[torch.Tensor | None] = [None] * len(unique)
        pending: List[tuple[int, str]] = []
        for i, s in enumerate(unique):
            if s in self._emb_cache:
                emb_per[i] = self._emb_cache[s].to(self.device)
            else:
                pending.append((i, s))

        if pending:
            batch_s = [p[1] for p in pending]
            inputs = self.tokenizer(
                batch_s,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.text_model(**inputs)
            pooled = self._pool(outputs, inputs["input_ids"]).float()
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            for k, (i, s) in enumerate(pending):
                row = pooled[k].detach()
                if len(self._emb_cache) >= self._cache_max and s not in self._emb_cache:
                    self._emb_cache.pop(next(iter(self._emb_cache)))
                self._emb_cache[s] = row.cpu()
                emb_per[i] = row

        unique_stack = torch.stack([emb_per[i] for i in range(len(unique))], dim=0)
        return unique_stack[index_map]
