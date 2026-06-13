"""Model definitions for tool-position behavior cloning."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


class QwenBackboneStub(nn.Module):
    """Fallback image projection backbone when real Qwen is disabled."""

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
        del text_context
        del chunk_size
        x = image_uint8.float() / 255.0
        x = x.reshape(x.shape[0], -1)
        return self.img_proj(x)


class FrozenQwenVLBackbone(nn.Module):
    """Frozen Qwen2.5-VL encoder returning one embedding per sample."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        unfreeze_last_n_layers: int = 8,
        enable_gradient_checkpointing: bool = True,
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
        for p in self.model.parameters():
            p.requires_grad = False
        self._unfreeze_last_n_layers(int(unfreeze_last_n_layers))
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine hidden_size for Qwen VL model config")
        self.output_dim = int(hidden_size)
        self.device = device
        self._has_trainable_backbone = any(p.requires_grad for p in self.model.parameters())
        if self._has_trainable_backbone and bool(enable_gradient_checkpointing):
            try:
                self.model.model.language_model.gradient_checkpointing_enable()
            except Exception:
                pass
        if not self._has_trainable_backbone:
            self.model.eval()

    def _get_transformer_layers(self) -> list[nn.Module]:
        candidates = []
        if hasattr(self.model, "model"):
            core = self.model.model
            candidates.append(getattr(core, "layers", None))
            if hasattr(core, "language_model"):
                lm2 = core.language_model
                candidates.append(getattr(lm2, "layers", None))
                if hasattr(lm2, "model"):
                    candidates.append(getattr(lm2.model, "layers", None))
        if hasattr(self.model, "model"):
            candidates.append(getattr(self.model.model, "layers", None))
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            candidates.append(getattr(lm, "model", None))
            if hasattr(lm, "model"):
                candidates.append(getattr(lm.model, "layers", None))
            candidates.append(getattr(lm, "layers", None))
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, (nn.ModuleList, list, tuple)):
                return list(cand)
        raise RuntimeError("Could not locate transformer layer stack for Qwen backbone")

    def _unfreeze_last_n_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        layers = self._get_transformer_layers()
        for layer in layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def trainable_backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

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
        use_no_grad = not self._has_trainable_backbone
        with torch.no_grad() if use_no_grad else torch.enable_grad():
            out = self.model.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
            last = out.last_hidden_state
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


class ToolPositionBCRegressor(nn.Module):
    """Frozen VL backbone + MLP regressor to XYZ."""

    def __init__(
        self,
        *,
        use_real_qwen: bool,
        vl_model_id: str,
        device: torch.device,
        hf_cache_dir: str,
        qwen_local_files_only: bool,
        qwen_forward_chunk_size: int,
        unfreeze_last_n_layers: int = 8,
        enable_gradient_checkpointing: bool = True,
        hidden_dims: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        if use_real_qwen:
            self.vl = FrozenQwenVLBackbone(
                model_id=vl_model_id,
                device=device,
                cache_dir=hf_cache_dir,
                local_files_only=qwen_local_files_only,
                unfreeze_last_n_layers=unfreeze_last_n_layers,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
            )
        else:
            self.vl = QwenBackboneStub(output_dim=768)
        self.qwen_forward_chunk_size = int(qwen_forward_chunk_size)

        dims = [int(self.vl.output_dim), *[int(d) for d in hidden_dims], 3]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.xyz_head = nn.Sequential(*layers)

    def forward(self, obs: Dict[str, torch.Tensor | Sequence[str]]) -> torch.Tensor:
        fused = self.vl(
            obs["image"],
            obs.get("text_context"),
            chunk_size=self.qwen_forward_chunk_size,
        )
        return self.xyz_head(fused)

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        if hasattr(self.vl, "trainable_backbone_parameters"):
            return self.vl.trainable_backbone_parameters()
        return []

