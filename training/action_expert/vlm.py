"""PaliGemma context encoder with LoRA adapters for action-expert training."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


class PaliGemmaContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        enable_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "PaliGemmaContextEncoder requires transformers (PaliGemma) and peft."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            token=True,
        )
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            token=True,
        ).to(device)

        # Freeze everything first.
        for p in base_model.parameters():
            p.requires_grad = False

        # SigLIP vision encoder fully frozen.
        if hasattr(base_model, "vision_tower"):
            for p in base_model.vision_tower.parameters():
                p.requires_grad = False

        # Language attention LoRA on W_q/W_v only.
        lora_cfg = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(base_model, lora_cfg)
        # Enforce spec: trainable LoRA only on language attention W_q/W_v.
        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                continue
            if ("q_proj" not in name and "v_proj" not in name) or ("vision_tower" in name):
                p.requires_grad = False
        if bool(enable_gradient_checkpointing):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine PaliGemma hidden_size")
        self.d_model = int(hidden_size)
        self.device = device

        self.label_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        ).to(device=device, dtype=torch.float32)
        self.label_head.train()
        for p in self.label_head.parameters():
            p.requires_grad = True
        self._label_debug_print_budget = 400

    def _embed_tokens_layer(self) -> nn.Embedding:
        # Common path for Gemma/PaliGemma language model embeddings.
        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb
        language_model = getattr(self.model, "language_model", None)
        if language_model is None and hasattr(self.model, "base_model"):
            language_model = getattr(self.model.base_model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Could not locate language_model for PaliGemma")

        if hasattr(language_model, "model") and hasattr(language_model.model, "embed_tokens"):
            return language_model.model.embed_tokens
        if hasattr(language_model, "get_input_embeddings"):
            emb = language_model.get_input_embeddings()
            if emb is not None:
                return emb
        raise RuntimeError("Could not locate token embedding table in PaliGemma language model")

    def lora_parameters(self) -> List[nn.Parameter]:
        # PEFT marks trainable params; include only LoRA params.
        params: List[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and "lora_" in name:
                params.append(p)
        return params

    def label_projection_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.label_head.parameters() if p.requires_grad]

    def format_text_input(self, system_prompt: str, instruction: str, object_labels: Sequence[str]) -> str:
        labels_block = ", ".join(str(lbl) for lbl in object_labels)
        # PaliGemmaProcessor expects an `<image>` placeholder when passing paired text+images.
        return (
            "<image>\n"
            f"{system_prompt.strip()}\n\n"
            f"Instruction: {instruction.strip()}\n"
            f"Objects: {labels_block}"
        )

    def forward_context(
        self,
        *,
        images_uint8: torch.Tensor,
        system_prompts: Sequence[str],
        instructions: Sequence[str],
        object_labels: Sequence[Sequence[str]],
    ) -> dict[str, torch.Tensor]:
        if len(system_prompts) != images_uint8.shape[0]:
            raise ValueError("system_prompts length must match batch size")
        if len(instructions) != images_uint8.shape[0]:
            raise ValueError("instructions length must match batch size")
        if len(object_labels) != images_uint8.shape[0]:
            raise ValueError("object_labels length must match batch size")

        text_payloads: List[str] = []
        imgs: List[np.ndarray] = []
        for i in range(images_uint8.shape[0]):
            text_payloads.append(
                self.format_text_input(system_prompts[i], instructions[i], object_labels[i])
            )
            imgs.append(images_uint8[i].detach().cpu().numpy().astype(np.uint8))

        model_inputs = self.processor(
            text=text_payloads,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        # Processor may attach `labels`; PaliGemma then runs full-vocab LM loss and spikes VRAM.
        # Context encoding only needs hidden states, not supervised LM loss.
        model_inputs.pop("labels", None)

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            last_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("PaliGemma forward did not return hidden states")
        return {
            "context": last_hidden.float(),
            "attention_mask": model_inputs["attention_mask"],
        }

    def embed_labels(self, batch_labels: Sequence[Sequence[str]]) -> List[torch.Tensor]:
        """Return one [num_keypoints, d_model] tensor per batch item."""
        embed_table = self._embed_tokens_layer()
        out: List[torch.Tensor] = []
        for labels in batch_labels:
            rows: List[torch.Tensor] = []
            for label in labels:
                ids = self.processor.tokenizer(
                    str(label),
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(self.device)
                raw = embed_table(ids).squeeze(0)  # [n_tok, d_model]
                pooled = raw.mean(dim=0)
                label_emb = self.label_head(pooled.float())
                if self._label_debug_print_budget > 0:
                    print(
                        f"[label_debug] label_emb_mean={label_emb.mean().item():.4e} "
                        f"label_emb_std={label_emb.std().item():.4e}"
                    )
                    self._label_debug_print_budget -= 1
                rows.append(label_emb)
            if rows:
                out.append(torch.stack(rows, dim=0))
            else:
                out.append(torch.zeros(0, self.d_model, device=self.device))
        return out

