"""Parameter-Efficient Fine-Tuning (PEFT) Wrapper for MedSAM.

Implements all PEFT variants for Experiments E04 through E08:
- E04: Decoder-only PEFT (mask decoder tuned, image encoder frozen)
- E05: LoRA on ViT attention projections (r=16, alpha=32)
- E06: Standard Bottleneck Adapter (r=16, constant gamma=1.0)
- E07: Contrast-Gated Adapter (CG-Adapter, r=16, gamma=g(c_prompt))
- E08: Causal Ablations (constant gate, shuffled contrast, sign-flipped contrast)
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adapters import LoRALinear, BottleneckAdapter, ContrastGatingMLP
from src.models.medsam_wrapper import MockMedSAMModel


class PEFTMedSAM(nn.Module):
    """Unified PEFT MedSAM model supporting Decoder-only, LoRA, and CG-Adapter architectures."""

    VALID_MODES = ["decoder_only", "lora", "standard_adapter", "cg_adapter", "ablation"]

    def __init__(
        self,
        base_medsam: Optional[nn.Module] = None,
        mode: str = "cg_adapter",
        bottleneck_rank: int = 16,
        lora_alpha: float = 32.0,
        ablation_type: Optional[str] = None,  # "constant", "shuffled", "sign_flipped"
        device: Optional[torch.device] = None
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")

        self.mode = mode
        self.rank = bottleneck_rank
        self.lora_alpha = lora_alpha
        self.ablation_type = ablation_type
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if base_medsam is None:
            self.base_model = MockMedSAMModel()
            self.is_mock = True
        else:
            self.base_model = base_medsam
            self.is_mock = False

        self._freeze_backbone()
        self.adapters = nn.ModuleList()
        self.lora_layers = nn.ModuleList()

        if not self.is_mock:
            self._inject_peft_modules()
        else:
            # Mock setup with trainable parameters to reflect parameter budgets
            if mode == "cg_adapter":
                self.mock_adapter = BottleneckAdapter(embed_dim=768, bottleneck_rank=bottleneck_rank, gated=True)
            elif mode == "standard_adapter":
                self.mock_adapter = BottleneckAdapter(embed_dim=768, bottleneck_rank=bottleneck_rank, gated=False)
            elif mode == "lora":
                dummy_lin = nn.Linear(768, 768)
                self.mock_lora = LoRALinear(dummy_lin, r=bottleneck_rank, lora_alpha=lora_alpha)

    def _freeze_backbone(self):
        """Freeze the entire backbone initially."""
        for param in self.base_model.parameters():
            param.requires_grad = False

        if not self.is_mock:
            # Mask decoder is trainable in all PEFT configurations
            for param in self.base_model.mask_decoder.parameters():
                param.requires_grad = True

    def _inject_peft_modules(self):
        """Inject adapters or LoRA layers into the ViT backbone."""
        vit_blocks = self.base_model.image_encoder.blocks

        if self.mode == "lora":
            for block in vit_blocks:
                # Replace q_proj and v_proj in multi-head self-attention
                q_proj = block.attn.qkv
                # Wrap linear
                lora_qkv = LoRALinear(q_proj, r=self.rank, lora_alpha=self.lora_alpha)
                block.attn.qkv = lora_qkv
                self.lora_layers.append(lora_qkv)

        elif self.mode in ["standard_adapter", "cg_adapter", "ablation"]:
            is_gated = (self.mode == "cg_adapter") or (self.mode == "ablation" and self.ablation_type != "constant")
            for block in vit_blocks:
                adapter = BottleneckAdapter(
                    embed_dim=768,
                    bottleneck_rank=self.rank,
                    gated=is_gated
                )
                self.adapters.append(adapter)

    def count_parameters(self) -> Dict[str, Any]:
        """Calculate total, trainable, and frozen parameter counts and percentages."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        fraction_pct = (trainable / float(total)) * 100.0 if total > 0 else 0.0

        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(frozen),
            "trainable_fraction_percent": round(fraction_pct, 4)
        }

    def forward(
        self,
        image_tensor: torch.Tensor,
        box_tensor: torch.Tensor,
        c_prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass generating mask logits.

        Args:
            image_tensor: (B, 3, 1024, 1024)
            box_tensor: (B, 4) in 1024 coordinates
            c_prompt: (B, 1) normalized scalar contrast feature
        """
        # Handle ablation controls
        if self.mode == "ablation":
            if self.ablation_type == "constant":
                c_prompt = torch.zeros_like(c_prompt) if c_prompt is not None else None
            elif self.ablation_type == "shuffled":
                if c_prompt is not None:
                    perm = torch.randperm(c_prompt.size(0))
                    c_prompt = c_prompt[perm]
            elif self.ablation_type == "sign_flipped":
                if c_prompt is not None:
                    c_prompt = -c_prompt

        if self.is_mock:
            logits = self.base_model(image_tensor, box_tensor)
            if hasattr(self, "mock_adapter"):
                dummy_feat = torch.zeros((image_tensor.size(0), 1, 768), device=image_tensor.device, dtype=logits.dtype)
                g = self.mock_adapter(dummy_feat, c_prompt=c_prompt).sum()
                logits = logits + 0.0 * g
            elif hasattr(self, "mock_lora"):
                dummy_feat = torch.zeros((image_tensor.size(0), 1, 768), device=image_tensor.device, dtype=logits.dtype)
                g = self.mock_lora(dummy_feat).sum()
                logits = logits + 0.0 * g
            return logits

        # Real MedSAM forward pass with PEFT adapters
        if self.mode in ["standard_adapter", "cg_adapter", "ablation"]:
            # Forward through ViT blocks with adapter residuals
            x = self.base_model.image_encoder.patch_embed(image_tensor)
            if self.base_model.image_encoder.pos_embed is not None:
                x = x + self.base_model.image_encoder.pos_embed

            for idx, block in enumerate(self.base_model.image_encoder.blocks):
                x = block(x)
                if idx < len(self.adapters):
                    x = self.adapters[idx](x, c_prompt=c_prompt)
            image_embeddings = self.base_model.image_encoder.neck(x.permute(0, 3, 1, 2))
        else:
            image_embeddings = self.base_model.image_encoder(image_tensor)

        sparse_emb, dense_emb = self.base_model.prompt_encoder(
            points=None,
            boxes=box_tensor.unsqueeze(1),
            masks=None
        )
        low_res_masks, _ = self.base_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.base_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False
        )
        return low_res_masks
