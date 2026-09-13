"""Adapter architectures: LoRA, Standard Bottleneck Adapter, and Contrast-Gated Adapter (CG-Adapter).

Implements parameter-efficient fine-tuning (PEFT) modules:
1. LoRALinear: Low-rank adaptation for multi-head attention projections (r=16, alpha=32).
2. BottleneckAdapter: Residual bottleneck adapter (d=768, r=16) with GELU activation.
3. ContrastGatingMLP: 2-layer MLP mapping scalar contrast c_prompt to gate gamma in [0, 2].
4. ContrastGatedAdapter: Synergistic bottleneck adapter modulated by contrast gate gamma.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a frozen Linear layer.

    W = W_0 + (alpha / r) * B @ A
    Where A ~ N(0, 1/r) and B = 0, ensuring identity behavior at initialization.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r)

        # Base frozen linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zero initialization for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        base_out = self.base_linear(x)
        # LoRA branch
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


class ContrastGatingMLP(nn.Module):
    """2-layer MLP gating network mapping scalar contrast feature to residual scale gamma.

    gamma = 2.0 * Sigmoid(W_2 * GELU(W_1 * c_prompt + b_1) + b_2)
    gamma in [0, 2], initialized near 1.0 (identity scale at initialization).
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        # Initialize fc2 near zero so that Sigmoid(0) = 0.5 -> gamma = 2 * 0.5 = 1.0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, c_prompt: torch.Tensor) -> torch.Tensor:
        """Args:

            c_prompt: Tensor of shape (B, 1) or (B,) with normalized scalar contrast.

        Returns:
            gamma: Tensor of shape (B, 1) bounded in [0, 2].
        """
        if c_prompt.ndim == 1:
            c_prompt = c_prompt.unsqueeze(-1)
        h = self.act(self.fc1(c_prompt))
        logits = self.fc2(h)
        gamma = 2.0 * torch.sigmoid(logits)
        return gamma


class BottleneckAdapter(nn.Module):
    """Fixed-rank bottleneck residual adapter (r=16) with optional contrast gating.

    a_l(h) = Dropout(GELU(h @ W_down)) @ W_up
    h' = h + gamma * a_l(h)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        bottleneck_rank: int = 16,
        dropout: float = 0.1,
        gated: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = bottleneck_rank
        self.gated = gated

        self.down_proj = nn.Linear(embed_dim, bottleneck_rank, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.up_proj = nn.Linear(bottleneck_rank, embed_dim, bias=False)

        if gated:
            self.gate = ContrastGatingMLP(hidden_dim=16)
        else:
            self.gate = None

        self.reset_parameters()

    def reset_parameters(self):
        # Near-zero initialization for up_proj to guarantee identity at step 0
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        c_prompt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:

            x: Input feature tensor of shape (B, N, D)
            c_prompt: Optional scalar contrast feature (B, 1) for gating

        Returns:
            x_adapted: Adapted tensor of shape (B, N, D)
        """
        residual = self.dropout(self.act(self.down_proj(x)))
        adapter_out = self.up_proj(residual)

        if self.gated and self.gate is not None:
            if c_prompt is None:
                raise ValueError("Contrast feature c_prompt must be provided when gated=True.")
            gamma = self.gate(c_prompt)  # (B, 1)
            gamma_view = gamma.view(x.size(0), *([1] * (x.ndim - 1)))
            adapter_out = gamma_view * adapter_out

        return x + adapter_out


class ContrastGatedLoRALinear(LoRALinear):
    """Contrast-Gated Low-Rank Adaptation (CG-LoRA) layer.

    W(x) = W_0 x + gamma * (alpha / r) * (B @ A @ x)
    where gamma = 2.0 * Sigmoid(MLP(c_prompt)) modulates the low-rank delta.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        gating_mlp: Optional[ContrastGatingMLP] = None
    ):
        super().__init__(base_linear, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.gate = gating_mlp if gating_mlp is not None else ContrastGatingMLP(hidden_dim=16)
        self.current_c_prompt: Optional[torch.Tensor] = None

    def set_contrast(self, c_prompt: Optional[torch.Tensor]):
        """Set the active prompt contrast tensor for downstream forward passes."""
        self.current_c_prompt = c_prompt

    def forward(self, x: torch.Tensor, c_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T

        c = c_prompt if c_prompt is not None else self.current_c_prompt
        if c is not None and self.gate is not None:
            gamma = self.gate(c)
            gamma_view = gamma.view(x.size(0), *([1] * (x.ndim - 1)))
            return base_out + gamma_view * (self.scaling * lora_out)

        return base_out + self.scaling * lora_out
