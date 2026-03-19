from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseAttention, TensorStats


class ConvTransform1D(nn.Module):
    """DCTL-inspired local transform over the sequence dimension.

    This module is trained end-to-end with Adam on the downstream transformer
    task loss. It borrows locality/shared parameters from deep convolutional
    transform learning, but is not a paper-faithful alternating solver.
    """

    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.local_kernel = nn.Parameter(torch.randn(kernel_size, d_model) * 0.02)
        self.pointwise = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        radius = self.kernel_size // 2
        y = torch.zeros_like(x)
        for idx, offset in enumerate(range(-radius, radius + 1)):
            shifted = torch.roll(x, shifts=offset, dims=1)
            if offset < 0:
                shifted[:, offset:, :] = 0.0
            elif offset > 0:
                shifted[:, :offset, :] = 0.0
            y = y + shifted * self.local_kernel[idx].view(1, 1, -1)
        y = self.pointwise(y)
        return self.norm(x + torch.tanh(self.gate) * y)

    def regularization_loss(self, reg_weight: float) -> torch.Tensor:
        reg = torch.sum(self.local_kernel.square()) + torch.sum(self.pointwise.weight.square())
        return reg_weight * reg


def _apply_attention_mask(attn_logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return attn_logits
    mask = attention_mask.to(dtype=torch.bool, device=attn_logits.device)
    while mask.dim() < attn_logits.dim():
        mask = mask.unsqueeze(0)
    return attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)


class TransformConvAttention(BaseAttention):
    """Convolutional transform attention for the proposed transformer variant."""

    def __init__(self, d_model: int, kernel_size: int = 3, groups: int = 1, reg_weight: float = 1e-3) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("Current transformed_conv implementation uses a single shared pointwise mix; set conv_groups=1")
        self.d_model = d_model
        self.reg_weight = reg_weight
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_transform = ConvTransform1D(d_model, kernel_size=kernel_size)
        self.k_transform = ConvTransform1D(d_model, kernel_size=kernel_size)
        self.v_transform = ConvTransform1D(d_model, kernel_size=kernel_size)
        self.o = nn.Linear(d_model, d_model)

    def regularization_loss(self) -> torch.Tensor:
        return (
            self.q_transform.regularization_loss(self.reg_weight)
            + self.k_transform.regularization_loss(self.reg_weight)
            + self.v_transform.regularization_loss(self.reg_weight)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, TensorStats]:
        q = self.q_transform(self.q_proj(x))
        k = self.k_transform(self.k_proj(x))
        v = self.v_transform(self.v_proj(x))
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        attn_logits = _apply_attention_mask(attn_logits, attention_mask)
        attn = torch.softmax(attn_logits, dim=-1)
        y = self.o(torch.matmul(attn, v))
        stats: TensorStats = {
            "reg_loss": self.regularization_loss().detach(),
            "attn_mean": attn.mean().detach(),
            "attn_std": attn.std().detach(),
            "q_gate": torch.tanh(self.q_transform.gate).detach(),
            "k_gate": torch.tanh(self.k_transform.gate).detach(),
            "v_gate": torch.tanh(self.v_transform.gate).detach(),
        }
        return y, stats
