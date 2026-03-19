from __future__ import annotations

import torch
from torch import nn

from transformed_transformer.models.encoder import TinyEncoderBlock


class CausalTokenEmbeddings(nn.Module):
    """Minimal token + position embeddings for decoder-only language models."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return hidden


class MiniSLMBackbone(nn.Module):
    """
    Small decoder-only stack intended as a clean extension path toward miniSLM demos.

    This model uses causal masking over the same interchangeable attention backends
    already exposed elsewhere in the package.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        attention_factory: callable,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embeddings = CausalTokenEmbeddings(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model)
        self.layers = nn.ModuleList(
            [
                TinyEncoderBlock(
                    d_model=d_model,
                    ff_dim=ff_dim,
                    attention=attention_factory(),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def extra_loss(self) -> torch.Tensor:
        losses = [layer.attention.regularization_loss() for layer in self.layers]
        if not losses:
            return torch.tensor(0.0, device=self.final_norm.weight.device)
        return torch.stack(losses).sum()

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embeddings(input_ids)
        causal_mask = self._causal_mask(seq_len=input_ids.shape[1], device=input_ids.device)
        stats: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden, layer_stats = layer(hidden, attention_mask=causal_mask)
            for key, value in layer_stats.items():
                stats[f"layer_{layer_idx}_{key}"] = value
        hidden = self.final_norm(hidden)
        return hidden, stats


class MiniSLMForCausalLM(nn.Module):
    """Mini decoder-only causal LM head for next-token prediction demos."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        ff_dim: int,
        num_layers: int,
        attention_factory: callable,
    ) -> None:
        super().__init__()
        self.backbone = MiniSLMBackbone(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ff_dim=ff_dim,
            num_layers=num_layers,
            attention_factory=attention_factory,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embeddings.token_embeddings.weight

    def extra_loss(self) -> torch.Tensor:
        return self.backbone.extra_loss()

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden, stats = self.backbone(input_ids=input_ids)
        logits = self.lm_head(hidden)
        return logits, stats
