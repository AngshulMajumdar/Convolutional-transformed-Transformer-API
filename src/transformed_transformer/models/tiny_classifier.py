from __future__ import annotations

import torch
from torch import nn

from transformed_transformer.attention.base import BaseAttention
from transformed_transformer.models.encoder import TinyEncoderBlock


class TinyEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        ff_dim: int,
        num_classes: int,
        attention: BaseAttention,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.encoder = TinyEncoderBlock(d_model=d_model, ff_dim=ff_dim, attention=attention)
        self.classifier = nn.Linear(d_model, num_classes)

    def extra_loss(self) -> torch.Tensor:
        return self.encoder.attention.regularization_loss()

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.embedding(tokens) + self.positional[:, : tokens.shape[1], :]
        hidden, stats = self.encoder(hidden)
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, stats
