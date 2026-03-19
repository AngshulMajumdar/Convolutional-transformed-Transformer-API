from __future__ import annotations

import torch
from torch import nn


def train_one_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits, labels)
    extra = model.extra_loss() if hasattr(model, "extra_loss") else torch.tensor(0.0, device=loss.device)
    total = loss + extra
    total.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "extra_loss": float(extra.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
