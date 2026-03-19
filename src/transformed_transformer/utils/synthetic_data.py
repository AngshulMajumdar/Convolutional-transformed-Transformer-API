from __future__ import annotations

import random

import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_classification_data(
    n_samples: int,
    seq_len: int = 12,
    vocab_size: int = 20,
    noise_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(1, vocab_size, (n_samples, seq_len))
    labels = ((tokens[:, 0] == tokens[:, -1]) | ((tokens[:, 2] + tokens[:, seq_len - 3]) > vocab_size)).long()

    if noise_prob > 0:
        mask = torch.rand(n_samples, seq_len) < noise_prob
        noise = torch.randint(1, vocab_size, (n_samples, seq_len))
        tokens = torch.where(mask, noise, tokens)

    return tokens, labels
