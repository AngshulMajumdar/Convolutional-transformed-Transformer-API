from __future__ import annotations

import torch
import torch.nn.functional as F

from transformed_transformer import ModelConfig, build_mini_slm


def run_demo(attention_type: str) -> None:
    config = ModelConfig(
        vocab_size=40,
        seq_len=12,
        d_model=24,
        ff_dim=48,
        num_layers=2,
        attention_type=attention_type,
        sparse_k=4,
    )
    model = build_mini_slm(config)
    batch = torch.randint(0, config.vocab_size, (4, config.seq_len))
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits, stats = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)) + model.extra_loss()
    loss.backward()

    print(f"[miniSLM] backend={attention_type}")
    print(f"  logits_shape={tuple(logits.shape)}")
    print(f"  loss={loss.item():.4f}")
    preview_keys = sorted(stats.keys())[:4]
    print(f"  stats_preview={preview_keys}")


if __name__ == "__main__":
    for backend in ["standard", "transformed_dense", "transformed_sparse"]:
        run_demo(backend)
