from __future__ import annotations

import torch
import torch.nn.functional as F

from transformed_transformer import ModelConfig, build_mini_bert_classifier
from transformed_transformer.utils import make_synthetic_classification_data, set_seed


def run_demo(attention_type: str) -> None:
    config = ModelConfig(
        vocab_size=64,
        seq_len=16,
        d_model=48,
        ff_dim=96,
        num_classes=2,
        num_layers=2,
        attention_type=attention_type,
        conv_kernel_size=3,
        reg_weight=1e-4,
    )
    x, y = make_synthetic_classification_data(
        n_samples=16,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        noise_prob=0.0,
    )
    token_type_ids = torch.zeros_like(x)
    model = build_mini_bert_classifier(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits, _ = model(x, token_type_ids=token_type_ids)
    loss = F.cross_entropy(logits, y) + model.extra_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    print(f"[miniBERT-demo::{attention_type}] loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    set_seed(0)
    for attention_type in ["standard", "transformed_conv"]:
        run_demo(attention_type)
