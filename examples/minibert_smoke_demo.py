from __future__ import annotations

import torch

from transformed_transformer import ModelConfig, build_mini_bert_classifier, build_mini_bert_mlm
from transformed_transformer.utils import make_synthetic_classification_data, set_seed, train_one_step


def run_classifier_demo(attention_type: str) -> None:
    config = ModelConfig(
        vocab_size=40,
        seq_len=10,
        d_model=32,
        ff_dim=64,
        num_classes=3,
        num_layers=2,
        attention_type=attention_type,
    )
    x, y = make_synthetic_classification_data(n_samples=8, seq_len=config.seq_len, vocab_size=config.vocab_size)
    model = build_mini_bert_classifier(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    info = train_one_step(model, x, y, optimizer)
    print(f"[miniBERT-cls::{attention_type}] loss={info['total_loss']:.4f}")


def run_mlm_demo(attention_type: str) -> None:
    config = ModelConfig(vocab_size=40, seq_len=10, d_model=32, ff_dim=64, num_layers=2, attention_type=attention_type)
    model = build_mini_bert_mlm(config)
    input_ids = torch.randint(0, config.vocab_size, (4, config.seq_len))
    input_ids[:, 0] = 2
    mlm_input = input_ids.clone()
    mlm_input[:, 3] = config.mask_token_id
    logits, stats = model(mlm_input)
    print(f"[miniBERT-mlm::{attention_type}] logits={tuple(logits.shape)} stats={len(stats)}")


if __name__ == "__main__":
    set_seed(0)
    for attention_type in ["standard", "transformed_dense", "transformed_sparse"]:
        run_classifier_demo(attention_type)
        run_mlm_demo(attention_type)
