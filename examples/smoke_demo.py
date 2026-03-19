from transformed_transformer import ModelConfig, build_tiny_classifier
from transformed_transformer.utils import make_synthetic_classification_data, set_seed, train_one_step

import torch


def run_demo(attention_type: str) -> None:
    set_seed(0)
    config = ModelConfig(
        vocab_size=32,
        seq_len=12,
        d_model=32,
        ff_dim=64,
        num_classes=2,
        attention_type=attention_type,
        sparse_k=4,
    )
    model = build_tiny_classifier(config)
    x, y = make_synthetic_classification_data(n_samples=8, seq_len=config.seq_len, vocab_size=config.vocab_size)
    logits, stats = model(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step_info = train_one_step(model, x, y, optimizer)

    print(f"backend={attention_type}")
    print("logits_shape=", tuple(logits.shape))
    print("stats_keys=", sorted(stats.keys()))
    print("train_step=", step_info)
    print("-" * 60)


if __name__ == "__main__":
    for name in ["standard", "transformed_dense", "transformed_sparse"]:
        run_demo(name)
